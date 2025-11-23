"""
Pytorch dataset that loads from raw data files
"""

import time
from torch.utils.data import Dataset
import os
import torch
import xarray as xr
from omegaconf import DictConfig
import logging
import numpy as np
from dask.distributed import Client
from dask.diagnostics import ProgressBar


class ClimSimFromRawDataset(Dataset):
    def __init__(
        self,
        mode: str,
        dataset_testing_type: str,
        dataset_cfg: DictConfig,
        model: str = None,
        normalisation_stats: dict = None,
        unit_test_specific_methods: bool = False,
        num_workers: int = 1,
    ):
        super().__init__()
        self.mode = mode
        self.dataset_testing_type = dataset_testing_type
        self.dataset_cfg = dataset_cfg
        self.model = model
        self.normalisation_stats = normalisation_stats

        if unit_test_specific_methods:
            # For unit testing, we can skip the full initialization as we want to test specific methods only
            return

        if mode in ["val", "test"] and self.normalisation_stats is None:
            raise ValueError(
                "Normalisation stats must be provided for validation and test modes."
            )

        target_years, target_months = self._select_target_years_months(
            mode, dataset_cfg
        )

        self.input_filenames, self.target_filenames = self._get_dataset_filenames(
            dataset_cfg.base_folder_path,
            target_years,
            target_months,
        )
        logging.info(f"Total files found: {len(self.input_filenames)}")

        self.sampled_input_filenames, self.sampled_target_filenames = (
            self._sample_filenames(
                self.input_filenames,
                self.target_filenames,
                sample_rate=dataset_cfg.dataset_testing_sample_rates[
                    dataset_testing_type
                ],
            )
        )
        logging.info(
            f"Files after sampling ({dataset_testing_type}): {len(self.sampled_input_filenames)}"
        )

        if num_workers > 1:
            client = Client(
                n_workers=num_workers, threads_per_worker=1, memory_limit="8GB"
            )
            parallel = True
        else:
            parallel = False

        start_time = time.time()
        input_ds, target_ds = self._combine_datasets(
            self.sampled_input_filenames,
            self.sampled_target_filenames,
            v1_inputs=dataset_cfg.v1_inputs,
            v1_targets=dataset_cfg.v1_targets,
            parallel=parallel,
        )
        logging.info(f"Dataset combination time: {time.time() - start_time} seconds")

        logging.info(f"Combined dataset samples: {input_ds.sizes['sample']}")

        start_time = time.time()
        normalised_input_ds, normalised_target_ds, self.normalisation_stats = (
            self._normalise_datasets(
                self.normalisation_stats,
                input_ds,
                target_ds,
                output_scale_file_path=dataset_cfg.output_scale_file_path,
                v1_targets=dataset_cfg.v1_targets,
            )
        )
        logging.info(f"Normalised data in {time.time() - start_time} seconds")

        start_time = time.time()
        self.input, self.target = self._prepare_data(
            self.model,
            normalised_input_ds,
            normalised_target_ds,
            v1_inputs=dataset_cfg.v1_inputs,
            v1_targets=dataset_cfg.v1_targets,
        )
        logging.info(f"Prepared data in {time.time() - start_time} seconds")

    def _select_target_years_months(self, mode: str, dataset_cfg: DictConfig) -> tuple:
        """
        Selects the target years and months based on the mode (train/val/test) and dataset configuration.

        Args:
            mode: Mode of the dataset ('train', 'val', 'test').
            dataset_cfg: Dataset configuration containing year and month ranges.
        Returns:
            Tuple of (target_years, target_months).
        """
        assert hasattr(
            dataset_cfg, "group_method"
        ), "Dataset config must have 'group_method'"

        if dataset_cfg.group_method is False:
            target_years = dataset_cfg.target_years
            target_months = dataset_cfg.target_months
            return target_years, target_months

        assert hasattr(
            dataset_cfg, dataset_cfg.group_method
        ), f"Dataset config must have grouping method '{dataset_cfg.group_method}' defined."

        group_method_cfg = dataset_cfg[dataset_cfg.group_method]
        target_years = dataset_cfg.target_years
        if mode == "train":
            target_months = group_method_cfg.groups[int(group_method_cfg.target_group)]
        elif mode == "val":
            raise NotImplementedError("Validation mode is not implemented yet.")
        elif mode == "test":
            target_months = group_method_cfg.test_group

        return target_years, target_months

    def _get_dataset_filenames(
        self,
        base_folder_path: str,
        target_years: list,
        target_months: list,
        input_regex: str = "E3SM-MMF.mli",
        target_regex: str = "E3SM-MMF.mlo",
    ) -> tuple:
        """
        Extracts filenames for the dataset, based on requirements given in the dataset config, ensuring both input and target files are found.

        Args:
            base_folder_path: Base folder path where raw data files are stored.
            target_years: List of years to include in the dataset.
            target_months: List of months to include in the dataset.
            input_regex: Regex pattern to identify input files.
            target_regex: Regex pattern to identify target files.

        Returns:
            Tuple of (sorted) filenames for the dataset.
        """
        input_filelist = []
        target_filelist = []
        for year in target_years:
            for month in target_months:
                folder_path = os.path.join(base_folder_path, f"{year}-{month}")
                if not os.path.isdir(folder_path):
                    logging.warning(
                        f"Folder path does not exist or is not a directory: {folder_path}. Skipping."
                    )
                    continue
                for filename in os.listdir(folder_path):
                    if input_regex in filename:
                        input_filelist.append(os.path.join(folder_path, filename))
                    elif target_regex in filename:
                        target_filelist.append(os.path.join(folder_path, filename))

        input_filelist, target_filelist = sorted(input_filelist), sorted(
            target_filelist
        )
        if check_matching_files(input_filelist, target_filelist) is False:
            input_filelist, target_filelist = prune_mismached_files(
                input_filelist, target_filelist
            )

        return input_filelist, target_filelist

    def _sample_filenames(
        self,
        input_filelist: list,
        target_filelist: list,
        sample_rate: int,
        start_index: int = 0,
        end_index: int = -1,
    ) -> tuple:
        """
        Downsamples from the full list of filenames based on the a specified scale factor (stride).

        args:
            input_filelist: List of input filenames.
            target_filelist: List of target filenames.
            sample_rate: Integer scale factor for downsampling (e.g., 2 means every 2nd file is kept).
            start_index: Starting index for sampling.
            end_index: Ending index for sampling.

        Returns:
            Tuple of (sorted) sampled filenames for the dataset.
        """
        end_index = end_index if end_index != -1 else len(input_filelist)
        sampled_input_filelist = sorted(input_filelist)[
            start_index:end_index:sample_rate
        ]
        sampled_target_filelist = sorted(target_filelist)[
            start_index:end_index:sample_rate
        ]
        assert check_matching_files(sampled_input_filelist, sampled_target_filelist)
        return sampled_input_filelist, sampled_target_filelist

    def _combine_datasets(
        self,
        sampled_input_filenames: list,
        sampled_target_filenames: list,
        v1_inputs: list,
        v1_targets: list,
        *,
        chunk_size=None,
        parallel=True,
    ) -> tuple:
        """ """
        combined_input_ds = xr.open_mfdataset(
            sampled_input_filenames,
            combine="nested",
            parallel=parallel,
            concat_dim="sample",
            chunks={"sample": chunk_size},
        )

        combined_input_ds = combined_input_ds[list(v1_inputs)]

        combined_target_ds = xr.open_mfdataset(
            sampled_target_filenames,
            combine="nested",
            parallel=parallel,
            concat_dim="sample",
            chunks={"sample": chunk_size},
        )

        combined_target_ds["ptend_t"] = (
            combined_target_ds["state_t"] - combined_input_ds["state_t"]
        ) / 1200

        combined_target_ds["ptend_q0001"] = (
            combined_target_ds["state_q0001"] - combined_input_ds["state_q0001"]
        ) / 1200

        combined_target_ds = combined_target_ds[list(v1_targets)]

        return combined_input_ds, combined_target_ds

    def _combine_datasets_also_old(
        self,
        sampled_input_filenames: list,
        sampled_target_filenames: list,
        v1_inputs: list,
        v1_targets: list,
    ) -> tuple:
        """
        Loads and combines the data from the selected filenames into a single xarray dataset. It sub selects the taget variables and removes samples where target variables may not be present.

        Returns:
            Tuple of (combined input dataset, combined target dataset).
        """
        combined_input_ds = xr.open_mfdataset(
            sampled_input_filenames,
            combine="nested",
            concat_dim="sample",
            # coords="minimal",
            # data_vars="minimal",
            # compat="override",
        )

        combined_target_ds = xr.open_mfdataset(
            sampled_target_filenames,
            combine="nested",
            concat_dim="sample",
            # coords="minimal",
            # data_vars="minimal",
            # compat="override",
        )

        try:
            combined_input_ds = combined_input_ds[list(v1_inputs)]
            combined_target_ds["ptend_t"] = (
                combined_target_ds["state_t"] - combined_input_ds["state_t"]
            ) / 1200

            combined_target_ds["ptend_q0001"] = (
                combined_target_ds["state_q0001"] - combined_input_ds["state_q0001"]
            ) / 1200

            combined_target_ds = combined_target_ds[list(v1_targets)]

        except KeyError as e:
            raise RuntimeError(
                f"A required variable is missing from the lazy-loaded dataset: {e}"
            )

        return combined_input_ds, combined_target_ds

    def _combine_datasets_old(
        self,
        sampled_input_filenames: list,
        sampled_target_filenames: list,
        v1_inputs: list,
        v1_targets: list,
    ) -> tuple:
        """
        Loads and combines the data from the selected filenames into a single xarray dataset. It sub selects the taget variables and removes samples where target variables may not be present.

        Returns:
            Tuple of (combined input dataset, combined target dataset).
        """
        new_input_dataset_list = []
        new_target_dataset_list = []
        for input_file, target_file in zip(
            sampled_input_filenames, sampled_target_filenames
        ):
            input_ds = xr.open_dataset(input_file)
            target_ds = xr.open_dataset(target_file)
            target_ds["ptend_t"] = (
                target_ds["state_t"] - input_ds["state_t"]
            ) / 1200  # Tendancy [K/s] (sample rate of 20mins)
            target_ds["ptend_q0001"] = (
                target_ds["state_q0001"] - input_ds["state_q0001"]
            ) / 1200  # Q1 Tendancy [kg/kg/s] (sample rate of 20mins)
            try:
                new_input_dataset_list.append(input_ds[list(v1_inputs)])
                new_target_dataset_list.append(target_ds[list(v1_targets)])
            except:
                # raise a warning
                logging.warning(
                    f"Skipping file pair: {input_file}, {target_file} due to missing variables."
                )
                continue
        combined_input_ds = xr.concat(new_input_dataset_list, dim="sample")
        combined_target_ds = xr.concat(new_target_dataset_list, dim="sample")
        return combined_input_ds, combined_target_ds

    def _normalise_datasets(
        self,
        normalisation_stats: dict,
        input_ds: xr.Dataset,
        target_ds: xr.Dataset,
        output_scale_file_path: str,
        v1_targets: list,
    ) -> tuple:
        """
        Normalises the dataset using (x-xmean)/range for each variable (and each column if 3d variable). As per the original ClimSim paper

        Returns:
            Tuple of (normalised dataset, normalisation statistics).
        """
        stats_dims = ["sample", "ncol"]

        if normalisation_stats is None:

            mean_ds = input_ds.mean(dim=stats_dims)
            range_ds = input_ds.max(dim=stats_dims) - input_ds.min(dim=stats_dims)

            # Note: Not handling zero ranges here, assuming data variability is sufficient.

            normalisation_stats = {"mean": mean_ds, "range": range_ds}

        input_ds = (input_ds - normalisation_stats["mean"]) / normalisation_stats[
            "range"
        ]

        # Scaling targets based on original ClimSim approach
        out_scale = xr.open_dataset(output_scale_file_path)
        out_scale = out_scale[list(v1_targets)]
        target_ds = target_ds * out_scale

        return input_ds, target_ds, normalisation_stats

    def _prepare_data(
        self,
        model_name: str,
        input_ds: xr.Dataset,
        target_ds: xr.Dataset,
        v1_inputs: list,
        v1_targets: list,
    ) -> tuple:
        """
        Prepares the data into a format ready for the specified model.

        Returns:
            Tuple of (input tensor, target tensor). With shape (num_samples, num_features)
        """
        standard_data = [None, "mlp", "yus_mlp"]
        if model_name not in standard_data:
            raise NotImplementedError(
                f"Data preparation for model '{model_name}' is not implemented."
            )

        input_tensor = self._dataset_to_tensor(input_ds)
        target_tensor = self._dataset_to_tensor(target_ds)

        return input_tensor, target_tensor

    def _dataset_to_tensor(self, input_ds: xr.Dataset) -> tuple:
        """ """
        ds_stacked = input_ds.stack(obs=("sample", "ncol"))

        da_list = []
        feature_names = []

        for varname, da in ds_stacked.data_vars.items():
            if "lev" in da.dims:
                # bring to (lev, obs) view (no copy; xarray reorder)
                da_view = da.transpose("lev", "obs")
                L = da_view.sizes["lev"]
                # name the features for each level
                var_feature_names = [f"{varname}_lev{i}" for i in range(L)]
            else:
                # expand a fake lev dimension of length 1 -> (lev=1, obs)
                da_view = da.expand_dims({"lev": [0]}).transpose("lev", "obs")
                var_feature_names = [f"{varname}"]

            # set lev coordinate to the feature names so concat produces a feature axis with labels
            da_view = da_view.assign_coords({"lev": var_feature_names})
            # rename lev into 'feature' so concat dims match; we'll convert dim name after concat
            da_view = da_view.rename({"lev": "feature"})
            da_list.append(da_view)
            feature_names.extend(var_feature_names)

        # 2) Concatenate all (feature, obs) DataArrays along 'feature' (still lazy if dask)
        combined = xr.concat(da_list, dim="feature")

        # Ensure feature coordinate is the string list (concat should have done this, but ensure)
        combined = combined.assign_coords(feature=("feature", feature_names))

        combined = combined.transpose("obs", "feature")

        combined = combined.astype("float32")

        # compute once (may be large)
        start_time = time.time()
        arr = combined.data.compute()  # returns numpy array if small enough
        logging.info(
            f"DataArray to numpy array compute time: {time.time() - start_time}"
        )

        # ensure contiguous and owned
        arr = np.ascontiguousarray(arr)

        tensor = torch.from_numpy(arr)

        return tensor

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        input_data = self.input[idx]
        target_data = self.target[idx]
        return input_data, target_data


def check_matching_files(input_files, target_files):
    """
    Checks if every input file has a corresponding target file
    by comparing their normalized filenames (excluding the .mli./.mlo. part).
    """

    # 1. Define the normalization function (removes the file type marker)
    # This function replaces '.mli.' and '.mlo.' with just '.'
    # to create a common, comparable string (the unique ID).
    normalize = lambda s: s.replace(".mli.", ".").replace(".mlo.", ".")

    # 2. Normalize both lists
    normalized_inputs = [normalize(f) for f in input_files]
    normalized_targets = [normalize(f) for f in target_files]

    # 3. Check for matching length and content
    # They must have the same length AND the sorted normalized lists must be identical.
    if len(normalized_inputs) != len(normalized_targets):
        logging.warning(
            f"Numerical mismatch: Input list has {len(input_files)} files, Target has {len(target_files)}."
        )
        return False

    # Check if all normalized file names are an exact match (requires sorting)
    if sorted(normalized_inputs) != sorted(normalized_targets):
        logging.warning("Filename mismatch: between input and target files.")
        return False

    return True


def prune_mismached_files(input_files, target_files):
    """
    Prunes input and target file lists to only include matching pairs.
    Returns pruned lists of (input_files, target_files).
    """
    # 1. Define the normalization function (removes the file type marker)
    normalize = lambda s: s.replace(".mli.", ".").replace(".mlo.", ".")

    # 2. Normalize both lists and create a mapping from normalized ID to original filename
    normalized_input_map = {normalize(f): f for f in input_files}
    normalized_target_map = {normalize(f): f for f in target_files}

    # 3. Find the set of common, matching IDs
    input_ids = set(normalized_input_map.keys())
    target_ids = set(normalized_target_map.keys())

    # IDs present in both sets
    common_ids = input_ids.intersection(target_ids)

    # 4. Filter the original file lists using the common IDs
    pruned_input_files = sorted([normalized_input_map[id_] for id_ in common_ids])
    pruned_target_files = sorted([normalized_target_map[id_] for id_ in common_ids])

    # Optional: Log how many files were pruned
    pruned_count = len(input_files) - len(pruned_input_files)
    if pruned_count > 0:
        logging.warning(
            f"Pruned {pruned_count} files (input and target pairs) that did not have a match."
        )

    return pruned_input_files, pruned_target_files

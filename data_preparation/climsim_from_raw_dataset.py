"""
Pytorch dataset that loads from raw data files
"""

from torch.utils.data import Dataset
import os
import torch
import xarray as xr
from omegaconf import DictConfig
import logging
import numpy as np


class ClimSimFromRawDataset(Dataset):
    def __init__(
        self,
        mode: str,
        dataset_testing_type: str,
        dataset_cfg: DictConfig,
        model: str = None,
        normalisation_stats: dict = None,
        unit_test_specific_methods: bool = False,
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

        self.input_filenames, self.target_filenames = self._get_dataset_filenames(
            dataset_cfg.base_folder_path,
            dataset_cfg.target_years,
            dataset_cfg.target_months,
        )

        self.sampled_input_filenames, self.sampled_target_filenames = (
            self._sample_filenames(
                self.input_filenames,
                self.target_filenames,
                sample_rate=dataset_cfg.sample_rate,
            )
        )

        input_ds, target_ds = self._combine_datasets(
            self.sampled_input_filenames,
            self.sampled_target_filenames,
            v1_inputs=dataset_cfg.v1_inputs,
            v1_targets=dataset_cfg.v1_targets,
        )

        normalised_input_ds, normalised_target_ds, self.normalisation_stats = (
            self._normalise_datasets(
                self.normalisation_stats,
                input_ds,
                target_ds,
                output_scale_file_path=dataset_cfg.output_scale_file_path,
                v1_targets=dataset_cfg.v1_targets,
            )
        )

        self.input, self.target = self._prepare_data(
            self.model,
            normalised_input_ds,
            normalised_target_ds,
            v1_inputs=dataset_cfg.v1_inputs,
            v1_targets=dataset_cfg.v1_targets,
        )

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
                for filename in os.listdir(folder_path):
                    if input_regex in filename:
                        input_filelist.append(os.path.join(folder_path, filename))
                    elif target_regex in filename:
                        target_filelist.append(os.path.join(folder_path, filename))

        input_filelist, target_filelist = sorted(input_filelist), sorted(
            target_filelist
        )
        assert check_matching_files(input_filelist, target_filelist)
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

            if np.any(range_ds[var].values == 0 for var in range_ds.data_vars):
                logging.warning(
                    "Some variables have zero range during normalisation. This may lead to NaNs."
                )

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
        if model_name in standard_data:
            input_array = []
            for var in v1_inputs:
                arr = input_ds[var].values
                if arr.ndim == 2:
                    # make a column vector from flattened 2D array
                    x = np.expand_dims(arr.flatten(), 1)
                elif arr.ndim == 3:
                    # keep first axis (e.g. lev) and collapse the rest
                    arr = arr.transpose(0, 2, 1)
                    x = arr.reshape(-1, arr.shape[2])
                else:
                    raise ValueError(
                        f"Unsupported number of dims ({arr.ndim}) for input var '{var}'"
                    )
                input_array.append(x)
            input_array = np.concatenate(input_array, axis=1)

            # build target array the same way
            target_array = []
            for var in v1_targets:
                arr = target_ds[var].values
                if arr.ndim == 2:
                    y = np.expand_dims(arr.flatten(), 1)
                elif arr.ndim == 3:
                    arr = arr.transpose(0, 2, 1)
                    y = arr.reshape(-1, arr.shape[2])
                else:
                    raise ValueError(
                        f"Unsupported number of dims ({arr.ndim}) for target var '{var}'"
                    )

                target_array.append(y)

            target_array = np.concatenate(target_array, axis=1)

            return torch.from_numpy(input_array), torch.from_numpy(target_array)
        else:
            raise NotImplementedError(
                f"Data preparation for model '{model_name}' is not implemented."
            )

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
        print(
            f"Mismatch: Input list has {len(input_files)} files, Target has {len(target_files)}."
        )
        return False

    # Check if all normalized file names are an exact match (requires sorting)
    return sorted(normalized_inputs) == sorted(normalized_targets)

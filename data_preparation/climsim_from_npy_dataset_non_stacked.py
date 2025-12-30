"""
Script that builds a pytorch dataset around an input and target npy files that have been partially processed and saved to disk BUT NON STACKED (spatially).

"""

import torch
import numpy as np
from torch.utils.data import Dataset
import os
import xarray as xr
import logging


class ClimSimNpyDatasetNonStacked(Dataset):
    def __init__(
        self,
        dataset_cfg,
        dataset_testing_type: str,
        mode: str = None,
        normalisation_stats=None,
        model=None,
        seed=None,
    ):
        """
        Initializes the ClimSimNpyDatasetNonStacked.

        Args:
            dataset_cfg: Configuration object containing dataset parameters.
            dataset_testing_type (str): Type of testing e.g qt, dr, full
            group_idx (str): Identifier for the data group to load.
            normalisation_stats (dict, optional): Precomputed normalization statistics. If None, statistics will be computed from the data.
            mode (str, optional): Mode of the dataset ('train', 'val', 'test', or None). Defaults to None.
            model (str, optional): Model name to determine data format conversion. Defaults to None.
            seed (int, optional): Random seed for data splitting. Defaults to None.
        """
        super().__init__()
        self.seed = seed
        self.dataset_cfg = dataset_cfg
        self.data_dir = dataset_cfg.data_dir
        self.input_file = dataset_cfg.input_file
        self.target_file = dataset_cfg.target_file
        self.mode = mode
        self.levels = dataset_cfg.levels
        self.model = model

        self.keep_spatial_groups = (
            True
            if (self.model == "vib_unet_spatial" or self.model == "my_model_1")
            else False
        )

        self._get_group_idx()

        sample_rate = f"sample_rate_{self.dataset_cfg.dataset_testing_sample_rates[dataset_testing_type]}"
        logging.info(f"Using sample rate directory: {sample_rate}")

        # Load input and target data from npy files
        self.input = np.load(
            os.path.join(self.data_dir, self.group_idx, sample_rate, self.input_file)
        )
        self.target = np.load(
            os.path.join(self.data_dir, self.group_idx, sample_rate, self.target_file)
        )
        logging.info(f"Loaded input shape: {self.input.shape}")
        logging.info(f"Loaded target shape: {self.target.shape}")

        # Split data based on mode
        self._split_data()

        # Apply feature wise normalization if specified
        self._feature_wise_normalization(normalisation_stats)

        # Load spatial information
        grid_info = xr.open_dataset(dataset_cfg.path_to_grid_info)
        self.latitudes = grid_info["lat"]

        # Convert to torch tensors
        self.input = torch.tensor(self.input, dtype=torch.float32)
        self.target = torch.tensor(self.target, dtype=torch.float32)

        # TODO: spatial grouping
        if self.keep_spatial_groups:
            northern_hemisphere_indices = self.latitudes.values >= 0
            southern_hemisphere_indices = self.latitudes.values < 0
            logging.info(
                f"Northern hemisphere grid points: {sum(northern_hemisphere_indices)}"
            )
            logging.info(
                f"Southern hemisphere grid points: {sum(southern_hemisphere_indices)}"
            )
            self.northern_hemisphere_input = self.input[
                :, northern_hemisphere_indices, :
            ].flatten(end_dim=1)
            self.southern_hemisphere_input = self.input[
                :, southern_hemisphere_indices, :
            ].flatten(end_dim=1)
            self.northern_hemisphere_target = self.target[
                :, northern_hemisphere_indices, :
            ].flatten(end_dim=1)
            self.southern_hemisphere_target = self.target[
                :, southern_hemisphere_indices, :
            ].flatten(end_dim=1)

            self.northern_hemisphere_input = self._convert_input_model_format(
                self.northern_hemisphere_input
            )
            self.southern_hemisphere_input = self._convert_input_model_format(
                self.southern_hemisphere_input
            )
        else:
            # Convert input to required model format
            self.input = self._convert_input_model_format(self.input.flatten(end_dim=1))
            self.target = self.target.flatten(end_dim=1)

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        if self.keep_spatial_groups:
            return len(self.northern_hemisphere_input)
        else:
            return len(self.input)

    def __getitem__(self, idx):
        """
        Retrieves the input-target pair at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.
        Returns:
            tuple: (input_tensor, target_tensor)
        """
        if self.keep_spatial_groups:
            return (
                self.northern_hemisphere_input[idx],
                self.southern_hemisphere_input[idx],
                self.northern_hemisphere_target[idx],
                self.southern_hemisphere_target[idx],
            )

        else:
            return self.input[idx], self.target[idx]

    def _convert_input_model_format(self, data: torch.Tensor):
        """
        Converts the input and target data to the required model format.
        Currently assumes data is already in the correct shape.
        """
        if self.model in [None, "mlp", "yus_mlp", "my_model_1"]:
            # Data is already in (samples, features) format
            logging.info("Data is in MLP format; no conversion needed.")
            return data
        else:
            reshaped_x = torch.stack(
                [
                    data[:, 0 : self.levels],
                    data[:, self.levels : self.levels + self.levels],
                    torch.repeat_interleave(
                        data[:, 2 * self.levels].unsqueeze(1), self.levels, dim=-1
                    ),
                    torch.repeat_interleave(
                        data[:, 2 * self.levels + 1].unsqueeze(1),
                        self.levels,
                        dim=-1,
                    ),
                    torch.repeat_interleave(
                        data[:, 2 * self.levels + 2].unsqueeze(1),
                        self.levels,
                        dim=-1,
                    ),
                    torch.repeat_interleave(
                        data[:, 2 * self.levels + 3].unsqueeze(1),
                        self.levels,
                        dim=-1,
                    ),
                ]
            )
            if (
                self.model == "climsim_unet"
                or self.model == "sparse_unet"
                or self.model == "vib_unet"
                or self.model == "vib_unet_no_skips"
                or self.model == "vib_unet_spatial"
            ):
                reshaped_x = reshaped_x.permute(
                    1, 0, 2
                )  # shape (batch, features, levels)
                data = torch.nn.functional.pad(
                    reshaped_x, (0, 3), mode="constant", value=0
                )
            elif self.model == "squeezeformer" or self.model == "vib_squeezeformer":
                data = reshaped_x.permute(1, 2, 0)  # shape (batch, levels, features)

        logging.info(f"Converted input shape: {data.shape}")
        return data

    def _get_group_idx(self) -> str:
        """
        Determines the target group from the config
        """
        if self.mode == "val":
            group_idx = list(
                self.dataset_cfg[self.dataset_cfg.group_method].val_group.keys()
            )[0]
        elif self.mode == "test":
            group_idx = list(
                self.dataset_cfg[self.dataset_cfg.group_method].test_group.keys()
            )[0]
        else:
            group_idx = self.dataset_cfg[self.dataset_cfg.group_method].target_group

        self.group_idx = group_idx
        logging.info(f"Building dataset for group: {self.group_idx}")

    def _process_output_scaling(
        self, path_to_scaling_file: str, target_variables: list, levels: int
    ) -> np.ndarray:
        """
        Loads the output scaling, selects target variables and levels of interest and returns a np arry of shape (features,)
        """
        max_levels = 60
        min_levels = max_levels - levels
        out_scale = xr.open_dataset(path_to_scaling_file)
        out_scale = out_scale[list(target_variables)]
        out_scale = out_scale.sel(lev=slice(min_levels, max_levels))
        out_scale_array = out_scale.to_array()

        v_vectors = out_scale_array[0:2]

        v_scalars = out_scale_array[2:, 0]

        out_scale = np.concatenate(
            [v_vectors.values.flatten(), v_scalars.values.flatten()]
        )

        return out_scale

    def _split_data(self):
        """
        Splits the data into train, val, and test sets based on the specified mode. The shuffling and splitting is only applied to the time sample dimension.

        Uses:
            self.dataset_cfg.split: A dictionary containing the split ratios for train, val, and test sets.
            self.mode: The mode of the dataset ('train', 'val', 'test', or None).
            self.seed: Seed for random number generator to ensure reproducibility.

        """
        if self.mode not in ["train", "val", "test"]:
            if self.mode == None:
                logging.info("Mode is None: Using full data")
                return
            raise ValueError(
                f"Invalid mode: {self.mode}. Expected one of ['train', 'val', 'test', None]."
            )

        np.random.seed(self.seed)
        N, _, _ = self.input.shape
        perm = np.random.permutation(N)
        n_train = int(N * self.dataset_cfg.split.train)
        n_val = int(N * self.dataset_cfg.split.val)
        if self.mode == "train":
            indices = perm[:n_train]
        elif self.mode == "val":
            indices = perm[n_train : n_train + n_val]
        elif self.mode == "test":
            indices = perm[n_train + n_val :]

        logging.info(f"Using: {self.mode} with {len(indices)} samples")
        self.input = self.input[indices]
        self.target = self.target[indices]

    def _feature_wise_normalization(self, normalisation_stats=None):
        """
        Applies feature-wise normalization to the input and target data.

        Args:
            normalisation_stats (dict, optional): Precomputed normalization statistics. If None, statistics will be computed from the data.
        """
        self.normalize = self.dataset_cfg.normalize
        if self.normalize:
            if normalisation_stats is not None:
                logging.info("Using provided normalization statistics.")
                self.normalisation_stats = normalisation_stats
            else:
                logging.info("Calculating normalization statistics from data.")
                self.normalisation_stats = {
                    "mean": self.input.mean(axis=(0, 1)),
                    "max": self.input.max(axis=(0, 1)),
                    "min": self.input.min(axis=(0, 1)),
                }

            self.input = (self.input - self.normalisation_stats["mean"]) / (
                self.normalisation_stats["max"] - self.normalisation_stats["min"]
            )
            out_scale = self._process_output_scaling(
                self.dataset_cfg.output_scale_file_path,
                self.dataset_cfg.v1_targets,
                self.dataset_cfg.levels,
            )
            self.target = self.target * out_scale

        else:
            logging.info("Normalization not applied as per configuration.")
            return

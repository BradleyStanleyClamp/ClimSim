"""
Script that builds a pytorch dataset around an input and target npy files that have been partially processed and saved to disk.

V1:
    Current processing steps applied to the raw data before saving to npy:
        - Grouped by months (e.g. DJF, MAM, JJA, SON)
        - Sampled at a lower rate (e.g. every 7th timestep)
        - Removed top levels of the atmosphere
        - Variable selection

    Processing steps to be applied in this script:
        - Normalization (e.g. min-max, standardization)
        - Convert to torch tensors

    Future processing steps to consider:
        - Splitting data into train/val/test sets
        - Selecting specific lat/lon regions

"""

import torch
import numpy as np
from torch.utils.data import Dataset
import os
import xarray as xr
import logging


class ClimSimNpyDataset(Dataset):
    def __init__(
        self,
        dataset_cfg,
        dataset_testing_type: str,
        mode: str = None,
        normalisation_stats=None,
        seed=None,
    ):
        """
        Initializes the ClimSimNpyDataset.

        Args:
            dataset_cfg: Configuration object containing dataset parameters.
            dataset_testing_type (str): Type of testing e.g qt, dr, full
            group_idx (str): Identifier for the data group to load.
            normalisation_stats (dict, optional): Precomputed normalization statistics. If None, statistics will be computed from the data.
        """
        if mode == "val":
            group_idx = list(dataset_cfg[dataset_cfg.group_method].val_group.keys())[0]
        elif mode == "test":
            group_idx = list(dataset_cfg[dataset_cfg.group_method].test_group.keys())[0]
        else:
            group_idx = dataset_cfg[dataset_cfg.group_method].target_group
            
        self.group_idx = group_idx
        logging.info(f"Building dataset for group: {self.group_idx}")

        self.data_dir = dataset_cfg.data_dir
        self.input_file = dataset_cfg.input_file
        self.target_file = dataset_cfg.target_file
        sample_rate = f"sample_rate_{dataset_cfg.dataset_testing_sample_rates[dataset_testing_type]}"
        logging.info(f"Using sample rate directory: {sample_rate}")

        self.normalize = dataset_cfg.normalize
        self.mode = mode
        if self.mode not in ["train", "val", "test"]:
            if mode == None:
                logging.info("Mode is None: Using full data")
            raise ValueError(
                f"Invalid mode: {self.mode}. Expected one of ['train', 'val', 'test', None]."
            )

        # Load input and target data from npy files
        self.input = np.load(
            os.path.join(self.data_dir, self.group_idx, sample_rate, self.input_file)
        )
        self.target = np.load(
            os.path.join(self.data_dir, self.group_idx, sample_rate, self.target_file)
        )

        # Split data based on mode
        np.random.seed(seed)
        N, _ = self.input.shape
        perm = np.random.permutation(N)
        n_train = int(N * dataset_cfg.split.train)
        n_val = int(N * dataset_cfg.split.val)
        if self.mode == "train":
            indices = perm[:n_train]
        elif self.mode == "val":
            indices = perm[n_train : n_train + n_val]
        elif self.mode == "test":
            indices = perm[n_train + n_val :]

        if self.mode != None:
            logging.info(f"Using: {self.mode} with {len(indices)} samples")
            self.input = self.input[indices]
            self.target = self.target[indices]

        # Apply feature wise normalization if specified
        if self.normalize:
            if normalisation_stats is not None:
                logging.info("Using provided normalization statistics.")
                self.normalisation_stats = normalisation_stats
            else:
                logging.info("Calculating normalization statistics from data.")
                self.normalisation_stats = {
                    "mean": self.input.mean(axis=0),
                    "max": self.input.max(axis=0),
                    "min": self.input.min(axis=0),
                }

            self.input = (self.input - self.normalisation_stats["mean"]) / (
                self.normalisation_stats["max"] - self.normalisation_stats["min"]
            )
            out_scale = self._process_output_scaling(
                dataset_cfg.output_scale_file_path,
                dataset_cfg.v1_targets,
                dataset_cfg.levels,
            )
            self.target = self.target * out_scale

        # Convert to torch tensors
        self.input = torch.tensor(self.input, dtype=torch.float32)
        self.target = torch.tensor(self.target, dtype=torch.float32)

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """

        return len(self.input)

    def __getitem__(self, idx):
        """
        Retrieves the input-target pair at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.
        Returns:
            tuple: (input_tensor, target_tensor)
        """
        return self.input[idx], self.target[idx]

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

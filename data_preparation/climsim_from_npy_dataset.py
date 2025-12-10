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
        model=None,
        seed=None,
    ):
        """
        Initializes the ClimSimNpyDataset.

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


        # Convert to torch tensors
        self.input = torch.tensor(self.input, dtype=torch.float32)
        self.target = torch.tensor(self.target, dtype=torch.float32)


        # Convert input to required model format
        self._convert_input_model_format()

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

    def _convert_input_model_format(self):
        """
        Converts the input and target data to the required model format.
        Currently assumes data is already in the correct shape.
        """
        if self.model in [None, "mlp", "yus_mlp"]:
            # Data is already in (samples, features) format
            logging.info("Data is in MLP format; no conversion needed.")
            return
        else:
            reshaped_x = torch.stack(
                [
                    self.input[:, 0 : self.levels],
                    self.input[:, self.levels : self.levels + self.levels],
                    torch.repeat_interleave(
                        self.input[:, 2 * self.levels].unsqueeze(1), self.levels, dim=-1
                    ),
                    torch.repeat_interleave(
                        self.input[:, 2 * self.levels + 1].unsqueeze(1),
                        self.levels,
                        dim=-1,
                    ),
                    torch.repeat_interleave(
                        self.input[:, 2 * self.levels + 2].unsqueeze(1),
                        self.levels,
                        dim=-1,
                    ),
                    torch.repeat_interleave(
                        self.input[:, 2 * self.levels + 3].unsqueeze(1),
                        self.levels,
                        dim=-1,
                    ),
                ]
            )
            if self.model == "climsim_unet":
                reshaped_x = reshaped_x.permute(
                    1, 0, 2
                )  # shape (batch, features, levels)
                self.input = torch.nn.functional.pad(
                    reshaped_x, (0, 3), mode="constant", value=0
                )
            elif self.model == "squeezeformer":
                self.input = reshaped_x.permute(
                    1, 2, 0
                )  # shape (batch, levels, features)

        logging.info(f"Converted input shape: {self.input.shape}")

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
        Splits the data into train, val, and test sets based on the specified mode.

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
        N, _ = self.input.shape
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
                    "mean": self.input.mean(axis=0),
                    "max": self.input.max(axis=0),
                    "min": self.input.min(axis=0),
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

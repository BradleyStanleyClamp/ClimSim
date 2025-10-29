"""
Script to build pytorch dataset for the subsampled low resolution data
"""

import logging
from pathlib import Path
from torch.utils.data import Dataset
import xarray as xr
from climsim_utils.data_utils import *
from omegaconf import DictConfig
from .select_data import sample_data_based_on_testing_type, select_year_of_data
import os 

class SubSampledLowResDataset(Dataset):
    def __init__(
        self, mode: str, dataset_testing_type: str, dataset_config: DictConfig, group_by_year: bool = False
    ):
        """
        Args:
            mode: (str) one of 'train', 'val' or 'test', specifying which dataset split to return
            dataset_testing_type: (str) size of dataset to be used, related to the type of testing e.g quick, reduced, full
            dataset_config: (DictConfig) configuration for the dataset
        """

        self.mode = mode
        self.dataset_testing_type = dataset_testing_type
        self.dataset_config = dataset_config
        self.group_by_year = group_by_year
        if dataset_testing_type == "quick":
            base_dir = Path(__file__).resolve().parents[1]
            self.data_path = os.path.join(base_dir, dataset_config.precomputed_quick_data_path)

        else:
            self.data_path = dataset_config.data_path

        # Setup ClimSim data class (not sure if necessary but may be useful in the future)
        self.setup_data_class()

        # Loading data based on mode and sample based on testing type
        self.input, self.target = self.load_data()

    def setup_data_class(self):
        # Resolve paths relative to this file so imports from other CWDs work
        base_dir = Path(__file__).resolve().parents[1]
        grid_path = os.path.join(base_dir, "grid_info", "ClimSim_low-res_grid-info.nc")
        norm_path = os.path.join(base_dir, "preprocessing", "normalizations")
   

        grid_info = xr.open_dataset(grid_path)
        input_mean = xr.open_dataset(os.path.join(norm_path, "inputs", "input_mean.nc"))
        input_max = xr.open_dataset(os.path.join(norm_path, "inputs", "input_max.nc"))
        input_min = xr.open_dataset(os.path.join(norm_path, "inputs", "input_min.nc"))
        output_scale = xr.open_dataset(os.path.join(norm_path, "outputs", "output_scale.nc"))

        self.data_class = data_utils(
            grid_info=grid_info,
            input_mean=input_mean,
            input_max=input_max,
            input_min=input_min,
            output_scale=output_scale,
        )
        self.data_class.set_to_v1_vars()


    def load_data(self):
        if self.mode == "train":
            train_input_path = self.data_path + "train_input.npy"
            train_target_path = self.data_path + "train_target.npy"
            train_input = np.load(train_input_path)
            train_target = np.load(train_target_path)
            train_input, train_target = sample_data_based_on_testing_type(
                (train_input, train_target),
                self.dataset_testing_type,
                self.dataset_config.dataset_testing_fractions,
            )

            if self.group_by_year:
                logging.info("Selecting data for specific year as per configuration")
                train_input, train_target = select_year_of_data((train_input,train_target), self.dataset_config.group_by_year, self.dataset_config, self.dataset_testing_type)

            self.data_class.input_train = train_input
            self.data_class.target_train = train_target
            return self.data_class.input_train, self.data_class.target_train

        elif self.mode == "val":
            val_input_path = self.data_path + "val_input.npy"
            val_target_path = self.data_path + "val_target.npy"
            val_input = np.load(val_input_path)
            val_target = np.load(val_target_path)
            val_input, val_target = sample_data_based_on_testing_type(
                (val_input, val_target),
                self.dataset_testing_type,
                self.dataset_config.dataset_testing_fractions,
            )
            self.data_class.input_val = val_input
            self.data_class.target_val = val_target
            return self.data_class.input_val, self.data_class.target_val

        elif self.mode == "test":
            test_input_path = self.data_path + "scoring_input.npy"
            test_target_path = self.data_path + "scoring_target.npy"
            test_input = np.load(test_input_path)
            test_target = np.load(test_target_path)
            test_input, test_target = sample_data_based_on_testing_type(
                (test_input, test_target),
                self.dataset_testing_type,
                self.dataset_config.dataset_testing_fractions,
            )

            self.data_class.input_scoring = test_input
            self.data_class.target_scoring = test_target
            return self.data_class.input_scoring, self.data_class.target_scoring

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        return self.input[idx], self.target[idx]

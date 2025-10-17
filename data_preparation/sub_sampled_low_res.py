"""
Script to build pytorch dataset for the subsampled low resolution data
"""

from pathlib import Path
from torch.utils.data import Dataset
import xarray as xr
from climsim_utils.data_utils import *
from omegaconf import DictConfig
from .select_data import sample_data_based_on_testing_type
import os 

class SubSampledLowResDataset(Dataset):
    def __init__(
        self, mode: str, dataset_testing_type: str, dataset_config: DictConfig
    ):
        """
        Args:
            mode: (str) one of 'train', 'val' or 'test', specifying which dataset split to return
            dataset_testing_type: (str) size of dataset to be used, related to the type of testing e.g quick, reduced, full
            dataset_config: (DictConfig) configuration for the dataset
        """

        self.data_path = dataset_config.data_path
        self.mode = mode
        self.dataset_testing_type = dataset_testing_type

        # Setup ClimSim data class (not sure if necessary but may be useful in the future)
        self.setup_data_class()

        # Loading data based on mode
        input, target = self.load_data()

        # Subsample data based on dataset_testing_type

        self.input, self.target = sample_data_based_on_testing_type(
            (input, target),
            self.dataset_testing_type,
            dataset_config.dataset_testing_fractions,
        )

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

    def load_data(self):
        if self.mode == "train":
            train_input_path = self.data_path + "train_input.npy"
            train_target_path = self.data_path + "train_target.npy"
            self.data_class.input_train = self.data_class.load_npy_file(
                train_input_path
            )
            self.data_class.target_train = self.data_class.load_npy_file(
                train_target_path
            )
            return self.data_class.input_train, self.data_class.target_train

        elif self.mode == "val":
            val_input_path = self.data_path + "val_input.npy"
            val_target_path = self.data_path + "val_target.npy"
            self.data_class.input_val = self.data_class.load_npy_file(val_input_path)
            self.data_class.target_val = self.data_class.load_npy_file(val_target_path)
            return self.data_class.input_val, self.data_class.target_val

        elif self.mode == "test":
            test_input_path = self.data_path + "scoring_input.npy"
            test_target_path = self.data_path + "scoring_target.npy"
            self.data_class.input_test = self.data_class.load_npy_file(test_input_path)
            self.data_class.target_test = self.data_class.load_npy_file(
                test_target_path
            )
            return self.data_class.input_test, self.data_class.target_test

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        return self.input[idx], self.target[idx]

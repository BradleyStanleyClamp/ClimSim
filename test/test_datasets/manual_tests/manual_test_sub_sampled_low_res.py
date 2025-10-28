"""
Testing script for the sub-sampled low resolution dataset functionalities
"""

import pytest 
import data_preparation
from omegaconf import OmegaConf, DictConfig
from pathlib import Path
import os



def explore_pressure_grid(sub_sampled_low_res_config_path: str = "../../config/dataset/sub_sampled_low_res.yaml"):
    base_dir = Path(__file__).resolve().parents[2]
    data_path = os.path.join(base_dir, "unit_test_sets", "sub_sampled_low_res/")
    dataset_cfg: DictConfig = OmegaConf.create({
        'dataset_name': 'subsampled_low_res',
        'data_path': data_path,
        'dataset_testing_fractions': {
            'quick': 0.01,
            'reduced': 0.1,
            'full': 1.0
        }
    })
    mode = 'train'
    dataset_testing_type = 'full'
    dataset = data_preparation.SubSampledLowResDataset(mode, dataset_testing_type, dataset_cfg)
    
    dataset.data_class.set_pressure_grid(data_split=mode)

    pressure_grid = dataset.data_class.pressure_grid_train 

    print("Pressure grid for training data:", pressure_grid)


if __name__ == "__main__":
    explore_pressure_grid()



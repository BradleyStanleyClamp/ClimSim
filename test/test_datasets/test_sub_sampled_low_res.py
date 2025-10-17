"""
Testing script for the sub-sampled low resolution dataset functionalities
"""

import pytest 
import data_preparation
from omegaconf import OmegaConf, DictConfig
from pathlib import Path
import os

def test_sub_sampled_low_res_dataset_initialization(sub_sampled_low_res_config_path: str = "../../config/datasets/sub_sampled_low_res.yaml"):
    base_dir = Path(__file__).resolve().parents[1]
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
    
    config_path = Path(sub_sampled_low_res_config_path)
    if not config_path.is_absolute():
        config_path = (Path(__file__).resolve().parent / config_path).resolve()
    with config_path.open("r") as f:
        config = OmegaConf.load(f)

    assert len(dataset) == config.dataset_testing_fractions.unit_test

def test_sub_sampled_low_res_dataset_initialization_test(sub_sampled_low_res_config_path: str = "../../config/datasets/sub_sampled_low_res.yaml"):
    base_dir = Path(__file__).resolve().parents[1]
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
    mode = 'test'
    dataset_testing_type = 'full'
    dataset = data_preparation.SubSampledLowResDataset(mode, dataset_testing_type, dataset_cfg)
    
    config_path = Path(sub_sampled_low_res_config_path)
    if not config_path.is_absolute():
        config_path = (Path(__file__).resolve().parent / config_path).resolve()
    with config_path.open("r") as f:
        config = OmegaConf.load(f)

    assert len(dataset) == config.dataset_testing_fractions.unit_test


def test_sub_sampled_low_res_dataset_initialization_val(sub_sampled_low_res_config_path: str = "../../config/datasets/sub_sampled_low_res.yaml"):
    base_dir = Path(__file__).resolve().parents[1]
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
    mode = 'val'
    dataset_testing_type = 'full'
    dataset = data_preparation.SubSampledLowResDataset(mode, dataset_testing_type, dataset_cfg)
    
    config_path = Path(sub_sampled_low_res_config_path)
    if not config_path.is_absolute():
        config_path = (Path(__file__).resolve().parent / config_path).resolve()
    with config_path.open("r") as f:
        config = OmegaConf.load(f)

    assert len(dataset) == config.dataset_testing_fractions.unit_test

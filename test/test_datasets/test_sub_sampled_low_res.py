"""
Testing script for the sub-sampled low resolution dataset functionalities
"""

import pytest 
import data_preparation
from omegaconf import OmegaConf, DictConfig
from pathlib import Path
import os

def test_sub_sampled_low_res_dataset_initialization(sub_sampled_low_res_config_path: str = "../../config/dataset/sub_sampled_low_res.yaml"):
    base_dir = Path(__file__).resolve().parents[1]
    data_path = os.path.join(base_dir, "unit_test_sets", "sub_sampled_low_res/")
    dataset_cfg: DictConfig = OmegaConf.create({
        'dataset_name': 'subsampled_low_res',
        'data_path': data_path,
        'dataset_testing_fractions': {
            'quick': 0.01,
            'reduced': 0.1,
            'full': 1.0
        },
        'group_method': False,
        'remove_high_altitude_specific_humidity_levels': False
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

def test_sub_sampled_low_res_dataset_initialization_test(sub_sampled_low_res_config_path: str = "../../config/dataset/sub_sampled_low_res.yaml"):
    base_dir = Path(__file__).resolve().parents[1]
    data_path = os.path.join(base_dir, "unit_test_sets", "sub_sampled_low_res/")
    dataset_cfg: DictConfig = OmegaConf.create({
        'dataset_name': 'subsampled_low_res',
        'data_path': data_path,
        'dataset_testing_fractions': {
            'quick': 0.01,
            'reduced': 0.1,
            'full': 1.0
        },
        'remove_high_altitude_specific_humidity_levels': False
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


def test_sub_sampled_low_res_dataset_initialization_val(sub_sampled_low_res_config_path: str = "../../config/dataset/sub_sampled_low_res.yaml"):
    base_dir = Path(__file__).resolve().parents[1]
    data_path = os.path.join(base_dir, "unit_test_sets", "sub_sampled_low_res/")
    dataset_cfg: DictConfig = OmegaConf.create({
        'dataset_name': 'subsampled_low_res',
        'data_path': data_path,
        'dataset_testing_fractions': {
            'quick': 0.01,
            'reduced': 0.1,
            'full': 1.0
        },
        'remove_high_altitude_specific_humidity_levels': False
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


def test_sub_sampled_low_res_dataset_for_unet(sub_sampled_low_res_config_path: str = "../../config/dataset/sub_sampled_low_res.yaml"):
    base_dir = Path(__file__).resolve().parents[1]
    data_path = os.path.join(base_dir, "unit_test_sets", "sub_sampled_low_res/")
    dataset_cfg: DictConfig = OmegaConf.create({
        'dataset_name': 'subsampled_low_res',
        'data_path': data_path,
        'dataset_testing_fractions': {
            'quick': 0.01,
            'reduced': 0.1,
            'full': 1.0
        },
        'group_method': False,
        'remove_high_altitude_specific_humidity_levels': False
    })
    mode = 'train'
    dataset_testing_type = 'full'
    dataset = data_preparation.SubSampledLowResDataset(mode, dataset_testing_type, dataset_cfg, model='climsim_unet')
    x, y = dataset[0]
    assert x.shape[0] == 6
    assert x.shape[1] == 64


def test_sub_sampled_low_res_dataset_year_grouping(sub_sampled_low_res_config_path: str = "../../config/dataset/sub_sampled_low_res.yaml"):
    base_dir = Path(__file__).resolve().parents[1]
    data_path = os.path.join(base_dir, "unit_test_sets", "sub_sampled_low_res/")
    dataset_cfg: DictConfig = OmegaConf.create({
        'dataset_name': 'subsampled_low_res',
        'precomputed_quick_data_path': data_path,
        'dataset_testing_fractions': {
            'quick': 1,
            'reduced': 0.1,
            'full': 1.0
        },
        'input_dim': 124,
        'output_dim': 128,
        'num_spatial_points': 384,
        'samples_per_day': 72,
        'subsample_factors': {
            'train': 7
        },
        'group_method': 'group_by_year',
        'group_by_year': {
            'target_group': 0
        },
        'remove_high_altitude_specific_humidity_levels': False
    })
    mode = 'train'
    dataset_testing_type = 'quick'
    dataset = data_preparation.SubSampledLowResDataset(mode, dataset_testing_type, dataset_cfg, model='mlp')
    assert len(dataset) == 384

def test_sub_sampled_low_res_dataset_no_grouping(sub_sampled_low_res_config_path: str = "../../config/dataset/sub_sampled_low_res.yaml"):
    base_dir = Path(__file__).resolve().parents[1]
    data_path = os.path.join(base_dir, "unit_test_sets", "sub_sampled_low_res/")
    dataset_cfg: DictConfig = OmegaConf.create({
        'dataset_name': 'subsampled_low_res',
        'precomputed_quick_data_path': data_path,
        'dataset_testing_fractions': {
            'quick': 1,
            'reduced': 0.1,
            'full': 1.0
        },
        'input_dim': 124,
        'output_dim': 128,
        'num_spatial_points': 384,
        'samples_per_day': 72,
        'subsample_factors': {
            'train': 7
        },
        'group_method': False,
        'group_by_year': {
            'target_group': 0
        },
        'remove_high_altitude_specific_humidity_levels': False
    })
    mode = 'train'
    dataset_testing_type = 'quick'
    dataset = data_preparation.SubSampledLowResDataset(mode, dataset_testing_type, dataset_cfg, model='mlp')
    assert len(dataset) == 3840
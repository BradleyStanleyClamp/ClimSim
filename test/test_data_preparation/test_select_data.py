"""
Testing functionalities for getting datasets and data loaders
"""

import data_preparation
import pytest
from omegaconf import OmegaConf, DictConfig
from pathlib import Path
import os

def test_get_dataset_full(sub_sampled_low_res_config_path: str = "../../config/datasets/sub_sampled_low_res.yaml"):
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
    dataset = data_preparation.select_data.get_dataset(dataset_cfg, mode, dataset_testing_type)
    assert isinstance(dataset, data_preparation.sub_sampled_low_res.SubSampledLowResDataset)
    config_path = Path(sub_sampled_low_res_config_path)
    if not config_path.is_absolute():
        config_path = (Path(__file__).resolve().parent / config_path).resolve()
    with config_path.open("r") as f:
        config = OmegaConf.load(f)

    assert len(dataset) == config.dataset_testing_fractions.unit_test

def test_get_dataset_reduced(sub_sampled_low_res_config_path: str = "../../config/datasets/sub_sampled_low_res.yaml"):
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
    dataset_testing_type = 'reduced'
    dataset = data_preparation.select_data.get_dataset(dataset_cfg, mode, dataset_testing_type)
    assert isinstance(dataset, data_preparation.sub_sampled_low_res.SubSampledLowResDataset)
    config_path = Path(sub_sampled_low_res_config_path)
    if not config_path.is_absolute():
        config_path = (Path(__file__).resolve().parent / config_path).resolve()
    with config_path.open("r") as f:
        config = OmegaConf.load(f)

    assert len(dataset) == int(config.dataset_testing_fractions.unit_test * dataset_cfg.dataset_testing_fractions.reduced)

def test_get_dataset_quick(sub_sampled_low_res_config_path: str = "../../config/datasets/sub_sampled_low_res.yaml"):
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
    dataset_testing_type = 'quick'
    dataset = data_preparation.select_data.get_dataset(dataset_cfg, mode, dataset_testing_type)
    assert isinstance(dataset, data_preparation.sub_sampled_low_res.SubSampledLowResDataset)
    config_path = Path(sub_sampled_low_res_config_path)
    if not config_path.is_absolute():
        config_path = (Path(__file__).resolve().parent / config_path).resolve()
    with config_path.open("r") as f:
        config = OmegaConf.load(f)

    assert len(dataset) == int(config.dataset_testing_fractions.unit_test * dataset_cfg.dataset_testing_fractions.quick)
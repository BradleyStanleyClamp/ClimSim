"""
Testing functionalities for getting datasets and data loaders
"""

import data_preparation
import pytest
from omegaconf import OmegaConf, DictConfig
from pathlib import Path
import os
import numpy as np
from omegaconf import OmegaConf

def test_get_dataset_full(sub_sampled_low_res_config_path: str = "../../config/dataset/sub_sampled_low_res.yaml"):
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
    dataset = data_preparation.select_data.get_dataset(dataset_cfg, mode, dataset_testing_type)
    assert isinstance(dataset, data_preparation.sub_sampled_low_res.SubSampledLowResDataset)
    config_path = Path(sub_sampled_low_res_config_path)
    if not config_path.is_absolute():
        config_path = (Path(__file__).resolve().parent / config_path).resolve()
    with config_path.open("r") as f:
        config = OmegaConf.load(f)

    assert len(dataset) == 3840 #config.dataset_testing_fractions.unit_test

def test_get_dataset_reduced(sub_sampled_low_res_config_path: str = "../../config/dataset/sub_sampled_low_res.yaml"):
    base_dir = Path(__file__).resolve().parents[1]
    data_path = os.path.join(base_dir, "unit_test_sets", "sub_sampled_low_res/")
    dataset_cfg: DictConfig = OmegaConf.create({
        'dataset_name': 'subsampled_low_res',
        'data_path': data_path,
        'dataset_testing_fractions': {
            'quick': 0.1,
            'reduced': 0.5,
            'full': 1.0
        },
        'group_method': False,
        'remove_high_altitude_specific_humidity_levels': False

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

    assert len(dataset) == 1920 #int(config.dataset_testing_fractions.unit_test * dataset_cfg.dataset_testing_fractions.reduced)

def test_get_dataset_quick(sub_sampled_low_res_config_path: str = "../../config/dataset/sub_sampled_low_res.yaml"):
    base_dir = Path(__file__).resolve().parents[1]
    data_path = os.path.join(base_dir, "unit_test_sets", "sub_sampled_low_res/")
    dataset_cfg: DictConfig = OmegaConf.create({
        'dataset_name': 'subsampled_low_res',
        'data_path': data_path,
        'precomputed_quick_data_path': data_path,
        'dataset_testing_fractions': {
            'quick': 0.1,
            'reduced': 0.5,
            'full': 1.0
        },
        'group_method': False,
        'remove_high_altitude_specific_humidity_levels': False
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

    assert len(dataset) == 384 #int(config.dataset_testing_fractions.unit_test * dataset_cfg.dataset_testing_fractions.quick)


def test_sample_data_based_on_testing_type_fractions():

    data = (np.arange(3840).reshape(3840, 1), np.arange(3840).reshape(3840, 1))
    dataset_testing_fractions = OmegaConf.create({
        'quick': 0.1,
        'reduced': 0.5,
        'full': 1.0
    })

    # Test quick
    input_quick, target_quick = data_preparation.select_data.sample_data_based_on_testing_type(
        data, 'quick', dataset_testing_fractions
    )
    assert len(input_quick) == 384
    assert len(target_quick) == 384

    # Test reduced
    input_reduced, target_reduced = data_preparation.select_data.sample_data_based_on_testing_type(
        data, 'reduced', dataset_testing_fractions
    )
    assert len(input_reduced) == 1920
    assert len(target_reduced) == 1920

    # Test full
    input_full, target_full = data_preparation.select_data.sample_data_based_on_testing_type(
        data, 'full', dataset_testing_fractions
    )
    assert len(input_full) == 3840
    assert len(target_full) == 3840

def test_sample_data_based_on_testing_type_n_samples():

    data = (np.arange(100).reshape(100, 1), np.arange(100).reshape(100, 1))
    dataset_testing_fractions = OmegaConf.create({
        'quick': 10,
        'reduced': 50,
        'full': 100
    })

    # Test quick
    input_quick, target_quick = data_preparation.select_data.sample_data_based_on_testing_type(
        data, 'quick', dataset_testing_fractions
    )
    assert len(input_quick) == 10
    assert len(target_quick) == 10

    # Test reduced
    input_reduced, target_reduced = data_preparation.select_data.sample_data_based_on_testing_type(
        data, 'reduced', dataset_testing_fractions
    )
    assert len(input_reduced) == 50
    assert len(target_reduced) == 50

    # Test full
    input_full, target_full = data_preparation.select_data.sample_data_based_on_testing_type(
        data, 'full', dataset_testing_fractions
    )
    assert len(input_full) == 100
    assert len(target_full) == 100

def test_select_first_n_samples_single():
    data = (np.arange(100).reshape(100, 1), np.arange(100).reshape(100, 1))
    n_samples = [10]

    input_selected, target_selected = data_preparation.select_data.select_first_n_samples(data, n_samples)

    assert len(input_selected) == 10
    assert len(target_selected) == 10
    np.testing.assert_array_equal(input_selected, np.arange(10).reshape(10, 1))
    np.testing.assert_array_equal(target_selected, np.arange(10).reshape(10, 1))

def test_select_first_n_samples_multiple():
    data = (np.arange(100).reshape(100, 1), np.arange(100).reshape(100, 1))
    n_samples = [10, 20]

    input_selected, target_selected = data_preparation.select_data.select_first_n_samples(data, n_samples)

    assert len(input_selected) == 10
    assert len(target_selected) == 20
    np.testing.assert_array_equal(input_selected, np.arange(10).reshape(10, 1))
    np.testing.assert_array_equal(target_selected, np.arange(20).reshape(20, 1))
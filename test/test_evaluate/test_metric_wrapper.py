import evaluate
import torch
from omegaconf import DictConfig, OmegaConf
import data_preparation
from pathlib import Path
import os
from test.unit_test_sets.dummy_dataset.dummy_dataset import DummyDataset


def test_metric_wrapper():
    torch.random.manual_seed(0)
    X = torch.rand((1000, 2))
    Y = torch.rand((1500, 2))
    X_dataset = DummyDataset(X)
    Y_dataset = DummyDataset(Y)

    metric_wrapper = evaluate.MetricWrapper(
        samples_size=False,
        metric_name="energy_distance",
    )

    dist = metric_wrapper.calculate(X_dataset, Y_dataset)
    expected_dist = evaluate.energy_distance(X, Y)
    assert abs(dist - expected_dist) < 1e-7


def test_metric_wrapper_sampled():
    torch.random.manual_seed(0)
    X = torch.rand((1000, 2))
    Y = torch.rand((1500, 2))
    X_dataset = DummyDataset(X)
    Y_dataset = DummyDataset(Y)

    metric_wrapper = evaluate.MetricWrapper(
        samples_size=100,
        metric_name="energy_distance",
    )

    dist = metric_wrapper.calculate(X_dataset, Y_dataset)
    assert len(X_dataset) == 100
    assert len(Y_dataset) == 100
    expected_dist = evaluate.energy_distance(X_dataset.input, Y_dataset.input)
    assert abs(dist - expected_dist) < 1e-7


def test_metric_wrapper_on_sub_sampled_low_res_no_further_sampling():
    base_dir = Path(__file__).resolve().parents[1]
    data_path = os.path.join(base_dir, "unit_test_sets", "sub_sampled_low_res/")
    dataset_cfg: DictConfig = OmegaConf.create(
        {
            "dataset_name": "subsampled_low_res",
            "precomputed_quick_data_path": data_path,
            "dataset_testing_fractions": {"quick": 1, "reduced": 0.1, "full": 1.0},
            "input_dim": 124,
            "output_dim": 128,
            "num_spatial_points": 384,
            "samples_per_day": 72,
            "subsample_factors": {"train": 7},
            "group_method": "group_by_year",
            "group_by_year": {"target_group": 0},
        }
    )
    dataset_testing_type = "quick"
    trainset = data_preparation.SubSampledLowResDataset(
        "train", dataset_testing_type, dataset_cfg, model="mlp"
    )
    testset = data_preparation.SubSampledLowResDataset(
        "test", dataset_testing_type, dataset_cfg, model="mlp"
    )
    metric_wrapper = evaluate.MetricWrapper(
        samples_size=False,
        metric_name="energy_distance",
    )
    dist = metric_wrapper.calculate(trainset, testset)
    expected_dist = evaluate.energy_distance(trainset.input, testset.input)
    assert abs(dist - expected_dist) < 1e-7


def test_metric_wrapper_on_sub_sampled_low_res_further_sampling():
    base_dir = Path(__file__).resolve().parents[1]
    data_path = os.path.join(base_dir, "unit_test_sets", "sub_sampled_low_res/")
    dataset_cfg: DictConfig = OmegaConf.create(
        {
            "dataset_name": "subsampled_low_res",
            "precomputed_quick_data_path": data_path,
            "dataset_testing_fractions": {"quick": 1, "reduced": 0.1, "full": 1.0},
            "input_dim": 124,
            "output_dim": 128,
            "num_spatial_points": 384,
            "samples_per_day": 72,
            "subsample_factors": {"train": 7},
            "group_method": "group_by_year",
            "group_by_year": {"target_group": 0},
        }
    )
    dataset_testing_type = "quick"
    trainset = data_preparation.SubSampledLowResDataset(
        "train", dataset_testing_type, dataset_cfg, model="mlp"
    )
    testset = data_preparation.SubSampledLowResDataset(
        "test", dataset_testing_type, dataset_cfg, model="mlp"
    )
    metric_wrapper = evaluate.MetricWrapper(
        samples_size=100,
        metric_name="energy_distance",
    )
    dist = metric_wrapper.calculate(trainset, testset)
    expected_dist = evaluate.energy_distance(trainset.input, testset.input)
    assert abs(dist - expected_dist) < 1e-7


def test_metric_wrapper_batched():
    torch.random.manual_seed(0)
    X = torch.rand((1000, 2))
    Y = torch.rand((1500, 2))
    X_dataset = DummyDataset(X)
    Y_dataset = DummyDataset(Y)

    metric_wrapper = evaluate.MetricWrapper(
        samples_size=False,
        metric_name="energy_distance",
        batch_size=100
    )

    dist = metric_wrapper.calculate(X_dataset, Y_dataset)
    expected_dist = evaluate.energy_distance(X, Y)
    assert abs(dist - expected_dist) < 1e-6


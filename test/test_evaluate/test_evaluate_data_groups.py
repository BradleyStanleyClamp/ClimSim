import torch
import evaluate
import data_preparation
from test.unit_test_sets.dummy_dataset.dummy_dataset import DummyDataset
from omegaconf import DictConfig


def test_evaluate_data_group():

    torch.random.manual_seed(0)
    X = torch.rand((1000, 2))
    Y = torch.rand((1500, 2))
    X_dataset = DummyDataset(X)
    Y_dataset = DummyDataset(Y)

    cfg = DictConfig(
        {
            "evaluate": {"sample_size": False},
            "metric_name": "energy_distance",
            "testing": {"batch_size": 100},
            'dataset' : {"general_dataset_config": {
                "num_workers": 2,
                "prefetch_factor": 2,
                "persistent_workers": False,
            }},
        }
    )

    results = evaluate.evaluate_data_group(cfg, X_dataset, Y_dataset, ["multivariate"])

    dist = results["multivariate"]["value"]
    expected_dist = evaluate.energy_distance(X, Y)
    assert abs(dist - expected_dist) < 1e-6

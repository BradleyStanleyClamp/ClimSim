"""
Testing distribution evaluation metrics.
"""

from omegaconf import DictConfig
import torch 
import evaluate
import time 
from scipy.stats import energy_distance

from geomloss import SamplesLoss
import dcor

import numpy as np
import pytest

from test.unit_test_sets.dummy_dataset.dummy_dataset import DummyDataset


def test_matmul_trick():
    X = torch.randn(1000, 10)
    Y = torch.randn(2000, 10)
    torch.set_float32_matmul_precision("medium")
    start_time = time.perf_counter()
    estimate = evaluate.squared_euclidean_distance_matrix_matmul_trick(X, Y)
    elapsed = time.perf_counter() - start_time
    print(f"estimated took {elapsed:.3f} seconds ")

    start_time = time.perf_counter()
    true = torch.cdist(X, Y, p=2) ** 2
    elapsed = time.perf_counter() - start_time
    print(f"true took {elapsed:.3f} seconds ")

    assert estimate.shape == true.shape
    assert torch.allclose(estimate, true, atol=1e-7)


def test_different_energy_distances_1D():
    torch.random.manual_seed(0)
    X = torch.rand((100, 1))
    Y = torch.rand((100, 1))


    Loss =  SamplesLoss("energy")

    geomloss_dist = Loss(X, Y ).item()
    dcor_distance = dcor.energy_distance(X, Y)/2 
    scipy_dist = energy_distance(X.squeeze(), Y.squeeze()) ** 2 /2
    my_dist =  evaluate.energy_distance(X, Y) / 2

    print(X.shape)
    print(f"Energy Distance (geomloss): {geomloss_dist}")
    print(f"Energy Distance (dcor): {dcor_distance}")
    print(f"Energy Distance (scipy): {scipy_dist}")
    print(f"Energy Distance (my): {my_dist}")

    assert abs(geomloss_dist - dcor_distance) < 1e-3
    assert abs(geomloss_dist - scipy_dist) < 1e-3
    assert abs(geomloss_dist - my_dist) < 1e-3

def test_different_energy_distances_2D():
    torch.random.manual_seed(0)
    X = torch.rand((100, 2))
    Y = torch.rand((100, 2))
    Loss =  SamplesLoss("energy")

    geomloss_dist = Loss(X, Y ).item()
    dcor_distance = dcor.energy_distance(X, Y)/2 
    my_dist =  evaluate.energy_distance(X, Y) / 2

    print(X.shape)
    print(f"Energy Distance (geomloss): {geomloss_dist}")
    print(f"Energy Distance (dcor): {dcor_distance}")
    print(f"Energy Distance (my): {my_dist}")

    assert abs(geomloss_dist - dcor_distance) < 1e-3
    assert abs(geomloss_dist - my_dist) < 1e-3

def test_energy_distance_order():
    # (samples, features)
    X = torch.ones((50, 2))
    Y = torch.ones((100, 2))

    Loss =  SamplesLoss("energy")
    geomloss_dist = Loss(X, Y ).item()
    dcor_distance = dcor.energy_distance(X, Y)/2 
    my_dist =  evaluate.energy_distance(X, Y) / 2

    print(X.shape)
    print(f"Energy Distance (geomloss): {geomloss_dist}")
    print(f"Energy Distance (dcor): {dcor_distance}")
    print(f"Energy Distance (my): {my_dist}")

    assert abs(geomloss_dist - dcor_distance) < 1e-3
    assert abs(geomloss_dist - my_dist) < 1e-3


def test_energy_distance_different_dimensions():
    X1 = torch.ones((50, 3))
    Y1 = torch.ones((50, 2))
    with pytest.raises(Exception):  
        evaluate.energy_distance(X1, Y1)


def test_energy_distance_batch():
    torch.random.manual_seed(0)
    X = torch.rand((1000, 2))
    Y = torch.rand((1500, 2))

    dist = evaluate.energy_distance(X, Y, batch_size=128)
    expected_dist = evaluate.energy_distance(X, Y)
    print(f"Energy Distance with dataloader configs: {dist}")
    print(f"Expected Energy Distance: {expected_dist}")
    assert abs(dist - expected_dist) < 1e-6


def test_pca_on_gpu():
    torch.random.manual_seed(0)
    X = torch.rand((1000, 5))
    Y = torch.rand((1500, 5))


    n_components = 3
    X_proj, Y_proj, components, mean = evaluate.pca_gpu(X, Y, n_components=n_components)

    print(f"Original X shape: {X.shape}, Projected X shape: {X_proj.shape}")
    print(f"Original Y shape: {Y.shape}, Projected Y shape: {Y_proj.shape}")
    print(f"Components shape: {components.shape}, Mean shape: {mean.shape}")

    assert X_proj.shape == (1000, n_components)
    assert Y_proj.shape == (1500, n_components)
    assert components.shape == (5, n_components)
    assert mean.shape == (5,)
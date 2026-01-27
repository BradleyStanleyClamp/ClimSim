from evaluate import EMDMetric
import torch
import time
from scipy.stats import wasserstein_distance_nd


def test_emd_identical_distributions():
    emd_metric = EMDMetric()
    data = torch.randn(100, 5)
    emd = emd_metric(data, data)
    assert abs(emd) < 1e-6, f"EMD for identical distributions should be 0, got {emd}"


def test_emd_different_distributions():
    emd_metric = EMDMetric()
    data1 = torch.randn(100, 5)
    data2 = torch.randn(100, 5) + 5.0  # Shifted distribution
    emd = emd_metric(data1, data2)
    assert (
        emd > 0
    ), f"EMD for different distributions should be greater than 0, got {emd}"


def test_emd_with_pca():
    emd_metric = EMDMetric(n_components=2)
    data1 = torch.randn(100, 5)
    data2 = torch.randn(100, 5) + 3.0  # Shifted distribution
    emd = emd_metric(data1, data2)
    assert (
        emd > 0
    ), f"EMD with PCA for different distributions should be greater than 0, got {emd}"


def test_emd_high_dimensional_data():
    emd_metric = EMDMetric(n_components=10)
    data1 = torch.randn(51, 50)
    data2 = torch.randn(51, 50) + 1.0  # Slightly shifted distribution
    emd = emd_metric(data1, data2)
    assert (
        emd > 0
    ), f"EMD for high-dimensional data should be greater than 0, got {emd}"


def test_emd_functional_correctness():
    data1 = torch.randn(100, 2)
    data2 = torch.randn(100, 2) + 1.0  #
    emd_metric = EMDMetric()
    emd = emd_metric(data1, data2)
    emd_scipy = wasserstein_distance_nd(data1.numpy(), data2.numpy())
    assert (
        abs(emd - emd_scipy) < 1e-5
    ), f"EMD mismatch: {emd} (custom) vs {emd_scipy} (scipy)"


def run_timing_analysis():
    emd_metric = EMDMetric()
    data1 = torch.randn(1000, 5)
    data2 = torch.randn(1000, 5) + 2.0

    print("Running timing analysis for EMD computation...")
    start_time = time.perf_counter()
    emd = emd_metric(data1, data2)
    end_time = time.perf_counter()
    print(f"EMD: {emd}, Computation Time: {end_time - start_time:.4f} seconds")


if __name__ == "__main__":
    run_timing_analysis()

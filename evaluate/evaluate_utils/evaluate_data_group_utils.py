"""
Script containing utility functions for evaluating different data groups for distribution shifts.
"""

import logging
import time
import numpy as np
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

# from torch import cdist
from scipy.spatial.distance import cdist
from tqdm import tqdm
import torch
import math
from typing import Optional
from evaluate.evaluate_utils.kl_divergence import KLDivergenceMetric
from evaluate.evaluate_utils.energy_distance import EnergyDistanceMetric


class MetricWrapper:
    def __init__(
        self, samples_size: int, metric_name: str, pca_components: Optional[int] = None, batch_size: Optional[int] = None, device: Optional[torch.device] = None
    ):
        """
        Wrapper for different metrics to handle optimisations and significance testing.
        V2: only supports:
            - Metrics: Energy Distance, KL Divergence [+ PCA]
            - Optimisations: Fixed single sampling process from datasets, batching for large datasets

        Future versions to include:
            - Optimisations: Monte Carlo sampling of multiple samples from datasets
            - Significance Testing: Permutation Tests

        Args:
            samples_size (int): Number of samples to draw from each dataset for metric calculation.
            metric_name (str): Name of the metric to use.
            pca_components (int): Number of PCA components to use (only for KL Divergence).
            batch_size (Optional[int]): Batch size for large datasets.
        """
        self.sample_size = samples_size
        self.batch_size = batch_size
        self.pca_components = pca_components
        # Note: metrics should expect torch tensors in the form (samples, features)
        self.metric_function = self._get_metric_function(metric_name)

        self.observed = None

    def calculate(self, X: torch.Tensor, Y: torch.Tensor) -> float:
        """
        Calculate the metric between datasets X and Y.
        Args:
            X (torch.Tensor): First dataset. (features, samples)
            Y (torch.Tensor): Second dataset. (features, samples)

        Returns:
            metric_value (float): Calculated metric value.
        """

        assert isinstance(X, torch.Tensor) and isinstance(
            Y, torch.Tensor
        ), f"X and Y must be torch Tensors"

        # For now we just have functionality to take a single sample instead of multiple samples (monte carlo sampling)
        if self.sample_size and self.sample_size < min(len(X), len(Y)):
            X = X[torch.randperm(X.shape[0])[:self.sample_size]]
            Y = Y[torch.randperm(Y.shape[0])[:self.sample_size]]
            logging.info(f"Subsampled data has shapes X: {X.shape}, Y: {Y.shape}")

        self.observed = self.metric_function(X, Y)
        return self.observed

    def permutation_test(self, X, Y, num_permutations=500):
        """
        Perform a permutation test to assess significance of the metric.
        Args:
            X (np.ndarray): First dataset.
            Y (np.ndarray): Second dataset.
            num_permutations (int): Number of permutations to perform.

        Returns:
            pval (float): p-value from the permutation test.
        """
        if self.observed is None:
            self.observed = self.calculate(X, Y)

        pass

    def _get_metric_function(self, metric_name: str):
        """
        Get the metric function based on the metric name.
        Args:
            metric_name (str): Name of metric
        """
        if metric_name == "energy_distance":
            return EnergyDistanceMetric(self.batch_size)
        elif metric_name == "kl_divergence":
            if self.pca_components is None:
                raise ValueError("pca_components must be specified for KL Divergence metric.")
            return KLDivergenceMetric(n_components=self.pca_components, device=None)
        else:
            raise ValueError(f"Unknown metric name: {metric_name}")


def evaluate_data_group(
    cfg: DictConfig,
    trainset: Dataset,
    testset: Dataset,
    evaluation_options: list[str],
):
    """
    Evaluate a data group distribution shift compared to the test set using specified evaluation options.

    Args:
        cfg (DictConfig): Configuration object containing evaluation settings.
        trainset (torch.utils.data.Dataset): Training dataset.
        testset (torch.utils.data.Dataset): Test dataset.
        Note: Datasets must have:
            - a .input attribute containing the data as a torch tensor
            - a .sample(num_samples) method to sample num_samples from the dataset (if sampling is required)
        evaluation_options (list[str]): List of evaluation options to perform.

    Returns:
        results (dict): Dictionary containing evaluation results for each option.

    """
    assert isinstance(trainset, Dataset) and isinstance(
        testset, Dataset
    ), f"trainset and testset must be torch Datasets"
    results = {}

    if "multivariate" in evaluation_options:
        metric = MetricWrapper(
            cfg.evaluate.sample_size,
            metric_name=cfg.metric_name,
            pca_components=cfg.evaluate.pca_components,
            batch_size=cfg.testing.batch_size,
        )
        start_time = time.perf_counter()
        observed_metric = metric.calculate(trainset.input, testset.input)
        elapsed = time.perf_counter() - start_time
        logging.info(
            f"Multivariate {cfg.metric_name} calculation took {elapsed:.3f} seconds "
        )
        results["multivariate"] = {"value": observed_metric}
        logging.info(f"Multivariate {cfg.metric_name} Value: {observed_metric:.6f}")


    if "marginals" in evaluation_options:
        num_features = trainset.input.shape[1]
        metric = MetricWrapper(
            cfg.evaluate.sample_size,
            metric_name=cfg.metric_name,
            pca_components=cfg.evaluate.pca_components,
            batch_size=cfg.testing.batch_size,
        )
        for i in range(num_features):
            start_time = time.perf_counter()
            observed_metric = metric.calculate(
                trainset.input[:, i : i + 1], testset.input[:, i : i + 1]
            )
            elapsed = time.perf_counter() - start_time
            logging.info(
                f"Marginal {i} {cfg.metric_name} calculation took {elapsed:.3f} seconds "
            )
            if "marginals" not in results:
                results["marginals"] = {"marginal_distances": [], "marginal_distribution_se": []}
            results["marginals"]["marginal_distances"].append(observed_metric)
            results["marginals"]["marginal_distribution_se"].append(0.0)  # Placeholder for standard error
            logging.info(
                f"Marginal {i} {cfg.metric_name} Value: {observed_metric:.6f}"
            )


    # if "energy_distance" in evaluation_options:
    #     # if cfg.testing.timings:
    #     #     start_time = time.perf_counter()
    #     #     observed, se_observed = monte_carlo_energy_distance(train_data, test_data, K_cross=cfg.energy_distance.K_cross, K_within=cfg.energy_distance.K_within, batch_size=cfg.energy_distance.batch_size, seed=cfg.project.seed)
    #     #     elapsed = time.perf_counter() - start_time
    #     #     logging.info(f"evaluate_data_groups took {elapsed:.3f} seconds ")
    #     #     energy_distance = {"value": observed, "std_err": se_observed, "p_value": 0.0}

    #     # else:
    #     start_time = time.perf_counter()
    #     energy, se_energy, pval = energy_test_permutation(
    #         train_data,
    #         test_data,
    #         num_permutations=cfg.energy_distance.num_permutations,
    #         K_cross=cfg.energy_distance.K_cross,
    #         K_within=cfg.energy_distance.K_within,
    #         batch_size=cfg.energy_distance.batch_size,
    #         seed=cfg.project.seed,
    #     )
    #     elapsed = time.perf_counter() - start_time
    #     logging.info(f"energy_test_permutation took {elapsed:.3f} seconds ")
    #     energy_distance = {"value": energy, "std_err": se_energy, "p_value": pval}
    #     results["energy_distance"] = energy_distance
    #     logging.info(
    #         f'Energy Distance: {energy_distance["value"]:.6f} ± {energy_distance["std_err"]:.6f}, p-value: {energy_distance["p_value"]:.6f}'
    #     )

    # if "vis_marginals" in evaluation_options:
    #     marginal_distributions = extract_marginal_distributions_for_visualization(
    #         train_data, cfg.vis_marginals, seed=cfg.project.seed
    #     )
    #     results["vis_marginals"] = marginal_distributions
    #     logging.info(f"Extracted marginal distributions for visualization.")

    # if "kl_divergence" in evaluation_options:
    #     start_time = time.perf_counter()
    #     # kl_estimate, (logp_test_vals, logp_train_vals), pca_obj = kde_kl_mc(train_data, test_data, n_components=cfg.kl_divergence.n_components, bandwidth=cfg.kl_divergence.bandwidth, bw_grid=cfg.kl_divergence.bw_grid, sample_size=cfg.kl_divergence.sample_size, rng=cfg.project.seed)
    #     kl_estimate, se, _ = pca_kde_kl_gpu(
    #         train_data,
    #         test_data,
    #         n_components=cfg.kl_divergence.n_components,
    #         bandwidth=cfg.kl_divergence.bandwidth,
    #         batch_ref=cfg.kl_divergence.sample_size,
    #     )
    #     elapsed = time.perf_counter() - start_time
    #     logging.info(f"kde_kl_mc took {elapsed:.3f} seconds ")
    #     kl_divergence = {
    #         "value": kl_estimate
    #     }  # , "logp_test_vals": logp_test_vals, "logp_train_vals": logp_train_vals}
    #     results["kl_divergence"] = kl_divergence
    #     logging.info(f'KL Divergence Estimate: {kl_divergence["value"]:.6f}')


    return results


# def evaluate_marginal_distributions(X, Y, cfg: DictConfig):
#     """
#     Quantifies the difference between marginal distributions of two datasets using energy distance.

#     Args:
#         X (np.ndarray): First dataset (samples, features).
#         Y (np.ndarray): Second dataset (samples, features).
#         cfg (DictConfig): Configuration object containing settings for marginal distribution evaluation.
#     """
#     distances = []
#     se_distances = []
#     for i in range(X.shape[1]):
#         if cfg.marginal_distributions.metric == "energy_distance":
#             dist, se_dist = monte_carlo_energy_distance(
#                 X[:, i : i + 1],
#                 Y[:, i : i + 1],
#                 K_cross=cfg.energy_distance.K_cross,
#                 K_within=cfg.energy_distance.K_within,
#                 batch_size=cfg.energy_distance.batch_size,
#                 seed=cfg.project.seed,
#             )
#         else:
#             raise ValueError(
#                 f"Unknown marginal distribution metric: {cfg.marginal_distributions.metric}, or not yet implemented."
#             )

#         distances.append(dist)
#         se_distances.append(se_dist)
    # return distances, se_distances


def extract_marginal_distributions_for_visualization(
    data: np.ndarray, vis_marginals_cfg: DictConfig, seed: Optional[int] = None
):
    """
    Extracts marginal distributions from the dataset for visualization purposes.

    Args:
        data (np.ndarray): The input data array.
        vis_marginals_cfg (DictConfig): Configuration for extracting marginal distributions for visualization, containing:
            sample_size (int): Number of samples to extract for each variable.
            variables (list): List of variable names to extract.
        seed (Optional[int]): Random seed for reproducibility.

    Returns:
        dict: A dictionary containing the extracted marginal distributions.
    """

    rng = np.random.default_rng(seed)
    i_idx = rng.integers(0, len(data), size=vis_marginals_cfg.sample_size)

    results = {}
    if "near_surface_specific_humidity" in vis_marginals_cfg.variables:
        results["near_surface_specific_humidity"] = data[:, 119][i_idx]

    if "near_surface_air_temperature" in vis_marginals_cfg.variables:
        results["near_surface_air_temperature"] = data[:, 59][i_idx]

    if "top_3_levels_specific_humidity" in vis_marginals_cfg.variables:
        for i in range(3):
            results[f"top_{i}_level_specific_humidity"] = data[:, i][i_idx]

    return results

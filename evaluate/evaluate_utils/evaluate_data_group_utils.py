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
        self,
        samples_size: int,
        metric_name: str,
        pca_components: Optional[int] = None,
        batch_size: Optional[int] = None,
        device: Optional[torch.device] = None,
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
            X = X[torch.randperm(X.shape[0])[: self.sample_size]]
            Y = Y[torch.randperm(Y.shape[0])[: self.sample_size]]
            logging.info(f"Subsampled data has shapes X: {X.shape}, Y: {Y.shape}")

        self.observed = self.metric_function(X, Y)
        return self.observed

    def permutation_test(self, X, Y, num_permutations=500):
        """
        Perform a permutation test to assess significance of the metric.
            H_0: X and Y are from the same distribution.
            H_1: X and Y are from different distributions.

        Args:
            X (np.ndarray): First dataset.
            Y (np.ndarray): Second dataset.
            num_permutations (int): Number of permutations to perform.

        Returns:
            pval (float): p-value from the permutation test.
        """
        if self.sample_size and self.sample_size < min(len(X), len(Y)):
            X = X[torch.randperm(X.shape[0])[: self.sample_size]]
            Y = Y[torch.randperm(Y.shape[0])[: self.sample_size]]
            logging.info(f"perm test subsample results X: {X.shape}, Y: {Y.shape}")
            self.sample_size = None  # Disable further sampling within metric functions

        if self.observed is None:
            self.observed = self.calculate(X, Y)

        pooled = torch.vstack([X, Y])
        n = X.shape[0]
        count = 0
        for i in range(num_permutations):
            idx = torch.randperm(pooled.shape[0])
            Xp = pooled[idx[:n]]
            Yp = pooled[idx[n:]]
            stat = self.metric_function(Xp, Yp)
            if stat >= self.observed:
                count += 1
            if i % 50 == 0:
                logging.info(
                    f"Permutation test progress: {i}/{num_permutations} permutations completed."
                )
        pval = (count + 1) / (num_permutations + 1)
        return pval

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
                raise ValueError(
                    "pca_components must be specified for KL Divergence metric."
                )
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

        if cfg.permutation_test > 0:
            start_time = time.perf_counter()
            pval = metric.permutation_test(
                trainset.input,
                testset.input,
                num_permutations=cfg.permutation_test,
            )
            elapsed = time.perf_counter() - start_time
            logging.info(
                f"Multivariate {cfg.metric_name} permutation test took {elapsed:.3f} seconds "
            )
            results["multivariate"]["p_value"] = pval
            logging.info(f"Multivariate {cfg.metric_name} p-value: {pval:.6f}")

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
                results["marginals"] = {
                    "marginal_distances": [],
                    "marginal_distribution_se": [],
                }
            results["marginals"]["marginal_distances"].append(observed_metric)
            results["marginals"]["marginal_distribution_se"].append(
                0.0
            )  # Placeholder for standard error
            logging.info(f"Marginal {i} {cfg.metric_name} Value: {observed_metric:.6f}")

    if "composition_metrics" in evaluation_options:
        num_features = trainset.input.shape[1]
        results["composition_metrics"] = []
        for i in range(num_features):
            trainset_feature = trainset.input[:, i]
            testset_feature = testset.input[:, i]
            composition_result = marginal_composition_evaluation(
                trainset_feature, testset_feature
            )

            results["composition_metrics"].append(composition_result)

    # if "vis_marginals" in evaluation_options:
    #     marginal_distributions = extract_marginal_distributions_for_visualization(
    #         train_data, cfg.vis_marginals, seed=cfg.project.seed
    #     )
    #     results["vis_marginals"] = marginal_distributions
    #     logging.info(f"Extracted marginal distributions for visualization.")

    return results


def marginal_composition_evaluation(train_data: torch.Tensor, test_data: torch.Tensor):
    """
    Evaluates the differences in marginal distributions between train and test data.
    """
    # Check overlapp
    mask = torch.isin(test_data, train_data)
    percentage = mask.float().mean() * 100.0

    # Check min and max of train and test for each feature
    train_min = train_data.min().item()
    train_max = train_data.max().item()
    test_min = test_data.min().item()
    test_max = test_data.max().item()
    if test_min >= train_min and test_max <= train_max:
        coverage = True
    else:
        coverage = False

    return {
        "overlap_percentage": percentage.item(),
        "coverage": coverage,
        "train_min": train_min,
        "train_max": train_max,
        "test_min": test_min,
        "test_max": test_max,
    }


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


# def permutation_test(
#     X,
#     Y,
#     num_permutations=500,
#     K_cross=2_000_000,
#     K_within=2_000_000,
#     batch_size=200000,
#     seed=None,
# ):
#     """
#     To provide a single value with significance, we can perform a permutation test.
#     H_0: X and Y are from the same distribution.
#     H_1: X and Y are from different distributions.
#     This function utilises the optimised monte_carlo_energy_distance function to allow for larger datasets.

#     Args:
#         X (np.ndarray): First dataset.
#         Y (np.ndarray): Second dataset.
#         num_permutations (int): Number of permutations to perform.
#         K_cross (int): Number of random pairs for cross term estimation.
#         K_within (int): Number of random pairs for within term estimation.
#         batch_size (int): Batch size for pair computations.
#         seed (int, optional): Random seed for reproducibility. Defaults to None.

#     Returns:
#         observed (float): Observed energy distance.
#         se_observed (float): Standard error of the observed energy distance.
#         pval (float): p-value from the permutation test.
#     """

#     rng = np.random.default_rng(seed)
#     observed, se_observed = monte_carlo_energy_distance(
#         X, Y, K_cross=K_cross, K_within=K_within, batch_size=batch_size, seed=seed
#     )
#     pooled = np.vstack([X, Y])
#     n = X.shape[0]
#     count = 0
#     for _ in range(num_permutations):
#         idx = rng.permutation(pooled.shape[0])
#         Xp = pooled[idx[:n]]
#         Yp = pooled[idx[n:]]
#         stat, se_energy = monte_carlo_energy_distance(
#             Xp, Yp, K_cross=K_cross, K_within=K_within, batch_size=batch_size, seed=seed
#         )
#         if stat >= observed:
#             count += 1
#     pval = (count + 1) / (num_permutations + 1)
#     return observed, se_observed, pval

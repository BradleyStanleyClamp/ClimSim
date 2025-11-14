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
from evaluate.evaluate_utils.kl_divergence import kde_kl_mc, pca_kde_kl_gpu
from evaluate.evaluate_utils.energy_distance import energy_distance


class MetricWrapper:
    def __init__(
        self, samples_size: int, n_samples: int, batch_size: int, metric_name: str, dataloader_cfg: DictConfig
    ):
        """
        Wrapper for different metrics to handle optimisations and significance testing
        Args:
            n_samples (int): Number of samples if doing sampling
            batch_size (int): Batch size for computations.
            metric_name (str): Name of the metric to use.
            dataloader_cfg (DictConfig): Configuration for the dataloader.
        """
        self.sample_size = samples_size
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.dataloader_cfg = dataloader_cfg
        self.metric_function = self._get_metric_function(metric_name)
        self.observed = None

    def calculate(self, X: Dataset, Y: Dataset) -> float:
        """
        Calculate the metric between datasets X and Y.
        """

        assert isinstance(X, Dataset) and isinstance(Y, Dataset), f"X and Y must be torch Datasets"

        # For now we just have functionality to take a single sample instead of multiple samples (monte carlo sampling)
        if self.sample_size and self.sample_size < len(X):
            if (
                hasattr(X, "sample")
                and callable(getattr(X, "sample"))
                and hasattr(Y, "sample")
                and callable(getattr(Y, "sample"))
            ):
                X.sample(self.sample_size)
                Y.sample(self.sample_size)
                assert len(X) == self.sample_size
                assert len(Y) == self.sample_size
            else:
                raise ValueError(
                    "Datasets must have a callable 'sample' method to perform sampling."
                )

        # For memory constraints we can do batchwise computation and utilise the torch dataloaders for efficiency
        if not self.batch_size:
            self.batch_size = max(len(X), len(Y))
            
        dataloader_X = DataLoader(
            X,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=int(self.dataloader_cfg.num_workers),
            persistent_workers=self.dataloader_cfg.persistent_workers,
            prefetch_factor=self.dataloader_cfg.prefetch_factor,
        )
        dataloader_Y = DataLoader(
            Y,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=int(self.dataloader_cfg.num_workers),
            persistent_workers=self.dataloader_cfg.persistent_workers,
            prefetch_factor=self.dataloader_cfg.prefetch_factor,
        )
        all_metrics = []
        for batch_X, batch_Y in zip(dataloader_X, dataloader_Y):
                batch_metric = self.metric_function(batch_X, batch_Y)
                all_metrics.append(batch_metric)

        return float(np.mean(all_metrics))


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
            return energy_distance
        else:
            raise ValueError(f"Unknown metric name: {metric_name}")


def evaluate_data_groups(
    cfg: DictConfig,
    train_data: np.ndarray,
    test_data: np.ndarray,
    evaluation_options: list[str],
):
    """
    Evaluate a data groups distribution shift compared to the test set using specified evaluation options.

    Args:
        cfg (DictConfig): Configuration object containing evaluation settings.
        train_data (np.ndarray)(samples, features): Training data.
        test_data (np.ndarray)(samples, features): Test data.
        evaluation_options (list[str]): List of evaluation options to perform.

    Returns:
        results (dict): Dictionary containing evaluation results for each option.

    """
    results = {}

    if "multivariate" in evaluation_options:
        metric = MetricWrapper(
            cfg.evaluate.sample_size,
            cfg.evaluate.n_samples,
            cfg.evaluate.batch_size,
            metric_name=cfg.metric_name,
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

    # if "marginal_distributions" in evaluation_options:
    #     marginal_distances, se_marginal_distances = evaluate_marginal_distributions(
    #         train_data, test_data, cfg
    #     )
    #     results["marginal_distributions"] = {
    #         "marginal_distances": marginal_distances,
    #         "marginal_distribution_se": se_marginal_distances,
    #     }
    #     logging.info(f"marginal distribution distances computed.")

    return results


def evaluate_marginal_distributions(X, Y, cfg: DictConfig):
    """
    Quantifies the difference between marginal distributions of two datasets using energy distance.

    Args:
        X (np.ndarray): First dataset (samples, features).
        Y (np.ndarray): Second dataset (samples, features).
        cfg (DictConfig): Configuration object containing settings for marginal distribution evaluation.
    """
    distances = []
    se_distances = []
    for i in range(X.shape[1]):
        if cfg.marginal_distributions.metric == "energy_distance":
            dist, se_dist = monte_carlo_energy_distance(
                X[:, i : i + 1],
                Y[:, i : i + 1],
                K_cross=cfg.energy_distance.K_cross,
                K_within=cfg.energy_distance.K_within,
                batch_size=cfg.energy_distance.batch_size,
                seed=cfg.project.seed,
            )
        else:
            raise ValueError(
                f"Unknown marginal distribution metric: {cfg.marginal_distributions.metric}, or not yet implemented."
            )

        distances.append(dist)
        se_distances.append(se_dist)
    return distances, se_distances


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

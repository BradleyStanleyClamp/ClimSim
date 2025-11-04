"""
Script containing utility functions for evaluating different data groups for distribution shifts.
"""
import logging
import time
import numpy as np
from omegaconf import DictConfig
from scipy.stats import wasserstein_distance_nd
# from torch import cdist
from scipy.spatial.distance import cdist
from tqdm import tqdm
import torch
import math
from typing import Optional
from evaluate.evaluate_utils.kl_divergence import kde_kl_mc, pca_kde_kl_gpu
def evaluate_data_groups(cfg: DictConfig, train_data: np.ndarray, test_data: np.ndarray, evaluation_options: list[str]):
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
    if 'energy_distance' in evaluation_options:
        # if cfg.testing.timings:
        #     start_time = time.perf_counter()
        #     observed, se_observed = monte_carlo_energy_distance(train_data, test_data, K_cross=cfg.energy_distance.K_cross, K_within=cfg.energy_distance.K_within, batch_size=cfg.energy_distance.batch_size, seed=cfg.project.seed)
        #     elapsed = time.perf_counter() - start_time
        #     logging.info(f"evaluate_data_groups took {elapsed:.3f} seconds ")
        #     energy_distance = {"value": observed, "std_err": se_observed, "p_value": 0.0}


        # else:
        start_time = time.perf_counter()
        energy, se_energy, pval = energy_test_permutation(train_data, test_data, num_permutations=cfg.energy_distance.num_permutations, K_cross=cfg.energy_distance.K_cross, K_within=cfg.energy_distance.K_within, batch_size=cfg.energy_distance.batch_size, seed=cfg.project.seed)
        elapsed = time.perf_counter() - start_time
        logging.info(f"energy_test_permutation took {elapsed:.3f} seconds ")
        energy_distance = {"value": energy, "std_err": se_energy, "p_value": pval}
        results['energy_distance'] = energy_distance
        logging.info(f'Energy Distance: {energy_distance["value"]:.6f} ± {energy_distance["std_err"]:.6f}, p-value: {energy_distance["p_value"]:.6f}')

    if  'vis_marginals' in evaluation_options:
        marginal_distributions = extract_marginal_distributions_for_visualization(train_data, cfg.vis_marginals, seed=cfg.project.seed)
        results['vis_marginals'] = marginal_distributions
        logging.info(f'Extracted marginal distributions for visualization.')

    if 'kl_divergence' in evaluation_options:
        start_time = time.perf_counter()
        # kl_estimate, (logp_test_vals, logp_train_vals), pca_obj = kde_kl_mc(train_data, test_data, n_components=cfg.kl_divergence.n_components, bandwidth=cfg.kl_divergence.bandwidth, bw_grid=cfg.kl_divergence.bw_grid, sample_size=cfg.kl_divergence.sample_size, rng=cfg.project.seed)
        kl_estimate, se, _ = pca_kde_kl_gpu(train_data, test_data, n_components=cfg.kl_divergence.n_components, bandwidth=cfg.kl_divergence.bandwidth, batch_ref=cfg.kl_divergence.sample_size)
        elapsed = time.perf_counter() - start_time
        logging.info(f"kde_kl_mc took {elapsed:.3f} seconds ")
        kl_divergence = {"value": kl_estimate} #, "logp_test_vals": logp_test_vals, "logp_train_vals": logp_train_vals}
        results['kl_divergence'] = kl_divergence
        logging.info(f'KL Divergence Estimate: {kl_divergence["value"]:.6f}')

    if 'marginal_distributions' in evaluation_options:
        marginal_distances, se_marginal_distances = evaluate_marginal_distributions(train_data, test_data, cfg)
        results['marginal_distributions'] = {'marginal_distances': marginal_distances, 'marginal_distribution_se': se_marginal_distances}
        logging.info(f'marginal distribution distances computed.')

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
        if cfg.marginal_distributions.metric == 'energy_distance':
            dist, se_dist = monte_carlo_energy_distance(X[:,i:i+1], Y[:, i:i+1], K_cross=cfg.energy_distance.K_cross, K_within=cfg.energy_distance.K_within, batch_size=cfg.energy_distance.batch_size, seed=cfg.project.seed)
        else:
            raise ValueError(f"Unknown marginal distribution metric: {cfg.marginal_distributions.metric}, or not yet implemented.")
        
        distances.append(dist)
        se_distances.append(se_dist)
    return distances, se_distances

def energy_test_permutation(X, Y, num_permutations=500, K_cross=2_000_000, K_within=2_000_000, batch_size=200000, seed=None):
    """
    To provide a single value with significance, we can perform a permutation test.
    H_0: X and Y are from the same distribution.
    H_1: X and Y are from different distributions.
    This function utilises the optimised monte_carlo_energy_distance function to allow for larger datasets.

    Args: 
        X (np.ndarray): First dataset.
        Y (np.ndarray): Second dataset.
        num_permutations (int): Number of permutations to perform.
        K_cross (int): Number of random pairs for cross term estimation.
        K_within (int): Number of random pairs for within term estimation.
        batch_size (int): Batch size for pair computations.
        seed (int, optional): Random seed for reproducibility. Defaults to None.
    
    Returns:
        observed (float): Observed energy distance.
        se_observed (float): Standard error of the observed energy distance.
        pval (float): p-value from the permutation test.
    """

    rng = np.random.default_rng(seed)
    observed, se_observed = monte_carlo_energy_distance(X, Y, K_cross=K_cross, K_within=K_within, batch_size=batch_size, seed=seed)
    pooled = np.vstack([X, Y])
    n = X.shape[0]
    count = 0
    for _ in range(num_permutations):
        idx = rng.permutation(pooled.shape[0])
        Xp = pooled[idx[:n]]
        Yp = pooled[idx[n:]]
        stat, se_energy = monte_carlo_energy_distance(Xp, Yp, K_cross=K_cross, K_within=K_within, batch_size=batch_size, seed=seed)
        if stat >= observed:
            count += 1
    pval = (count + 1) / (num_permutations + 1)
    return observed, se_observed, pval

def energy_distance_chunked(X, Y, chunk=1000):
    """
    Compute the energy distance between two datasets using chunked computation.
    Note: This is computationally intensive and is planned only to be used for a baseline comparison to optimised versions.

    Args:
        X (np.ndarray): First dataset.
        Y (np.ndarray): Second dataset.
        chunk (int): Chunk size for block computation.
    Returns:
        float: Energy distance between X and Y.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # logging.info(f'Computing energy distance on device: {device}')

    X = torch.tensor(X, dtype=torch.float32, device=device)
    Y = torch.tensor(Y, dtype=torch.float32, device=device)

    n, m = X.shape[0], Y.shape[0]

    # --- cross term ---
    total_xy = 0.0
    count_xy = 0
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        dXY_block = torch.cdist(X[start:end], Y)
        total_xy += dXY_block.sum().item()
        count_xy += dXY_block.numel()
    mean_xy = total_xy / count_xy

    # --- within-X term ---
    total_xx = 0.0
    count_xx = 0
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        dXX_block = torch.cdist(X[start:end], X)
        total_xx += dXX_block.sum().item()
        count_xx += dXX_block.numel()
    mean_xx = total_xx / count_xx

    # --- within-Y term ---
    total_yy = 0.0
    count_yy = 0
    for start in range(0, m, chunk):
        end = min(start + chunk, m)
        dYY_block = torch.cdist(Y[start:end], Y)
        total_yy += dYY_block.sum().item()
        count_yy += dYY_block.numel()
    mean_yy = total_yy / count_yy

    # --- energy distance formula ---
    return 2 * mean_xy - mean_xx - mean_yy

def estimate_pair_mean_distance_mc(X, Y, K=2_000_000, batch_size=200_000, device=None, seed=None):
    """
    Monte-Carlo estimate of mean_{i,j} ||X_i - Y_j|| by sampling K random pairs.
    Args:
        X, Y: numpy arrays or torch tensors; if numpy, converted to torch.float32 on device.
        K: number of random pairs to sample (total).
        batch_size: pairs computed per matmul call (must fit on device memory).
        device: torch device to use (CPU or GPU). If None, auto-detects.
        seed: random seed for reproducibility.

    Returns:
        mean: estimated mean distance.
        se: standard error of the mean estimate.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Convert to torch tensors on device
    X_t = torch.tensor(X, dtype=torch.float32, device=device) if not torch.is_tensor(X) else X.to(device).float()
    Y_t = torch.tensor(Y, dtype=torch.float32, device=device) if not torch.is_tensor(Y) else Y.to(device).float()
    n, m = X_t.shape[0], Y_t.shape[0]

    rng = np.random.default_rng(seed)
    total_sum = torch.tensor(0.0, device=device)
    total_sum_sq = torch.tensor(0.0, device=device)
    done = 0

    while done < K:
        this_batch = min(batch_size, K - done)
        # sample indices
        i_idx = rng.integers(0, n, size=this_batch)
        j_idx = rng.integers(0, m, size=this_batch)
        Ai = X_t[i_idx]  # (this_batch, d)
        Bj = Y_t[j_idx]  # (this_batch, d)

        # squared distances via matmul trick
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
        An = (Ai * Ai).sum(dim=1)  # (b,)
        Bn = (Bj * Bj).sum(dim=1)
        AB = (Ai * Bj).sum(dim=1)
        D2 = An + Bn - 2.0 * AB
        D2 = torch.clamp(D2, min=0.0)
        D = torch.sqrt(D2)
        s = D.sum()
        total_sum = total_sum + s
        total_sum_sq = total_sum_sq + (D * D).sum()   # to compute variance of D if needed
        done += this_batch
        # logging.info(f"Estimated {done}/{K} pairs for mean distance MC")

    mean = (total_sum / float(K)).item()
    # approximate standard error of the mean
    var = (total_sum_sq / float(K) - (total_sum / float(K))**2).clamp(min=0.0)
    se = math.sqrt((var / K).cpu().item())
    return mean, se

def monte_carlo_energy_distance(X, Y, K_cross=2_000_000, K_within=2_000_000, batch_size=200_000, device=None, seed=None):
    """
    Calculates the energy distance between two distributions X and Y. With the following optimisations:
    - Monte Carlo estimation of the mean pairwise distances by sampling K random pairs.
    - Batch computation of pairwise distances to fit in memory.
    - Option to use GPU if available.
    - squared distances via matmul trick for efficiency.

    Args:
        X: First input distribution (numpy array or torch tensor).
        Y: Second input distribution (numpy array or torch tensor).
        K_cross: Number of random pairs to sample for cross distribution.
        K_within: Number of random pairs to sample for within distribution.
        batch_size: Number of pairs to compute per batch.
        device: Device to perform computation on (CPU or GPU).
        seed: Random seed for reproducibility.

    Returns:
        energy: Estimated energy distance between distributions X and Y.
        se_energy: Standard error of the estimated energy distance.
    """
    mean_xy, se_xy = estimate_pair_mean_distance_mc(X, Y, K=K_cross, batch_size=batch_size, device=device, seed=seed)
    mean_xx, se_xx = estimate_pair_mean_distance_mc(X, X, K=K_within, batch_size=batch_size, device=device, seed=(None if seed is None else seed+1))
    mean_yy, se_yy = estimate_pair_mean_distance_mc(Y, Y, K=K_within, batch_size=batch_size, device=device, seed=(None if seed is None else seed+2))
    energy = 2*mean_xy - mean_xx - mean_yy
    se_energy = math.sqrt((2*se_xy)**2 + se_xx**2 + se_yy**2)
    return energy, se_energy

def extract_marginal_distributions_for_visualization(data: np.ndarray, vis_marginals_cfg: DictConfig, seed: Optional[int] = None):
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
        results["near_surface_specific_humidity"] = data[:,119][i_idx]

    if "near_surface_air_temperature" in vis_marginals_cfg.variables:
        results["near_surface_air_temperature"] = data[:,59][i_idx]

    return results
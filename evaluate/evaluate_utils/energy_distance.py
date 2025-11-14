import math
import numpy as np
import torch

from evaluate.evaluate_utils.general_metric_utils import squared_euclidean_distance_matrix_matmul_trick

def energy_distance(X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Compute the energy distance between two datasets X and Y (on GPU, if possible).
    Args:
        X (torch.Tensor): First dataset.
        Y (torch.Tensor): Second dataset.
    Returns:
        float: Energy distance between X and Y.
    """
    assert isinstance(X, torch.Tensor) and isinstance(Y, torch.Tensor), f"Inputs must be torch Tensors, got {type(X)} and {type(Y)}"
    device = X.device

    n, m = X.shape[0], Y.shape[0]

    # --- cross term ---
    dXY = squared_euclidean_distance_matrix_matmul_trick(X, Y)
    mean_xy = torch.sqrt(dXY).mean().item()

    # --- within-X term ---
    dXX = squared_euclidean_distance_matrix_matmul_trick(X, X)
    mean_xx = torch.sqrt(dXX + torch.eye(n, device=device) * 1e-8).mean().item()

    # --- within-Y term ---
    dYY = squared_euclidean_distance_matrix_matmul_trick(Y, Y)
    mean_yy = torch.sqrt(dYY + torch.eye(m, device=device) * 1e-8).mean().item()

    # --- energy distance formula ---
    energy_dist = 2 * mean_xy - mean_xx - mean_yy
    return energy_dist




def energy_test_permutation(
    X,
    Y,
    num_permutations=500,
    K_cross=2_000_000,
    K_within=2_000_000,
    batch_size=200000,
    seed=None,
):
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
    observed, se_observed = monte_carlo_energy_distance(
        X, Y, K_cross=K_cross, K_within=K_within, batch_size=batch_size, seed=seed
    )
    pooled = np.vstack([X, Y])
    n = X.shape[0]
    count = 0
    for _ in range(num_permutations):
        idx = rng.permutation(pooled.shape[0])
        Xp = pooled[idx[:n]]
        Yp = pooled[idx[n:]]
        stat, se_energy = monte_carlo_energy_distance(
            Xp, Yp, K_cross=K_cross, K_within=K_within, batch_size=batch_size, seed=seed
        )
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


def estimate_pair_mean_distance_mc(
    X, Y, K=2_000_000, batch_size=200_000, device=None, seed=None
):
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
    X_t = (
        torch.tensor(X, dtype=torch.float32, device=device)
        if not torch.is_tensor(X)
        else X.to(device).float()
    )
    Y_t = (
        torch.tensor(Y, dtype=torch.float32, device=device)
        if not torch.is_tensor(Y)
        else Y.to(device).float()
    )
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
        total_sum_sq = (
            total_sum_sq + (D * D).sum()
        )  # to compute variance of D if needed
        done += this_batch
        # logging.info(f"Estimated {done}/{K} pairs for mean distance MC")

    mean = (total_sum / float(K)).item()
    # approximate standard error of the mean
    var = (total_sum_sq / float(K) - (total_sum / float(K)) ** 2).clamp(min=0.0)
    se = math.sqrt((var / K).cpu().item())
    return mean, se


def monte_carlo_energy_distance(
    X,
    Y,
    K_cross=2_000_000,
    K_within=2_000_000,
    batch_size=200_000,
    device=None,
    seed=None,
):
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
    mean_xy, se_xy = estimate_pair_mean_distance_mc(
        X, Y, K=K_cross, batch_size=batch_size, device=device, seed=seed
    )
    mean_xx, se_xx = estimate_pair_mean_distance_mc(
        X,
        X,
        K=K_within,
        batch_size=batch_size,
        device=device,
        seed=(None if seed is None else seed + 1),
    )
    mean_yy, se_yy = estimate_pair_mean_distance_mc(
        Y,
        Y,
        K=K_within,
        batch_size=batch_size,
        device=device,
        seed=(None if seed is None else seed + 2),
    )
    energy = 2 * mean_xy - mean_xx - mean_yy
    se_energy = math.sqrt((2 * se_xy) ** 2 + se_xx**2 + se_yy**2)
    return energy, se_energy

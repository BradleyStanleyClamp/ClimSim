import math
from typing import Optional
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from evaluate.evaluate_utils.general_metric_utils import (
    squared_euclidean_distance_matrix_matmul_trick,
)


class EnergyDistanceMetric:
    def __init__(self, batch_size: Optional[int] = None):
        """
        Energy distance metric class for computing energy distance between two datasets.

        Args:
            batch_size (int, optional): Batch size for processing. If None, uses full dataset size.
        """
        self.batch_size = batch_size
    def __call__(self, X_tensor: torch.Tensor, Y_tensor: torch.Tensor) -> float:
        return energy_distance(X_tensor, Y_tensor, batch_size=self.batch_size)


def energy_distance(
    X_tensor: torch.Tensor,
    Y_tensor: torch.Tensor,
    batch_size: Optional[int] = None,
) -> float:
    """

    Batched energy distance computation between two tensors.
    - Uses pin_memory + non_blocking copies for GPU
    - Uses torch.cdist on CUDA, matmul trick on CPU
    - Accumulates on-device to reduce .item() overhead
    - Reuses the same batching logic as DataLoader, but directly on tensors
    - Matches the 'biased' averaging (denominators n*n and m*m).

    Args:
        X_tensor (torch.Tensor): First dataset tensor.
        Y_tensor (torch.Tensor): Second dataset tensor.
        batch_size (int, optional): Batch size for processing. 

    Returns:
        float: Energy distance between X_tensor and Y_tensor.
    """
    assert isinstance(X_tensor, torch.Tensor) and isinstance(
        Y_tensor, torch.Tensor
    ), f"Inputs must be torch Tensors, got {type(X_tensor)} and {type(Y_tensor)}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # sensible default to avoid accidentally using too-large batches
    if batch_size is False or batch_size is None:
        batch_size = min(1024, max(len(X_tensor), len(Y_tensor)))

    # pin_memory is only relevant when moving to CUDA
    pin_memory = True if device.type == "cuda" else False

    # Ensure float (the cdist/matmul expects float)
    if not torch.is_floating_point(X_tensor):
        X_tensor = X_tensor.float()
    if not torch.is_floating_point(Y_tensor):
        Y_tensor = Y_tensor.float()

    n = X_tensor.shape[0]
    m = Y_tensor.shape[0]
    if n == 0 or m == 0:
        raise ValueError("Datasets must be non-empty")

    use_cdist = device.type == "cuda"

    # helper to move a CPU tensor (batch) to device efficiently
    def to_device(batch: torch.Tensor):
        if pin_memory and batch.device.type == "cpu":
            # pin the host memory and use non_blocking copy
            batch = batch.pin_memory()
        return batch.to(device=device, non_blocking=pin_memory)

    # accumulators on device
    cross_sum_t = torch.tensor(0.0, device=device)
    xx_sum_t = torch.tensor(0.0, device=device)
    yy_sum_t = torch.tensor(0.0, device=device)

    # --- cross term: sum_{i in X, j in Y} ||x_i - y_j|| ---
    with torch.no_grad():
        for i in range(0, n, batch_size):
            Xi = X_tensor[i : i + batch_size]
            Xi = to_device(Xi)
            for j in range(0, m, batch_size):
                Yj = Y_tensor[j : j + batch_size]
                Yj = to_device(Yj)
                if use_cdist:
                    d_block = torch.cdist(Xi, Yj, p=2)  # distances
                else:
                    d2 = squared_euclidean_distance_matrix_matmul_trick(Xi, Yj)
                    d_block = torch.sqrt(d2)
                cross_sum_t += d_block.sum()

    mean_xy = (cross_sum_t / (n * m)).item()

    # --- within-X term: biased (include diagonal), divide by n*n ---
    # iterate i over blocks, j from i to end (upper triangular blocks)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            Xi = X_tensor[i : i + batch_size]
            Xi = to_device(Xi)
            i_idx = i // batch_size
            for j in range(i, n, batch_size):
                Xj = X_tensor[j : j + batch_size]
                Xj = to_device(Xj)
                if use_cdist:
                    d_block = torch.cdist(Xi, Xj, p=2)
                else:
                    d2 = squared_euclidean_distance_matrix_matmul_trick(Xi, Xj)
                    d_block = torch.sqrt(d2)
                if i == j:
                    # diagonal block: include diagonal zeros (biased formula)
                    xx_sum_t += d_block.sum()
                else:
                    # off-diagonal upper triangular block contributes twice
                    xx_sum_t += 2.0 * d_block.sum()

    mean_xx = (xx_sum_t / (n * n)).item()

    # --- within-Y term: biased (include diagonal), divide by m*m ---
    with torch.no_grad():
        for i in range(0, m, batch_size):
            Yi = Y_tensor[i : i + batch_size]
            Yi = to_device(Yi)
            for j in range(i, m, batch_size):
                Yj = Y_tensor[j : j + batch_size]
                Yj = to_device(Yj)
                if use_cdist:
                    d_block = torch.cdist(Yi, Yj, p=2)
                else:
                    d2 = squared_euclidean_distance_matrix_matmul_trick(Yi, Yj)
                    d_block = torch.sqrt(d2)
                if i == j:
                    yy_sum_t += d_block.sum()
                else:
                    yy_sum_t += 2.0 * d_block.sum()

    mean_yy = (yy_sum_t / (m * m)).item()

    energy_dist = 2 * mean_xy - mean_xx - mean_yy
    return float(energy_dist)


def _get_tensor_from_dataset(D):
        t = getattr(D, "input", None)
        if t is not None:
            # if it's numpy array, convert
            if not isinstance(t, torch.Tensor):
                t = torch.as_tensor(t)
            return t
        # fallback: materialize (only if attribute not present)
        # assume dataset[i] returns tensor or (tensor, label)
        first = D[0]
        if isinstance(first, (tuple, list)):
            return torch.stack([D[i][0] for i in range(len(D))])
        else:
            return torch.stack([D[i] for i in range(len(D))])



#############################################
### Obsolete functions kept for reference ###
#############################################


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

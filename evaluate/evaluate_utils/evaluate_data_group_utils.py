"""
Script containing utility functions for evaluating different data groups for distribution shifts.
"""
import logging
import numpy as np
from scipy.stats import wasserstein_distance_nd
# from torch import cdist
from scipy.spatial.distance import cdist
from tqdm import tqdm
import torch
import math
from typing import Optional

def evaluate_data_groups(train_input_group: np.ndarray, train_target_group: np.ndarray, test_input: np.ndarray, test_target: np.ndarray, evaluation_options: list[str], chunk: int = 1000):
    """
    Evaluate a data groups distribution shift compared to the test set using specified evaluation options. This will look at three different distributions; train only, test only and the combination of the two.

    Args:
        train_input_group (np.ndarray) (samples, input_features): Input data for the training data group.
        train_target_group (np.ndarray) (samples, target_features): Target data for the training data group.
        test_input (np.ndarray) (samples, input_features): Input data for the test set.
        test_target (np.ndarray) (samples, target_features): Target data for the test set.
        evaluation_options (list[str]): List of evaluation options to perform.
        chunk (int): Chunk size for block computation. Defaults to 1000.
  
    """
    # Evaluate input distribution shift
    print("Evaluating input distribution shift:")
    return evaluate_distribution_shift(train_input_group, test_input, evaluation_options, chunk)

    # # Evaluate target distribution shift
    # print("Evaluating target distribution shift:")
    # evaluate_distribution_shift(train_target_group, test_target, evaluation_options)

    # # Evaluate combined distribution shift
    # print("Evaluating combined input-target distribution shift:")
    # train_combined = np.hstack([train_input_group, train_target_group])
    # test_combined = np.hstack([test_input, test_target])
    # evaluate_distribution_shift(train_combined, test_combined, evaluation_options)


def evaluate_distribution_shift(train_data: np.ndarray, test_data: np.ndarray, evaluation_options: list[str], chunk: int = 1000):
    """
    Evaluate the distribution shift between training and test data using a specified option.

    Args:
        train_data (np.ndarray)(samples, features): Training data.
        test_data (np.ndarray)(samples, features): Test data.
        evaluation_options (list[str]): List of evaluation options to perform.
        chunk (int): Chunk size for block computation. Defaults to 1000.
    Returns:
        Various: Results from the evaluation metrics.

    
    """
    # train_distribution, test_distribution, bin_edges = compute_distributions(train_data, test_data)

    if 'energy_distance' in evaluation_options:
        # Compute energy distance between train and test data distributions
        # return energy_distance_chunked(train_data, test_data, chunk=chunk)
        # mean, std = estimate_pair_mean_distance_mc(train_data, test_data, K=2000000, batch_pairs=chunk, device=None, seed=0)
        # return mean, std

        # return mean_pairwise_distance_full(train_data, test_data, batch_A=chunk, batch_B=chunk)
        energy, se_energy, (mean_xy, mean_xx, mean_yy) = monte_carlo_energy_distance(train_data, test_data, K_cross=2_000_000, K_within=2_000_000, batch_pairs=chunk, device=None, seed=0)
        logging.info(f'Energy Distance: {energy} ± {se_energy} (mean_xy: {mean_xy}, mean_xx: {mean_xx}, mean_yy: {mean_yy})')
        return energy
        # observed, pval = energy_test_permutation(train_data, test_data, num_permutations=500, chunk=1000)
        # observed, pval = energy_test_permutation(train_data, test_data, num_permutations=100)
        # logging.info(f'Energy Distance: {ed_value}, Observed: {observed}, p-value: {pval}')
        # return observed, pval

    # if 'EMD' in evaluation_options:
    #     # Compute Earth Mover's Distance (EMD) between train and test data distributions
    #     emd_value = earth_movers_distance(train_distribution, test_distribution, bin_edges)
    #     print(f'EMD: {emd_value}')

    # if 'KL' in evaluation_options:
    #     # Compute Kullback-Leibler (KL) divergence between train and test data distributions
    #     pass



def earth_movers_distance(train_data: np.ndarray, test_data: np.ndarray, bin_edges: list) -> float:
    """
    Compute the Earth Mover's Distance (EMD) between training and test data distributions.

    Args:
        train_data (np.ndarray): Training data.
        test_data (np.ndarray): Test data.
        bin_edges (list): List of bin edges used for each feature.

    Returns:
        float: Computed EMD value.
    """
    return wasserstein_distance_nd(train_data, test_data, bin_edges)


def energy_distance(X, Y, metric='euclidean', chunk=None):
    """
    Compute energy distance between X (nxd) and Y (mxd).
    If memory is limited, set chunk to an integer to compute dXY in blocks.

    Args:
        X (np.ndarray): First dataset.
        Y (np.ndarray): Second dataset.
        metric (str): Distance metric to use.
        chunk (int, optional): Chunk size for block computation. Defaults to None.
    Returns:
        float: Energy distance between X and Y.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n, m = X.shape[0], Y.shape[0]

    # cross-term 2 * mean_{i,j} ||X_i - Y_j||
    if chunk is None:
        dXY = cdist(X, Y, metric)
        term_xy = 2.0 * dXY.mean()
    else:
        # chunked computation to save memory
        s = 0.0
        count = 0
        for start in range(0, n, chunk):
            end = min(n, start + chunk)
            block = cdist(X[start:end], Y, metric)
            s += block.sum()
            count += block.size
        term_xy = 2.0 * (s / count)

    # within-X
    dXX = cdist(X, X, metric)
    term_xx = dXX.mean()

    # within-Y
    dYY = cdist(Y, Y, metric)
    term_yy = dYY.mean()

    return term_xy - term_xx - term_yy

def energy_test_permutation(X, Y, num_permutations=500, chunk=None, random_state=None):
    """
    To provide a single value with significance, we can perform a permutation test.
    """

    rng = np.random.default_rng(random_state)
    observed = energy_distance_chunked(X, Y, chunk=chunk)
    pooled = np.vstack([X, Y])
    n = X.shape[0]
    count = 0
    for _ in tqdm(range(num_permutations)):
        idx = rng.permutation(pooled.shape[0])
        Xp = pooled[idx[:n]]
        Yp = pooled[idx[n:]]
        stat = energy_distance_chunked(Xp, Yp, chunk=chunk)
        if stat >= observed:
            count += 1
    pval = (count + 1) / (num_permutations + 1)
    return observed, pval

def energy_distance_chunked(X, Y, chunk=1000):
    """
    Compute the energy distance between two datasets using chunked computation.

    Args:
        X (np.ndarray): First dataset.
        Y (np.ndarray): Second dataset.
        chunk (int): Chunk size for block computation.
    Returns:
        float: Energy distance between X and Y.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f'Computing energy distance on device: {device}')

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


def estimate_pair_mean_distance_mc(X, Y, K=2_000_000, batch_pairs=200_000, device=None, seed=None):
    """
    Monte-Carlo estimate of mean_{i,j} ||X_i - Y_j|| by sampling K random pairs.
    X, Y: numpy arrays or torch tensors; if numpy, converted to torch.float32 on device.
    K: number of random pairs to sample (total).
    batch_pairs: pairs computed per matmul call (must fit on device memory).
    Returns estimated mean (float) and standard error (float).
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
        this_batch = min(batch_pairs, K - done)
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

    mean = (total_sum / float(K)).item()
    # approximate standard error of the mean
    var = (total_sum_sq / float(K) - (total_sum / float(K))**2).clamp(min=0.0)
    se = math.sqrt((var / K).cpu().item())
    return mean, se


def mean_pairwise_distance_full(
    A,
    B,
    batch_A: int = 8192,
    batch_B: int = 8192,
    device: Optional[torch.device] = None,
    dtype=torch.float32,
    exclude_diagonal: bool = False,
):
    """
    Compute the exact mean pairwise Euclidean distance between all rows of A and all rows of B,
    using batched matrix-multiplication (no sampling).

    Args:
        A: numpy array or torch tensor of shape (n, d)
        B: numpy array or torch tensor of shape (m, d)
        batch_A: number of rows of A to process per outer batch
        batch_B: number of rows of B to process per inner batch
        device: torch.device (defaults to cuda if available else cpu)
        dtype: torch dtype for computations (default float32)
        exclude_diagonal: only meaningful when A and B are the same tensor (same object or identical values).
                          If True and A and B refer to the same dataset, the mean will be computed over
                          off-diagonal pairs only (useful for unbiased within-group term).

    Returns:
        mean_distance: float, mean over all ||a_i - b_j|| (or over i!=j if exclude_diagonal=True and A==B)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # convert to torch tensors on device
    if not torch.is_tensor(A):
        A_t = torch.tensor(A, dtype=dtype, device=device)
    else:
        A_t = A.to(device).to(dtype)

    if not torch.is_tensor(B):
        B_t = torch.tensor(B, dtype=dtype, device=device)
    else:
        B_t = B.to(device).to(dtype)

    n, d = A_t.shape
    m, d2 = B_t.shape
    assert d == d2, "feature dimension must match"

    # precompute norms for batching convenience? we compute per-block norms below
    total_sum = torch.tensor(0.0, dtype=dtype, device=device)
    total_count = 0

    # We'll loop over A in outer batches and B in inner batches to keep memory bounded.
    # This handles the general case including A == B and very large n,m.
    for i in range(0, n, batch_A):
        Ai = A_t[i : min(i + batch_A, n)]       # (ba, d)
        Ai_sq = (Ai * Ai).sum(dim=1).unsqueeze(1)  # (ba, 1)

        for j in range(0, m, batch_B):
            Bj = B_t[j : min(j + batch_B, m)]     # (bb, d)
            Bj_sq = (Bj * Bj).sum(dim=1).unsqueeze(0)  # (1, bb)

            # compute AB = Ai @ Bj^T  -> (ba, bb)
            AB = Ai @ Bj.t()

            # squared distances: Anorm + Bnorm - 2*AB
            D2 = Ai_sq + Bj_sq - 2.0 * AB
            # numerical stability clamp
            D2 = torch.clamp(D2, min=0.0)

            # distances
            D = torch.sqrt(D2)

            # If A and B are actually identical object (same memory) and the block aligns with itself,
            # we may be including diagonal elements. We'll handle diagonal exclusion after loops by adjusting count
            # because diagonal distances are zero -> they do not affect sum, only count.
            block_sum = D.sum()
            total_sum += block_sum
            total_count += D.numel()

    # If user requested exclude_diagonal and A and B are same (identical object or same shapes and contents),
    # we will adjust the count to exclude n diagonal terms (which are zero in sum).
    # Warning: we check identity by checking object identity OR if shapes equal and content equal (costly),
    # so the safest is to set exclude_diagonal only when you passed the same tensor object for A and B.
    if exclude_diagonal:
        # We only adjust if n == m and A and B refer to the same data (best if caller passed same object)
        if n == m:
            # Reduce count by n (diagonal entries). They contributed zero to total_sum so sum is fine.
            total_count -= n
        else:
            raise ValueError("exclude_diagonal=True asked but A and B have different sizes (cannot exclude diagonal)")

    # Convert to python float
    mean_dist = (total_sum / float(total_count)).item()
    return mean_dist


def monte_carlo_energy_distance(X, Y, K_cross=2_000_000, K_within=2_000_000, batch_pairs=200_000, device=None, seed=None):
    mean_xy, se_xy = estimate_pair_mean_distance_mc(X, Y, K=K_cross, batch_pairs=batch_pairs, device=device, seed=seed)
    mean_xx, se_xx = estimate_pair_mean_distance_mc(X, X, K=K_within, batch_pairs=batch_pairs, device=device, seed=(None if seed is None else seed+1))
    mean_yy, se_yy = estimate_pair_mean_distance_mc(Y, Y, K=K_within, batch_pairs=batch_pairs, device=device, seed=(None if seed is None else seed+2))
    energy = 2*mean_xy - mean_xx - mean_yy
    se_energy = math.sqrt((2*se_xy)**2 + se_xx**2 + se_yy**2)
    return energy, se_energy, (mean_xy, mean_xx, mean_yy)
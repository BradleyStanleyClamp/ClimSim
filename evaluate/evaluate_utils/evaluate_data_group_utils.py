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

def evaluate_data_groups(train_input_group: np.ndarray, train_target_group: np.ndarray, test_input: np.ndarray, test_target: np.ndarray, evaluation_options: list[str]):
    """
    Evaluate a data groups distribution shift compared to the test set using specified evaluation options. This will look at three different distributions; train only, test only and the combination of the two.

    Args:
        train_input_group (np.ndarray) (samples, input_features): Input data for the training data group.
        train_target_group (np.ndarray) (samples, target_features): Target data for the training data group.
        test_input (np.ndarray) (samples, input_features): Input data for the test set.
        test_target (np.ndarray) (samples, target_features): Target data for the test set.
        evaluation_options (list[str]): List of evaluation options to perform.
  
    """
    # Evaluate input distribution shift
    print("Evaluating input distribution shift:")
    return evaluate_distribution_shift(train_input_group, test_input, evaluation_options)

    # # Evaluate target distribution shift
    # print("Evaluating target distribution shift:")
    # evaluate_distribution_shift(train_target_group, test_target, evaluation_options)

    # # Evaluate combined distribution shift
    # print("Evaluating combined input-target distribution shift:")
    # train_combined = np.hstack([train_input_group, train_target_group])
    # test_combined = np.hstack([test_input, test_target])
    # evaluate_distribution_shift(train_combined, test_combined, evaluation_options)


def evaluate_distribution_shift(train_data: np.ndarray, test_data: np.ndarray, evaluation_options: list[str]):
    """
    Evaluate the distribution shift between training and test data using a specified option.

    Args:
        train_data (np.ndarray)(samples, features): Training data.
        test_data (np.ndarray)(samples, features): Test data.
        evaluation_options (list[str]): List of evaluation options to perform.

    
    """
    # train_distribution, test_distribution, bin_edges = compute_distributions(train_data, test_data)

    if 'energy_distance' in evaluation_options:
        # Compute energy distance between train and test data distributions
        ed_value = energy_distance_chunked(train_data, test_data, chunk=500)
        # observed, pval = energy_test_permutation(train_data, test_data, num_permutations=100)
        # logging.info(f'Energy Distance: {ed_value}, Observed: {observed}, p-value: {pval}')
        return ed_value

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
    observed = energy_distance(X, Y, chunk=chunk)
    pooled = np.vstack([X, Y])
    n = X.shape[0]
    count = 0
    for _ in tqdm(range(num_permutations)):
        idx = rng.permutation(pooled.shape[0])
        Xp = pooled[idx[:n]]
        Yp = pooled[idx[n:]]
        stat = energy_distance(Xp, Yp, chunk=chunk)
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
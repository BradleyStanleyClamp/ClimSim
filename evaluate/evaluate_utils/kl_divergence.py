"""
Script containing code for calculating the Kullback-Leibler (KL) divergence between two probability distributions.
"""

import logging
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import time 
import torch
from typing import Optional, Tuple
import math


class KLDivergenceMetric:
    def __init__(
        self,
        batch_size: int,
        n_components: Optional[int] = None,
        bandwidth: Optional[float] = None,
        device: Optional[torch.device] = None,
        leave_one_out: bool = True,
    ):
        """
        KL divergence estimate KL(P_Y || P_X) via Gaussian KDE on (optionally PCA-reduced) data.

        Args:
            batch_size: preferred batch size for query dimension (will be used for both ref/query blocks).
            n_components: if not None and < data dim, perform PCA to this many components.
            bandwidth: if None, will use Scott's rule on the (projected) data.
            device: torch.device or None (use CUDA if available).
            leave_one_out: whether to use leave-one-out when evaluating KDE on the same set (Y evaluated under KDE fitted on Y).
        """
        self.batch_size = int(batch_size)
        self.n_components = n_components
        self.bandwidth = bandwidth
        self.leave_one_out = leave_one_out
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    def __call__(self, X_tensor: torch.Tensor, Y_tensor: torch.Tensor) -> Tuple[float, float, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns: (kl_estimate, se_of_estimate, (logp_Y, logp_X))
          - kl_estimate is KL(P_Y || P_X) estimated as mean_y[ log p_hat_Y(y) - log p_hat_X(y) ]
          - se_of_estimate is the standard error of the per-point diffs (std / sqrt(n))
          - logp_Y, logp_X are per-point log densities (on CPU)
        """
        device = self.device
        X = X_tensor.to(device=device, dtype=torch.float32)
        Y = Y_tensor.to(device=device, dtype=torch.float32)

        # Optionally PCA-reduce
        if self.n_components is not None and self.n_components < X.shape[1] and X.shape[1] > 1:
            Xp, Yp, components, mean = pca_gpu(X, Y, n_components=self.n_components, device=device)
            X = Xp
            Y = Yp

        # choose bandwidth if not provided: Scott's rule on pooled data (projected)
        if self.bandwidth is None:
            pooled = torch.cat([X, Y], dim=0)
            n, d = pooled.shape
            # sample std per dimension then average (simple heuristic)
            sd = pooled.std(dim=0, unbiased=True).mean().item()
            # Scott's rule: h = sd * n^{-1/(d+4)}
            self.bandwidth = float(sd * (n ** (-1.0 / (d + 4.0)) + 1e-12))
            logging.info(f"Selected bandwidth {self.bandwidth:.4f} using Scott's rule on pooled data.")

        # compute log p_train(y) (X_ref = X, X_query = Y)
        logp_X = kde_logdensity_gpu(
            X_ref=X,
            X_query=Y,
            bandwidth=self.bandwidth,
            device=device,
            batch_ref=self.batch_size,
            batch_query=self.batch_size,
            eps=1e-12,
            leave_one_out=False,  # not relevant (ref != query)
        )

        # compute log p_test(y) (X_ref = Y, X_query = Y) — use leave-one-out to avoid self-contrib
        logp_Y = kde_logdensity_gpu(
            X_ref=Y,
            X_query=Y,
            bandwidth=self.bandwidth,
            device=device,
            batch_ref=self.batch_size,
            batch_query=self.batch_size,
            eps=1e-12,
            leave_one_out=self.leave_one_out,
        )

        # KL estimate
        diff = (logp_Y - logp_X).double()
        kl_est = float(diff.mean().item())
        se = float(diff.std(unbiased=True).item() / math.sqrt(diff.numel()))

        return kl_est, se, (logp_Y.cpu(), logp_X.cpu())


def kde_logdensity_gpu(
    X_ref: torch.Tensor,
    X_query: torch.Tensor,
    bandwidth: float,
    device: Optional[torch.device] = None,
    batch_ref: int = 65536,
    batch_query: int = 8192,
    eps: float = 1e-12,
    leave_one_out: bool = False,
) -> torch.Tensor:
    """
    Compute log KDE density estimates log p_hat(x_query) using Gaussian kernel with bandwidth `bandwidth`.
    Supports leave-one-out if X_ref is the same object as X_query (or if indices match).
    """
    if device is None:
        device = X_ref.device

    X_ref = X_ref.to(device=device, dtype=torch.float32)
    X_query = X_query.to(device=device, dtype=torch.float32)

    n_ref, d = X_ref.shape
    n_query = X_query.shape[0]

    # constant normalization term for Gaussian kernel
    const_term = -0.5 * d * math.log(2.0 * math.pi) - d * math.log(bandwidth)
    inv_two_h2 = 1.0 / (2.0 * (bandwidth ** 2))

    results = []
    # process queries in blocks
    for qstart in range(0, n_query, batch_query):
        qend = min(n_query, qstart + batch_query)
        Q = X_query[qstart:qend]  # (bq, d)
        Q_norm = (Q * Q).sum(dim=1)  # (bq,)

        # initialize per-query accumulated log-sum using -inf
        acc_logsum = torch.full((qend - qstart,), -float("inf"), device=device, dtype=torch.float32)

        for rstart in range(0, n_ref, batch_ref):
            rend = min(n_ref, rstart + batch_ref)
            R = X_ref[rstart:rend]  # (br, d)
            R_norm = (R * R).sum(dim=1)  # (br,)

            # compute D2 = ||Q||^2 + ||R||^2 - 2 Q R^T
            # Q_norm[:,None] + R_norm[None,:] - 2 * (Q @ R.T)
            AB = Q @ R.t()  # (bq, br)
            D2 = Q_norm.unsqueeze(1) + R_norm.unsqueeze(0) - 2.0 * AB
            D2 = torch.clamp(D2, min=0.0)

            exponents = -D2 * inv_two_h2  # (bq, br)
            # per-row log-sum-exp for this block:
            # block_logsum = log(sum_j exp(exponents_row_j))
            block_logsum = torch.logsumexp(exponents, dim=1)  # (bq,)

            # combine acc_logsum and block_logsum via elementwise logaddexp
            acc_logsum = torch.logaddexp(acc_logsum, block_logsum)

        # now acc_logsum is log(sum_j exp(exponent_j)) across all refs for each query
        # log(1/n * sum exp(...)) = acc_logsum - log(n_ref)
        log_density = acc_logsum - math.log(n_ref) + const_term  # (bq,)

        # if leave-one-out requested and X_ref is actually X_query (same underlying dataset),
        # we need to correct per-query to subtract the self-contribution (kernel at 0)
        if leave_one_out:
            # Only meaningful if lengths equal and the query block aligns with the ref indices
            # We'll attempt to correct wherever it's obvious that query index i corresponds to ref index i.
            # This assumes X_ref and X_query are the same ordering and length.
            # For generality you'd pass index maps; here we handle the common case n_ref == n_query and ordering aligned.
            if n_ref == X_query.shape[0] and n_ref == X_ref.shape[0]:
                # self kernel exponent at zero is 0 -> exp(0) = 1
                # So block: acc_sum = sum_j exp(exponent_j) ; to remove self: acc_sum' = acc_sum - 1
                # convert acc_logsum to acc_sum safely:
                acc_sum = torch.exp(acc_logsum)
                # subtract 1 (self-contrib)
                # However numeric underflow can make acc_sum < 1 for tiny bandwidths; guard:
                acc_sum = torch.clamp(acc_sum - 1.0, min=eps)
                # recompute log-density
                log_density = torch.log(acc_sum) - math.log(n_ref - 1) + const_term
            else:
                # if we cannot align indices, leave as-is but warn
                logging.warning("leave_one_out requested but alignment assumptions fail; skipping leave-one-out correction for this block.")
        results.append(log_density)

    return torch.cat(results, dim=0)

# class KLDivergenceMetric:
#     def __init__(self, batch_size: int, n_components: int, bandwidth: float = 1.0):
#         """
#         KL Divergence on optionally pca-reduced data with GPU-accelerated KDE. And batched for memory efficiency.
#         Args:
#             batch_size (int): Batch size for processing data in chunks.
#             n_components (int): Number of PCA components to reduce to (if PCA is used).
        
#         """
#         self.batch_size = batch_size
#         self.n_components = n_components
#         self.bandwidth = bandwidth

#     def __call__(self, X: torch.Tensor, Y: torch.Tensor) -> float:
#         """
#         Compute the KL divergence between two datasets X and Y.
#         if multivariate and n_components is smaller than data dimension, PCA is applied first.

#         Args:
#             X_tensor (torch.Tensor): First dataset. (n_samples_x, n_features)
#             Y_tensor (torch.Tensor): Second dataset. (n_samples_y, n_features)
#         Returns:
#             float: Estimated KL divergence KL(P_Y || P_X)
#         """

#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#         # PCA reduction if needed
#         if self.n_components is not None and self.n_components < X.shape[1] and X.shape[1] > 1:
#             logging.info(f"Applying PCA reduction to {self.n_components} components for KL divergence.")
#             X, Y, components, mean = pca_gpu(X, Y, n_components=self.n_components)
        
#         # compute log p_train(x) for test points
#         logp_X = kde_logdensity_gpu(
#             X_ref=X,
#             X_query=Y,
#             bandwidth=self.bandwidth,
#             device=device,
#             batch_ref=self.batch_size,
#             batch_query=self.batch_size,
#         )  # (n_test,)

#         # compute log p_test(x) by fitting KDE on test data (same bandwidth)
#         logp_Y = kde_logdensity_gpu(
#             X_ref=Y,
#             X_query=Y,
#             bandwidth=self.bandwidth,
#             device=device,
#             batch_ref=self.batch_size,
#             batch_query=self.batch_size,
#         )

#         # KL estimate: mean_test [ log p_test - log p_train ]
#         diff = (logp_Y - logp_X).double()  # convert to double for mean + var stability
#         kl_est = diff.mean().item()
#         # approximate SE of mean (use sample variance / sqrt(n))
#         se = diff.std(unbiased=True).item() / math.sqrt(diff.numel())

#         # return kl, se, and optionally the per-point diffs for diagnostics
#         return float(kl_est), float(se), (logp_Y.cpu(), logp_X.cpu())



# def kde_logdensity_gpu(
#     X_ref: torch.Tensor,
#     X_query: torch.Tensor,
#     bandwidth: float,
#     device: Optional[torch.device] = None,
#     batch_ref: int = 65536,
#     batch_query: int = 8192,
#     eps: float = 1e-12,
# ) -> torch.Tensor:
#     """
#     Compute log density estimates log p_hat(x_query) for a Gaussian isotropic KDE
#     with bandwidth `bandwidth` using reference points X_ref.
#     All tensors should be float32. Operates on GPU if device is CUDA.

#     log p_hat(x) = log( 1/n * sum_i exp( -||x - x_i||^2 / (2 h^2) ) ) - (d/2) * log(2π) - d * log(h)

#     Args:
#         X_ref: (n_ref, d) tensor of reference points (to fit KDE)
#         X_query: (n_query, d) tensor of query points (to evaluate log density
#         bandwidth: bandwidth h for the KDE
#         device: torch.device to use (if None, will use X_ref's device)
#         batch_ref: batch size for reference points (to limit memory usage)
#         batch_query: batch size for query points (to limit memory usage)
#         eps: small constant for numerical stability in log

#     Returns: tensor of shape (n_query,), device-matched.
#     """
#     if device is None:
#         device = X_ref.device

#     n_ref, d = X_ref.shape
#     n_query = X_query.shape[0]

#     # precompute constants that are reused across batches
#     # constant term describes the log of the normalization term of the Gaussian kernel 
#     const_term = -0.5 * d * math.log(2 * math.pi) - d * math.log(bandwidth)
#     # Constant inside the kernel exponent
#     inv_two_h2 = 1.0 / (2.0 * (bandwidth ** 2))



#     logs = []
#     # process query in batches
#     for qstart in range(0, n_query, batch_query):
#         qend = min(n_query, qstart + batch_query)

#         # precompute Q norms
#         Q = X_query[qstart:qend]  # (bq, d)
#         Q_norm = (Q * Q).sum(dim=1, keepdim=True)  # (bq, 1)


#         # accumulator for log-sum-exp over references
#         # We'll gather partial log-sum-exp values using the log-sum-exp trick across ref-blocks.
#         # For stable accumulation we keep a running max and sumexp.
#         max_per_query = None
#         sumexp_per_query = None

#         for rstart in range(0, n_ref, batch_ref):
#             rend = min(n_ref, rstart + batch_ref)

#             R = X_ref[rstart:rend]  # (br, d)
#             R_norm = (R * R).sum(dim=1).unsqueeze(0)  # (1, br)

#             # compute squared distances (bq, br) via matmul trick
#             # D2 = Q_norm + R_norm - 2 * Q @ R.T
#             AB = Q @ R.t()  # (bq, br)
#             D2 = Q_norm + R_norm - 2.0 * AB
#             D2 = torch.clamp(D2, min=0.0)

#             # kernel exponent: -D2 / (2 h^2)
#             exponents = -D2 * inv_two_h2  # (bq, br)

#             # For numerically stable sum across ref-blocks we keep log-sum-exp.
#             # For each query row we need to combine exponents over multiple ref-blocks.
#             # We'll do per-block max and accumulate using the standard trick.

#             # compute per-row max for this block
#             block_max, _ = exponents.max(dim=1)  # (bq,)
#             block_max = block_max.unsqueeze(1)  # (bq,1)
            
#             # compute sumexp relative to block_max
#             block_sumexp = torch.exp(exponents - block_max).sum(dim=1)  # (bq,)

#             if max_per_query is None:
#                 max_per_query = block_max.squeeze(1)  # (bq,)
#                 sumexp_per_query = block_sumexp
#             else:
#                 # combine previous (max_prev, sumexp_prev) and new (block_max, block_sumexp)
#                 # log(sum_i exp(a_i) + sum_j exp(b_j)) = max_all + ( sumexp_prev*exp(max_prev-max_all) + block_sumexp*exp(block_max-max_all) )
#                 # where max_all = max(max_prev, block_max)
#                 new_max = torch.maximum(max_per_query, block_max.squeeze(1))
#                 sumexp_per_query = (
#                     sumexp_per_query * torch.exp(max_per_query - new_max)
#                     + block_sumexp * torch.exp(block_max.squeeze(1) - new_max)
#                 )
#                 max_per_query = new_max

#         # after all ref-blocks, logsum = max_per_query + log(sumexp_per_query)
#         logsum = max_per_query + torch.log(sumexp_per_query + eps)  # (bq,)
#         # log(1/n * sum exp(...)) = logsum - log(n)
#         log_density = logsum - math.log(n_ref) + const_term  # (bq,)
#         logs.append(log_density)

#     return torch.cat(logs, dim=0)  # (n_query,)


# ---------------------------
# PCA on GPU
# ---------------------------
def pca_gpu(X: torch.Tensor, Y:torch.Tensor, n_components: int, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fit PCA on Y (n, d) (samples, features) and return (X_proj, Y_proj).
    Uses torch.pca_lowrank for speed on GPU (works for large n).
    Note: To ensure each train group X is projected in the same way, we use Y to find the main components. 

    Args:
        X: (n, d) tensor of data
        n_components: number of PCA components to return
        device: torch.device to use (if None, will use X's device)

    
    Returns:
        X_proj: (n, k) tensor of low dimensional projected data
        Y_proj: (m, k) tensor of low dimensional projected data
    """
    device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    if X.device != device:
        X = X.to(device=device, dtype=torch.float32)
    if Y.device != device:
        Y = Y.to(device=device, dtype=torch.float32)
        
    # The PCA is computed on centered data, so this is to center X first
    mean = Y.mean(dim=0, keepdim=True)
    Yc = Y - mean

    # torch.pca_lowrank returns U, S, V; V has shape (d, k)
    U, S, V = torch.pca_lowrank(Yc, q=n_components, center=False)
    assert V.shape == (Y.shape[1], n_components), f"Unexpected PCA components shape: {V.shape}"
    components = V  # (d, k) principal components
    Y_proj = Yc @ components  # (n, k)

    Xc = X - mean
    X_proj = Xc @ components

    explained_var = (S ** 2) / (Y.shape[0] - 1)
    total_var = torch.var(Yc, dim=0, unbiased=True).sum()
    explained_var_ratio = explained_var.sum() / total_var
    logging.info(f"PCA on Y: {n_components} components explain {100.0 * explained_var_ratio.item():.2f}% of variance.")

    total_var_X = torch.var(Xc, dim=0, unbiased=True).sum()
    explained_var_ratio_X = explained_var.sum() / total_var_X
    logging.info(f"PCA on X: {n_components} components explain {100.0 * explained_var_ratio_X.item():.2f}% of variance.")


    return X_proj, Y_proj




# def kde_kl_mc(X_train: np.ndarray, X_test: np.ndarray, n_components=30, bandwidth=None, bw_grid=None, sample_size=None, rng=None):
#     """
#     Estimate KL(P_test || P_train) by:
#       - PCA reduce to n_components
#       - Fit KDE on PCA(X_train) and KDE on PCA(X_test)
#       - Monte-Carlo estimate: mean_{x in X_test} [ log p_test(x) - log p_train(x) ]

#     Args:
#         X_train: (N_train, D) array of training data (to fit P_train)
#         X_test:  (N_test, D) array of test data (to fit P_test and evaluate KL)
#         n_components: number of PCA components to use
#         bandwidth: bandwidth for KDE (if None, will do CV to select)
#         bw_grid: grid of bandwidths to search over if bandwidth is None
#         sample_size: if not None, subsample this many points from X_train to fit KDE (for speed)
#         rng: optional np.random.Generator for reproducibility

#     Returns: kl_estimate, (logp_test_vals, logp_train_vals), pca_obj
#     """
#     rng = np.random.default_rng() if rng is None else np.random.default_rng(rng)
#     X_train = np.asarray(X_train)
#     X_test = np.asarray(X_test)

#     # 1) PCA (fit on train -> sensible for OOD detection)
#     time_start = time.perf_counter()
#     pca = PCA(n_components=n_components, svd_solver='randomized', whiten=False)
#     Xtrain_p = pca.fit_transform(X_train)
#     Xtest_p = pca.transform(X_test)
#     time_elapsed = time.perf_counter() - time_start
#     logging.info(f"PCA took {time_elapsed:.3f} seconds")

#     # optional subsample for speed
#     if sample_size is not None and sample_size < Xtrain_p.shape[0]:
#         time_start = time.perf_counter()
#         idx = rng.choice(Xtrain_p.shape[0], size=sample_size, replace=False)
#         Xtrain_p_sub = Xtrain_p[idx]
#         time_elapsed = time.perf_counter() - time_start
#         logging.info(f"Subsampling took {time_elapsed:.3f} seconds")
#     else:
#         Xtrain_p_sub = Xtrain_p
#         logging.info(f"No subsampling applied.")

#     # 2) bandwidth selection for KDE if not provided
#     if bandwidth is None:
#         time_start = time.perf_counter()
#         # simple grid search CV (cheap if small)
#         if bw_grid is None:
#             bw_grid = np.logspace(-1, 1, 10)
#         grid = GridSearchCV(KernelDensity(), {'bandwidth': bw_grid}, cv=3)
#         grid.fit(Xtrain_p_sub)
#         bandwidth = grid.best_params_['bandwidth']
#         time_elapsed = time.perf_counter() - time_start
#         logging.info(f"KDE bandwidth selection took {time_elapsed:.3f} seconds. Selected bandwidth: {bandwidth:.4f}")

#     # fit KDEs
#     time_start = time.perf_counter()
#     kde_train = KernelDensity(bandwidth=bandwidth).fit(Xtrain_p_sub)
#     kde_test = KernelDensity(bandwidth=bandwidth).fit(Xtest_p)  # fit test KDE on test points
#     time_elapsed = time.perf_counter() - time_start
#     logging.info(f"KDE fitting took {time_elapsed:.3f} seconds.")

#     # 3) compute log densities on test points
#     time_start = time.perf_counter()
#     logp_test = kde_test.score_samples(Xtest_p)   # log p_test(x)
#     logp_train = kde_train.score_samples(Xtest_p) # log p_train(x)
#     time_elapsed = time.perf_counter() - time_start
#     logging.info(f"KDE scoring took {time_elapsed:.3f} seconds.")

#     kl_est = np.mean(logp_test - logp_train)

#     return float(kl_est), (logp_test, logp_train), pca


# ---------------------------
# full pipeline: PCA + KDE KL on GPU
# ---------------------------
def pca_kde_kl_gpu(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_components: int = 30,
    bandwidth: float = 1.0,
    device: Optional[torch.device] = None,
    batch_ref: int = 65536,
    batch_query: int = 8192,
) -> Tuple[float, float, torch.Tensor]:
    """
    Estimate KL(test || train) via PCA on train -> reduce data, then GPU-KDE for log-densities.
    Returns (kl_estimate, kl_se_estimate_approx, details)

    Args:
        X_train: (n train samples, features) array of training data (to fit P_train)
        X_test:  (n test samples, features) array of test data (to fit P_test and evaluate KL)
        n_components: number of PCA components to use
        bandwidth: bandwidth for KDE
        device: torch.device to use (if None, will use GPU if available)
        batch_ref: batch size for reference points in KDE
        batch_query: batch size for query points in KDE
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # move data to torch tensors on device (float32)
    Xtr = torch.as_tensor(X_train, dtype=torch.float32, device=device)
    Xte = torch.as_tensor(X_test, dtype=torch.float32, device=device)

    time_start = time.perf_counter()
    # PCA fit on training (recommended for OOD detection)
    Xtr_proj, components, mean = pca_gpu(Xtr, n_components=n_components, device=device)
    time_elapsed = time.perf_counter() - time_start
    logging.info(f"PCA on GPU took {time_elapsed:.3f} seconds.")
    logging.info(f"{n_components} components explain {100.0 * torch.sum(torch.var(Xtr_proj, dim=0)) / torch.sum(torch.var(Xtr, dim=0)):.2f}% of variance.")


    # project test to same lower dimension space as the training data
    Xte_centered = Xte - mean.unsqueeze(0)
    Xte_proj = Xte_centered @ components

    # compute log p_train(x) for test points
    logp_train = kde_logdensity_gpu(
        X_ref=Xtr_proj,
        X_query=Xte_proj,
        bandwidth=bandwidth,
        device=device,
        batch_ref=batch_ref,
        batch_query=batch_query,
    )  # (n_test,)

    # compute log p_test(x) by fitting KDE on test data (same bandwidth)
    logp_test = kde_logdensity_gpu(
        X_ref=Xte_proj,
        X_query=Xte_proj,
        bandwidth=bandwidth,
        device=device,
        batch_ref=batch_ref,
        batch_query=batch_query,
    )

    # KL estimate: mean_test [ log p_test - log p_train ]
    diff = (logp_test - logp_train).double()  # convert to double for mean + var stability
    kl_est = diff.mean().item()
    # approximate SE of mean (use sample variance / sqrt(n))
    se = diff.std(unbiased=True).item() / math.sqrt(diff.numel())

    # return kl, se, and optionally the per-point diffs for diagnostics
    return float(kl_est), float(se), (logp_test.cpu(), logp_train.cpu())


# def select_bandwidth_gpu(X_train_np, n_components, bandwidths, device=None, val_frac=0.2):
#     """
#     Select the best bandwidth for KDE on the GPU using a validation set.
#     """
    
#     device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
#     X = torch.as_tensor(X_train_np, dtype=torch.float32, device=device)
#     n = X.shape[0]
#     val_n = int(n * val_frac)
#     idx = torch.randperm(n, device=device)
#     val_idx = idx[:val_n].cpu().numpy()
#     train_idx = idx[val_n:].cpu().numpy()
#     best_bw = None
#     best_score = -1e99
#     for bw in bandwidths:
#         # PCA fit on train_idx subset
#         Xtr = X[train_idx]
#         Xval = X[val_idx]
#         Xtr_proj, comps, mean = pca_gpu(Xtr, n_components=n_components, device=device)
#         Xval_centered = Xval - mean.unsqueeze(0)
#         Xval_proj = Xval_centered @ comps
#         # compute val log-likelihood under KDE fitted on train_proj
#         logp_val = kde_logdensity_gpu(X_ref=Xtr_proj, X_query=Xval_proj, bandwidth=bw, device=device)
#         avg_ll = logp_val.mean().item()
#         if avg_ll > best_score:
#             best_score = avg_ll
#             best_bw = bw
#     return best_bw, best_score
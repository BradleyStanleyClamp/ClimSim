import torch 
import evaluate
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def test_pca_gpu_simple_run():

    # Create random data
    X = torch.randn(100, 20)
    Y = torch.randn(50, 20)

    n_components = 10
    X_proj, Y_proj = evaluate.pca_gpu(X, Y, n_components)

    assert X_proj.shape == (100, n_components)
    assert Y_proj.shape == (50, n_components)

def test_pca_gpu_same_var():
    X = torch.randn(100, 20)
    Y = X.clone()

    n_components = 3
    X_proj, Y_proj = evaluate.pca_gpu(X, Y, n_components)

    assert X_proj.shape == (100, n_components)
    assert Y_proj.shape == (100, n_components)
    assert torch.allclose(X_proj, Y_proj, atol=1e-6)



def align_columns_by_sign_np(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Flip columns of B so each column aligns (positive dot) with the corresponding column in A.
    A, B: (n_samples, n_components)
    """
    dots = (A * B).sum(axis=0)          # length k
    signs = np.where(dots < 0, -1.0, 1.0)
    return B * signs[None, :]

def test_pca_compared_to_sklearn():
    import numpy as np
    from sklearn.decomposition import PCA

    # Use reproducible random data
    np.random.seed(0)
    torch.manual_seed(0)

    X = torch.randn(5, 15)
    Y = torch.randn(5, 15)

     # sizes and dims
    n_x = 60
    n_y = 70
    d = 20
    k_true = 3         # true latent dimensionality
    n_components = 3   # components to extract (must equal k_true here)

    # create a random loading matrix A (d x k_true) with full column rank
    A = np.random.randn(d, k_true)   # fixed loadings define the 3D subspace

    # create latent samples for X and Y (different samples but same subspace)
    Z_x = np.random.randn(n_x, k_true)
    Z_y = np.random.randn(n_y, k_true)

    # generate high-dim observations (noisy)
    noise_scale = 1e-6  # tiny isotropic noise to avoid exact degeneracy but keep signal dominant
    X_np = Z_x @ A.T + noise_scale * np.random.randn(n_x, d)
    Y_np = Z_y @ A.T + noise_scale * np.random.randn(n_y, d)

    # convert to torch
    X = torch.from_numpy(X_np.astype(np.float32))
    Y = torch.from_numpy(Y_np.astype(np.float32))


    X_proj_torch, Y_proj_torch = evaluate.pca_gpu(X, Y, n_components)

    # Fit sklearn PCA on Y to match the semantics of pca_gpu (PCA fit on Y)
    pca = PCA(n_components=n_components)
    Y_np = Y.numpy()
    pca.fit(Y_np)

    X_proj_sklearn = pca.transform(X.numpy())   # note order: transform X using PCA fit on Y
    Y_proj_sklearn = pca.transform(Y_np)

    # Align signs (scikit-learn vs torch may flip signs per component)
    Y_proj_torch_np = Y_proj_torch.numpy()
    X_proj_torch_np = X_proj_torch.numpy()

    Y_proj_sklearn_aligned = align_columns_by_sign_np(Y_proj_torch_np, Y_proj_sklearn)
    X_proj_sklearn_aligned = align_columns_by_sign_np(X_proj_torch_np, X_proj_sklearn)

    assert np.allclose(Y_proj_torch_np, Y_proj_sklearn_aligned, atol=1e-5)
    assert np.allclose(X_proj_torch_np, X_proj_sklearn_aligned, atol=1e-5)

def test_kl_divergence_function():
    np.random.seed(0)
    dist1 = np.random.multivariate_normal(np.array([1, 1]), np.identity(2), 10000)
    dist2 = np.random.multivariate_normal(np.array([1, 1]), np.identity(2), 10000)

    output = evaluate.KLdivergence(dist1,dist2)
    assert isinstance(output, float)
    # assert output >= 0.0 # Note this function is an approximation and may yield slight negative values
    assert np.isclose(output, 0.0, atol=0.1)  # Should be close to 0 for identical distributions

def test_kl_divergence_different_distributions():
    np.random.seed(0)
    ref = np.random.multivariate_normal(np.array([0, 0]), np.identity(2), 10000)
    dist2 = np.random.multivariate_normal(np.array([5, 5]), np.identity(2), 10000)
    dist3 = np.random.multivariate_normal(np.array([10, 10]), np.identity(2), 10000)


    output1 = evaluate.KLdivergence(ref,dist2)
    output2 = evaluate.KLdivergence(ref,dist3)
    assert output1 < output2  # KL divergence should be larger for more different distributions
  
def test_kl_divergence_methods_discrete_equivalent():
    torch.manual_seed(0)
    X = torch.randn(3, 5)
    Y = torch.randn(3, 5)

    kl_loss = nn.KLDivLoss(reduction='batchmean')  # or 'mean' depending on PyTorch version; see below
    input_log = F.log_softmax(X, dim=1)            # log P (model)
    target_prob = F.softmax(Y, dim=1)              # Q (true)

    torch_output = kl_loss(input_log, target_prob)

    # NumPy equivalent: compute KL(Q || P) = mean_over_batch sum_c Q(c) * (log Q(c) - log P(c))
    eps = 1e-12
    input_log_np = input_log.detach().numpy()        # shape (N, C)
    target_np = target_prob.detach().numpy()

    # Compute per-sample KL, then average (match reduction='batchmean' behavior)
    per_sample_kl = np.sum(target_np * (np.log(target_np + eps) - input_log_np), axis=1)
    np_output = per_sample_kl.mean()

    print("torch_output:", torch_output.item())
    print("np_output:", np_output)
    assert np.isclose(torch_output.item(), np_output, atol=1e-7)

from scipy.linalg import det, inv

def analytic_kl_gaussian(mu_p, Sigma_p, mu_q, Sigma_q):
    d = mu_p.shape[0]
    term1 = np.log(np.linalg.det(Sigma_q) / np.linalg.det(Sigma_p))
    term2 = np.trace(np.linalg.solve(Sigma_q, Sigma_p))
    diff = mu_q - mu_p
    term3 = diff.T @ np.linalg.solve(Sigma_q, diff)
    return 0.5 * (term1 - d + term2 + term3)

def test_knn_estimator_matches_analytic_gaussian():
    np.random.seed(0)
    d = 5
    n = 5000
    mu_p = np.zeros(d)
    Sigma_p = np.eye(d) * 1.0
    mu_q = np.ones(d) * 0.1
    Sigma_q = np.eye(d) * 1.2

    X = np.random.multivariate_normal(mu_p, Sigma_p, size=n)
    Y = np.random.multivariate_normal(mu_q, Sigma_q, size=n)

    est = evaluate.KLdivergence(X, Y)
    analytic = analytic_kl_gaussian(mu_p, Sigma_p, mu_q, Sigma_q)

    print("est:", est, "analytic:", analytic)
    assert np.isclose(est, analytic, atol=0.2)  # tolerance depends on n and estimator variance

def test_kl_divergence_with_pca():
    np.random.seed(0)
    d = 10
    n = 3000
    mu_p = np.zeros(d)
    Sigma_p = np.eye(d) * 1.0
    mu_q = np.ones(d) * 0.5
    Sigma_q = np.eye(d) * 1.5

    X = np.random.multivariate_normal(mu_p, Sigma_p, size=n)
    Y = np.random.multivariate_normal(mu_q, Sigma_q, size=n)

    n_components = 3
    X_torch = torch.from_numpy(X.astype(np.float32))
    Y_torch = torch.from_numpy(Y.astype(np.float32))

    X_pca, Y_pca = evaluate.pca_gpu(X_torch, Y_torch, n_components)

    est_pca = evaluate.KLdivergence(X_pca.numpy(), Y_pca.numpy())
    analytic_pca = analytic_kl_gaussian(mu_p[:n_components], Sigma_p[:n_components, :n_components],
                                        mu_q[:n_components], Sigma_q[:n_components, :n_components])

    print("est_pca:", est_pca, "analytic_pca:", analytic_pca)
    assert np.isclose(est_pca, analytic_pca, atol=0.3)  # looser tolerance due to PCA approximation
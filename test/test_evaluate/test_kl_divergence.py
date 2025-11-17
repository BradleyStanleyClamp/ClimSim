import torch 
import evaluate

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

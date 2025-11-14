import torch

def squared_euclidean_distance_matrix_matmul_trick(X: torch.Tensor, Y: torch.Tensor):
    """
    Computes the squared Euclidean distance matrix between two sets of vectors using the matmul trick for efficiency.

    Args:
        X (torch.Tensor): First input tensor of shape (n, d).
        Y (torch.Tensor): Second input tensor of shape (m, d).

    Returns:
        torch.Tensor: Squared Euclidean distance matrix of shape (n, m).
    """
    X_norm = (X**2).sum(dim=1).view(-1, 1)  # (n, 1)
    Y_norm = (Y**2).sum(dim=1).view(1, -1)  # (1, m)
    dist_matrix = X_norm + Y_norm - 2.0 * torch.matmul(X, Y.t())  # (n, m)
    dist_matrix = torch.clamp(dist_matrix, min=0.0)  # Ensure non-negative distances
    return dist_matrix
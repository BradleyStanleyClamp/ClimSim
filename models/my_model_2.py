"""
Model built up as part of P2.1.4.10
"""

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from .squeezeformer import (
    Conv1DBlock,
    ResidualGluMlp,
    HeadDense,
    GluMlp,
    SelfAttentionBlock,
)


class MyModel2(nn.Module):
    def __init__(
        self,
        input_dim: int,
        emb_dim: int,
        output_dim: int,
        head_dim: int,
        beta: float,
        invariant_levels: int,
    ):
        """
        Initializes the MyModel1 model.
        Args:
            input_dim (int): Dimension of the input features.
            emb_dim (int): Dimension of the embedding layer.
            output_dim (int): Dimension of the output.
            activation (callable): Activation function to be used. Default is ReLU.

        """
        super().__init__()
        self.name = "my_model_1"
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.output_dim = output_dim
        self.head_dim = head_dim
        self.beta = beta
        self.invariant_levels = invariant_levels

        self.embedding = nn.ModuleList(
            [nn.Linear(input_dim, emb_dim), nn.LayerNorm(emb_dim)]
        )

        self.encoder = nn.Sequential(
            Conv1DBlock(emb_dim, emb_dim),
            ResidualGluMlp(emb_dim, emb_dim * 4),
            SelfAttentionBlock(emb_dim, num_heads=4),
            ResidualGluMlp(emb_dim, emb_dim * 4),
        )

        self.decoder = nn.ModuleList(
            [
                HeadDense(emb_dim, head_dim),
                GluMlp(head_dim, expanded_dim=head_dim * 2),
            ]
        )

        self.prediction_head = nn.Linear(head_dim, output_dim)

    def forward(self, x):
        """
        Forward pass through the MLP.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, levels, in_dim).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, variables).
        """
        # Embedding
        for layer in self.embedding:
            x = layer(x)

        # Encoder
        for block in self.encoder:
            x = block(x)

        z = x.clone()  # latent representation (batch, levels, emb_dim)

        # Decoder
        for layer in self.decoder:
            x = layer(x)

        # Prediction head
        x = self.prediction_head(x)

        x = self._reshape_to_standard_format(x)
        return x, z

    def _reshape_to_standard_format(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshapes the output to standard format (batch, variables) from (batch, levels, features)
        Args:
            x: (torch.Tensor) (batch, levels, features) output data from the model
        Returns:
            reshaped_x: (torch.Tensor) (batch, features) reshaped output data
        """

        # flatten the first two levels into (n, 120) in one op
        first_two = (
            x[:, :, 0:2].transpose(1, 2).reshape(x.shape[0], -1)
        )  # shape (n, 2*60) == (n,120)
        # first_two = first_two[
        #     :, :-15
        # ]  # HARDCODED REMOVAL OF TOP SH Levels # shape (n, 105) which is set to 0 in the input

        # compute the per-level means for levels 2..9 in one op
        means_2_to_9 = x[:, :, 2:10].mean(dim=1)  # shape (n, 8)

        # concatenate once
        output = torch.cat([first_two, means_2_to_9], dim=1)  # shape (n, 128)
        return output

    def step(self, output, y, log, loss_metric, stage=None):
        """
        Generic step for training, validation, and testing.
        Args:
            batch (tuple): A tuple containing input data and target labels.
            batch_idx (int): Index of the batch.
            stage (str, optional): Stage of the step ('train', 'val', 'test'). Default is None.

        """
        output_nh, output_sh = output
        y_hat_nh, z_nh = output_nh
        y_hat_sh, z_sh = output_sh

        y_nh, y_sh = y

        z_nh_inv = z_nh[:, : self.invariant_levels, :]  # shape [samples, levels, emb_dim]
        z_sh_inv = z_sh[:, : self.invariant_levels, :]  # shape [samples, levels, emb_dim]

        ed_inv = energy_distance_per_level(z_nh_inv, z_sh_inv)  # shape [levels]
        emd_inv = sinkhorn_per_level(z_nh_inv, z_sh_inv)  # shape [levels]

        log(f"{stage}/energy_distance_invariant_levels", ed_inv.mean())
        log(
            f"{stage}/sinkhorn_distance_invariant_levels",
            emd_inv.mean(),
        )

        z_nh_noninv = z_nh[:, self.invariant_levels :, :].detach()
        z_sh_noninv = z_sh[:, self.invariant_levels :, :].detach()
        ed_non_inv = energy_distance_per_level(z_nh_noninv, z_sh_noninv)  # shape [levels]
        emd_non_inv = sinkhorn_per_level(z_nh_noninv, z_sh_noninv)
        log(
            f"{stage}/energy_distance_variant_levels",
            ed_non_inv.mean(),
        )
        log(
            f"{stage}/sinkhorn_distance_variant_levels",
            emd_non_inv.mean(),
        )

        ed_full = energy_distance_per_level(z_nh.detach(), z_sh.detach())  # shape [levels]
        emd_full = sinkhorn_per_level(z_nh.detach(), z_sh.detach())
        log(f"{stage}/mean_energy_distance", ed_full.mean())
        log(f"{stage}/mean_sinkhorn_distance", emd_full.mean())

        y_hat = torch.cat([y_hat_nh, y_hat_sh], dim=0)
        y = torch.cat([y_nh, y_sh], dim=0)
        standard_loss = loss_metric(y_hat, y)
        log(f"{stage}/loss", standard_loss)

        nh_loss = loss_metric(y_hat_nh, y_nh)
        sh_loss = loss_metric(y_hat_sh, y_sh)
        log(f"{stage}/nh_loss", nh_loss)
        log(f"{stage}/sh_loss", sh_loss)

        loss = standard_loss + self.beta * ed_inv.mean()
        log(f"{stage}/total_loss", loss)
        return loss


def energy_distance_per_level(
    z_nh: torch.Tensor, z_sh: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute energy distance per level.

    Args:
      z_nh, z_sh: tensors of shape [samples, levels, emb_dim]
    Returns:
      ed: tensor of shape [levels] with the energy distance for each level
    """
    # move levels -> batch dim
    # new shapes: [levels, samples, emb_dim]
    z1 = z_nh.permute(2, 0, 1).contiguous()
    z2 = z_sh.permute(2, 0, 1).contiguous()

    # batched pairwise distances:
    # d_xy: [levels, n, m]  (here n==m==samples typically)
    d_xy = torch.cdist(z1, z2, p=2)  # batch-aware
    d_xx = torch.cdist(z1, z1, p=2)
    d_yy = torch.cdist(z2, z2, p=2)

    # mean over the sample dims for each level
    exy = d_xy.mean(dim=(-2, -1))  # shape [levels]
    exx = d_xx.mean(dim=(-2, -1))
    eyy = d_yy.mean(dim=(-2, -1))

    ed2 = 2.0 * exy - exx - eyy
    ed2 = torch.clamp(ed2, min=0.0)
    return torch.sqrt(ed2 + eps)  # shape [levels]


def sinkhorn_per_level(
    z_nh: torch.Tensor,
    z_sh: torch.Tensor,
    reg: float = 0.1,
    n_iters: int = 100,
    eps: float = 1e-9,
    use_squared_cost: bool = True,
) -> torch.Tensor:
    """
    Batched Sinkhorn (entropic-regularized OT) per level.
    Args:
      z_nh, z_sh: [samples, levels, emb_dim]
      reg: entropic regularization (epsilon in many papers)
      n_iters: number of Sinkhorn iterations
      eps: small numerical epsilon
      use_squared_cost: if True use squared Euclidean cost; else L2 distances
    Returns:
      wass: [levels] approximate 1-Wasserstein cost (transport cost)
    """
    # move levels to batch dim: [L, n, d]
    z1 = z_nh.permute(1, 0, 2).contiguous()
    z2 = z_sh.permute(1, 0, 2).contiguous()
    L, n, d = z1.shape
    _, m, _ = z2.shape
    assert (
        n == m
    ), "samples per level must match for this implementation; can be relaxed"

    # uniform marginals
    a = torch.full((L, n), 1.0 / n, device=z1.device, dtype=z1.dtype)  # [L, n]
    b = torch.full((L, m), 1.0 / m, device=z1.device, dtype=z1.dtype)  # [L, m]

    # cost matrix: squared euclidean distances
    # shape -> [L, n, m]
    if use_squared_cost:
        # cdist squared: c(i,j) = ||x_i - y_j||^2
        x_norm = (z1**2).sum(dim=-1, keepdim=True)  # [L, n, 1]
        y_norm = (z2**2).sum(dim=-1, keepdim=True)  # [L, m, 1]
        # compute pairwise: ||x-y||^2 = ||x||^2 + ||y||^2 - 2<x,y>
        C = x_norm + y_norm.permute(0, 2, 1) - 2.0 * (z1 @ z2.permute(0, 2, 1))
        C = torch.clamp(C, min=0.0)
    else:
        C = torch.cdist(z1, z2, p=2)  # [L, n, m]

    # Kernel
    K = torch.exp(-C / reg)  # [L, n, m]
    K = torch.clamp(K, min=1e-100)  # avoid exact zeros

    # Sinkhorn iterations with scaling vectors u, v (batched)
    u = torch.ones((L, n), device=z1.device, dtype=z1.dtype) / n
    v = torch.ones((L, m), device=z1.device, dtype=z1.dtype) / m

    for _ in range(n_iters):
        Kt_v = torch.matmul(K.transpose(1, 2), v.unsqueeze(-1)).squeeze(-1)  # [L, n]
        u = a / (Kt_v + eps)

        K_u = torch.matmul(K, u.unsqueeze(-1)).squeeze(-1)  # [L, m]
        v = b / (K_u + eps)

    # transport matrix T = diag(u) @ K @ diag(v)
    # compute cost: sum_{i,j} T_ij * C_ij
    # we can compute elementwise product and sum: (u[:,:,None] * K * v[:,None,:]) * C
    u_col = u.unsqueeze(-1)  # [L, n, 1]
    v_row = v.unsqueeze(1)  # [L, 1, m]
    T = u_col * K * v_row  # [L, n, m]
    cost = (T * C).sum(dim=(1, 2))  # [L]

    # If C was squared distances and you want 1-Wasserstein (not squared),
    # you could take sqrt(cost). However standard entropic OT returns expected cost
    # with the chosen cost function. For cost=||x-y||, set use_squared_cost=False.
    return cost  # [levels]

"""
Model built up as part of P2.1.4.11
"""

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


class MyModel3(nn.Module):
    def __init__(
        self,
        input_dim: int,
        emb_dim: int,
        output_dim: int,
        z_dim: int,
        beta: float,
    ):
        """
        Initializes the MyModel3 model.
        Args:
            input_dim (int): Dimension of the input features.
            emb_dim (int): Dimension of the embedding layer.
            output_dim (int): Dimension of the output.

        """
        super().__init__()
        self.name = "my_model_3"
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.output_dim = output_dim
        self.beta = beta

        self.encoder = nn.ModuleList(
            [
                nn.Conv1d(input_dim, 64, kernel_size=1),
                nn.Conv1d(64, 64, kernel_size=1),
                nn.Conv1d(64, 64, kernel_size=1),
                nn.Conv1d(64, 64, kernel_size=1),
                nn.Conv1d(64, 3, kernel_size=1),
            ]
        )

        self.decoder = nn.ModuleList(
            [
                nn.Conv1d(3, 64, kernel_size=1),
                nn.Conv1d(64, 64, kernel_size=1),
                nn.Conv1d(64, output_dim, kernel_size=1),
            ]
        )

    def forward(self, x):
        """
        Forward pass through the MLP.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, variables, levels+padding).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        # print(f"Input shape: {x.shape}")
        # Encode
        for layer in self.encoder:
            x = layer(x)
            # print(f"    {x.shape}")

        z = x.clone()  # Latent representation

        # print(f"Latent shape: {z.shape}")
        # Decode
        for layer in self.decoder:
            x = layer(x)
            # print(f"    {x.shape}")

        y = self._reshape_to_standard_format(x)
        # print(f"Output shape: {y.shape}")
        return y, z

    def step(self, output, y, log, loss_metric, stage=None):
        """
        Computes the loss for a given output and target.
        Args:
            output (torch.Tensor): Model output.
            y (torch.Tensor): Target tensor.
            log (dict): Dictionary to log metrics.
            loss_metric (callable): Loss function to compute the loss.
            stage (str, optional): Stage of training (e.g., 'train', 'val', 'test').
        Returns:
            torch.Tensor: Computed loss.
        """
        output_nh, output_sh = output  # unpack output tuple
        y_hat_nh, z_nh = output_nh
        y_hat_sh, z_sh = output_sh

        y_nh, y_sh = y

        y_hat = torch.cat([y_hat_nh, y_hat_sh], dim=0)
        y_true = torch.cat([y_nh, y_sh], dim=0)
        standard_loss = loss_metric(y_hat, y_true)
        log(f"{stage}/loss", standard_loss.item())

        z_nh_invariant = z_nh[:, :2]
        z_sh_invariant = z_sh[:, :2]
        ed_invariant = energy_distance_per_variable(z_nh_invariant, z_sh_invariant)
        ed_invariant_loss = ed_invariant.mean()
        log(f"{stage}/ed_invariant_loss", ed_invariant_loss.item())

        z_nh_variant = z_nh[:, 2:].detach()
        z_sh_variant = z_sh[:, 2:].detach()
        ed_variant = energy_distance_per_variable(z_nh_variant, z_sh_variant)
        ed_variant_loss = ed_variant.mean()
        log(f"{stage}/ed_variant_loss", ed_variant_loss.item())

        z_nh_detached = z_nh.detach()
        z_sh_detached = z_sh.detach()
        ed_detached = energy_distance_per_variable(z_nh_detached, z_sh_detached)
        ed_detached_loss = ed_detached.mean()
        log(f"{stage}/ed_loss", ed_detached_loss.item())

        loss = standard_loss + self.beta * (ed_invariant_loss)
        return loss

    def _reshape_to_standard_format(self, x):
        """
        Reshapes the output to standard format (n_samples, features) from (n_samples, levels, variables)
        Args:
            x: (torch.Tensor) (n_samples, levels, variables) output data from the model
        Returns:
            reshaped_x: (torch.Tensor) (n_samples, features) reshaped output data
        """
        num_levels = 45

        xlevels = x[:, :, 0:num_levels]

        # flatten the first two levels into (n, 120) in one op
        first_two = xlevels[:, 0:2, :].reshape(
            x.shape[0], -1
        )  # shape (n, 2*60) == (n,120)
        # first_two = first_two[
        #     :, :-15
        # ]  # HARDCODED REMOVAL OF TOP SH Levels # shape (n, 105)
        # compute the per-level means for levels 2..9 in one op
        means_2_to_9 = xlevels[:, 2:10, :].mean(dim=2)  # shape (n, 8)

        # concatenate once
        output = torch.cat([first_two, means_2_to_9], dim=1)  # shape (n, features)
        return output


def energy_distance_per_variable(
    z_nh: torch.Tensor, z_sh: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute energy distance per variable.

    Args:
      z_nh, z_sh: tensors of shape [samples, z_dim(variables), emb_dim]
    Returns:
      ed: tensor of shape [z_dim(variables)] with the energy distance for each variable
    """
    # move variables -> batch dim
    # new shapes: [variables, samples, emb_dim]
    z1 = z_nh.transpose(0, 2)
    z2 = z_sh.transpose(0, 2)

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

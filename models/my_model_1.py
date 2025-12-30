"""
Model built up as part of P2.1.4.7
"""

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


class MyModel1(nn.Module):
    def __init__(
        self,
        input_dim: int,
        emb_dim: int,
        output_dim: int,
        z_dim: int,
        beta: float,
        activation=F.relu,
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
        self.activation = activation
        self.beta = beta

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(inplace=True),
        )

        self.fc_mu = nn.Linear(emb_dim, z_dim)
        self.fc_logvar = nn.Linear(emb_dim, z_dim)

        self.decoder = nn.Linear(z_dim, output_dim)

    def forward(self, x):
        """
        Forward pass through the MLP.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        mu, logvar = self.encode(x)
        x = self.reparameterize(mu, logvar)
        x = self.decoder(x)
        return x, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(logvar)
        return mu + eps * std

    def step(self, output, y, log, loss_metric, stage=None):
        """
        Generic step for training, validation, and testing.
        Args:
            batch (tuple): A tuple containing input data and target labels.
            batch_idx (int): Index of the batch.
            stage (str, optional): Stage of the step ('train', 'val', 'test'). Default is None.

        """
        output_nh, output_sh = output
        y_nh, y_sh = y

        y_hat_nh, mu_nh, logvar_nh = output_nh
        y_hat_sh, mu_sh, logvar_sh = output_sh

        y_hat = torch.cat([y_hat_nh, y_hat_sh], dim=0)
        y = torch.cat([y_nh, y_sh], dim=0)
        standard_loss = loss_metric(y_hat, y)
        log(f"{stage}/loss", standard_loss)

        log(f"{stage}/mu_nh_mean", mu_nh.mean())
        log(f"{stage}/logvar_nh_mean", logvar_nh.mean())
        log(f"{stage}/mu_sh_mean", mu_sh.mean())
        log(f"{stage}/logvar_sh_mean", logvar_sh.mean())

        kl_loss = self.kl_diag_gaussians_safe(mu_nh, logvar_nh, mu_sh, logvar_sh).mean()

        loss = standard_loss + self.beta * kl_loss
        log(f"{stage}/kl_loss", kl_loss)
        log(f"{stage}/total_loss", loss)

        return loss

    def kl_diag_gaussians_safe(
        self, mu_q, lv_q, mu_p, lv_p, clamp_min=-30.0, clamp_max=20.0, eps=1e-8
    ):
        """
        Analytical KL per-sample between two diagonal Gaussians:
        q = N(mu_q, exp(lv_q))
        p = N(mu_p, exp(lv_p))

        Returns: kl_per_sample (shape (B,))
        """
        # ensure shapes (B, D)
        mu_q = mu_q.view(mu_q.size(0), -1)
        mu_p = mu_p.view(mu_p.size(0), -1)
        lv_q = lv_q.view(lv_q.size(0), -1)
        lv_p = lv_p.view(lv_p.size(0), -1)

        # numeric safety
        lv_q = torch.clamp(lv_q, min=clamp_min, max=clamp_max)
        lv_p = torch.clamp(lv_p, min=clamp_min, max=clamp_max)

        var_q = torch.exp(lv_q)
        var_p = torch.exp(lv_p)

        # terms of the standard formula
        term_logratio = lv_p - lv_q  # log(var_p / var_q)
        term_varratio = var_q / (var_p + eps)  # var_q / var_p
        term_mahal = (mu_q - mu_p).pow(2) / (var_p + eps)  # (mu_q-mu_p)^2 / var_p

        kl_elem = 0.5 * (term_logratio + term_varratio + term_mahal - 1.0)
        kl_per_sample = kl_elem.sum(dim=1)  # sum over latent dims
        return kl_per_sample

    # def step(self, output, y, log, loss_metric, stage=None):
    #     """
    #     Generic step for training, validation, and testing.
    #     Args:
    #         output (torch.Tensor): Model output consiting of (y_hat, mu, logvar) with shapes (B, output_dim), (B, z_dim), (B, z_dim).
    #         y (torch.Tensor): Ground truth labels of shape (B, output_dim).
    #         log (callable): Logging function to log metrics.
    #         loss_metric (callable): Loss function to compute the standard loss.
    #         batch (tuple): A tuple containing input data and target labels.
    #         batch_idx (int): Index of the batch.
    #         stage (str, optional): Stage of the step ('train', 'val', 'test'). Default is None.

    #     """
    #     y_hat, mu, logvar = output
    #     standard_loss = loss_metric(y_hat, y)
    #     log(f"{stage}/loss", standard_loss)

    #     # mu, std shaped (B, C, L) for example
    #     kl_per_elem = mu.pow(2) + torch.exp(logvar) - 1 - logvar
    #     kl_per_sample = 0.5 * torch.sum(kl_per_elem, dim=[1])
    #     kl_loss = kl_per_sample.mean()
    #     loss = standard_loss + self.beta * kl_loss
    #     log(f"{stage}/kl_loss", kl_loss)
    #     log(f"{stage}/total_loss", loss)

    #     return loss

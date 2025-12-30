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
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        # Embedding
        for layer in self.embedding:
            x = layer(x)

        # Encoder
        for block in self.encoder:
            x = block(x)

        # Decoder
        for layer in self.decoder:
            x = layer(x)

        # Prediction head
        x = self.prediction_head(x)

        x = self._reshape_to_standard_format(x)
        return x

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

    # def step(self, output, y, log, loss_metric, stage=None):
    #     """
    #     Generic step for training, validation, and testing.
    #     Args:
    #         batch (tuple): A tuple containing input data and target labels.
    #         batch_idx (int): Index of the batch.
    #         stage (str, optional): Stage of the step ('train', 'val', 'test'). Default is None.

    #     """
    #     output_nh, output_sh = output
    #     y_nh, y_sh = y

    #     y_hat_nh, mu_nh, logvar_nh = output_nh
    #     y_hat_sh, mu_sh, logvar_sh = output_sh

    #     y_hat = torch.cat([y_hat_nh, y_hat_sh], dim=0)
    #     y = torch.cat([y_nh, y_sh], dim=0)
    #     standard_loss = loss_metric(y_hat, y)
    #     log(f"{stage}/loss", standard_loss)

    #     log(f"{stage}/mu_nh_mean", mu_nh.mean())
    #     log(f"{stage}/logvar_nh_mean", logvar_nh.mean())
    #     log(f"{stage}/mu_sh_mean", mu_sh.mean())
    #     log(f"{stage}/logvar_sh_mean", logvar_sh.mean())

    #     kl_loss = self.kl_diag_gaussians_safe(mu_nh, logvar_nh, mu_sh, logvar_sh).mean()

    #     loss = standard_loss + self.beta * kl_loss
    #     log(f"{stage}/kl_loss", kl_loss)
    #     log(f"{stage}/total_loss", loss)

    #     return loss

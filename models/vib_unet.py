"""
Unet architecture based on 'climsim_unet.py' with additional variational information bottleneck

VIB implementation based on https://github.com/udeepam/vib/blob/master/vib.ipynb
"""

import logging
from torch import nn
import torch
import numpy as np
from torch.nn import functional as F

from models.climsim_unet import make_residual_connection, ResBlock


class AttentionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 32):
        """
        Block for self attention mechanism.
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.

        """
        super().__init__()

        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels)

        self.Q = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.K = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.V = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.residual_connection = make_residual_connection(in_channels, out_channels)

    def forward(self, x):
        x = self.group_norm(x)

        B, C, L = x.shape
        q = self.Q(x).permute(0, 2, 1)  # (B, L, C)
        k = self.K(x).permute(0, 2, 1)  # (B, L, C)
        v = self.V(x).permute(0, 2, 1)  # (B, L, C)

        h = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
        )
        h = h.permute(0, 2, 1)  # (B, C, L)

        scale = torch.sqrt(torch.tensor(2.0, device=h.device, dtype=h.dtype))

        x = self.residual_connection(x) + h / scale
        # Residual connection with normalisation of the variance to improve stability

        return x


class Parameterize(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int):
        """
        Parameterization layer for the variational information bottleneck.
        Args:
            in_channels (int): Number of input channels.
            latent_dim (int): Dimension of the latent space.
        """
        super().__init__()
        self.mean_layer = nn.Conv1d(in_channels, latent_dim, kernel_size=1)
        self.logvar_layer = nn.Conv1d(in_channels, latent_dim, kernel_size=1)

    def forward(self, x):
        mu = self.mean_layer(x)
        # std = F.softplus(self.std_layer(x) - 5, beta=1)
        logvar = self.logvar_layer(x)
        return mu, logvar


class Reparameterize(nn.Module):
    def __init__(self):
        """
        Reparameterization layer for the variational information bottleneck.
        """
        super().__init__()

    def forward(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(logvar)
        return mu + eps * std


class VIBUNet(nn.Module):
    def __init__(self, beta: float):
        """
        Initializes the VIBUNet model.
        """
        super().__init__()
        self.name = "vib_unet"
        self.beta = beta
        self.enc = nn.ModuleList()
        self.dec = nn.ModuleList()

        self.conv_in = nn.Conv1d(6, 128, kernel_size=3, padding=1)
        self._make_levels()
        self.conv_out = nn.Conv1d(128, 10, kernel_size=3, padding=1)

    def forward(self, x):
        if x.dim() == 2:
            x = self._reshape_from_standard_format(x)

        x = self.conv_in(x)

        skips = []
        # Encoder
        # print("Encoder")
        for down_block in self.enc:
            # print(f"  Down block: {x.shape}")
            for layer in down_block:
                x = layer(x)
                # print(f"    After layer: {x.shape}")
            skips.append(x)

        # print(f"Bottom: {x.shape}")
        # Mid
        mu, logvar = self.mid[0](x)  # Parameterize
        x = self.mid[1](mu, logvar)  # Reparameterize
        x = self.mid[2](x)  # Sparse Attention
        x = self.mid[3](x)  # ResBlock
        # print(f"After mid: {x.shape}")

        # Decoder
        # print("Decoder")
        for up_block in self.dec:
            # print(f"  Up block: {x.shape}")
            skip = skips.pop()
            # print(f"    Skip: {skip.shape}")
            x = torch.cat((x, skip), dim=1)  # Concatenate skip connection
            # print(f"    After concat: {x.shape}")
            for layer in up_block:
                x = layer(x)
                # print(f"    After layer: {x.shape}")

        x = self.conv_out(x)

        x = self._reshape_to_standard_format(x)
        return x, mu, logvar

    def step(self, output, y, log, loss_metric, stage=None):
        """
        Generic step for training, validation, and testing.
        Args:
            batch (tuple): A tuple containing input data and target labels.
            batch_idx (int): Index of the batch.
            stage (str, optional): Stage of the step ('train', 'val', 'test'). Default is None.

        """
        y_hat, mu, logvar = output
        standard_loss = loss_metric(y_hat, y)
        log(f"{stage}/loss", standard_loss)
        # VIB KL Divergence Loss
        # kl_loss = 0.5 * torch.sum(mu.pow(2) + std.pow(2) - 2 * std.log() - 1)

        # mu, std shaped (B, C, L) for example
        kl_per_elem = mu.pow(2) + torch.exp(logvar) - 1 - logvar
        kl_per_sample = 0.5 * torch.sum(kl_per_elem, dim=[1, 2])
        kl_loss = kl_per_sample.mean()
        loss = standard_loss + self.beta * kl_loss
        log(f"{stage}/kl_loss", kl_loss)
        log(f"{stage}/total_loss", loss)

        return loss

    def _reshape_from_standard_format(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshapes the input from standard format (batch, features) to (batch, levels+padding, variables)
        (Note: I think I am interchanging 'variables' and 'features' here)

        Args:
            x: (torch.Tensor) (batch, features) input data in standard format

        Returns:
            padded_x: (torch.Tensor) (batch, features, levels+padding) reshaped input data

        """
        num_levels = 45

        reshaped_x = torch.stack(
            [
                x[:, 0:num_levels],
                x[:, num_levels : num_levels + num_levels],
                torch.repeat_interleave(
                    x[:, 2 * num_levels].unsqueeze(1), num_levels, dim=-1
                ),
                torch.repeat_interleave(
                    x[:, 2 * num_levels + 1].unsqueeze(1), num_levels, dim=-1
                ),
                torch.repeat_interleave(
                    x[:, 2 * num_levels + 2].unsqueeze(1), num_levels, dim=-1
                ),
                torch.repeat_interleave(
                    x[:, 2 * num_levels + 3].unsqueeze(1), num_levels, dim=-1
                ),
            ]
        )
        reshaped_x = reshaped_x.permute(1, 0, 2)  # shape (batch, features, levels)

        padded_x = torch.nn.functional.pad(reshaped_x, (0, 3), mode="constant", value=0)

        logging.info(
            f"Reshaped input from standard format to (batch, features, levels+padding): {padded_x.shape}"
        )
        return padded_x

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

    def _make_levels(self):
        """
        Creates all levels of the U-Net architecture, including the base level. For now hardcoded.
        """
        self._make_level(128, 128)  # Level 1
        self._make_level(128, 256)  # Level 2
        self._make_level(256, 256)  # Level 3
        self._make_level(256, 256, bottom_level=True)  # Level 4

        self.mid = nn.ModuleList(
            [
                Parameterize(in_channels=256, latent_dim=256),
                Reparameterize(),
                AttentionBlock(in_channels=256, out_channels=256),
                ResBlock(in_channels=256, out_channels=256),
            ]
        )

    def _make_level(
        self, in_channels: int, out_channels: int, bottom_level: bool = False
    ):
        """
        Creates a single latent depth of the U-Net architecture (can be seen as a single level / row in the classical U-Net diagram). This is made up of several residual blocks and an down or up sampling operation.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            num_res_blocks (int): Number of residual blocks in this level.
            down (bool): If True, includes a downsampling operation; if False, includes an upsampling operation.
        """

        down_block = self._make_down_block(
            in_channels=in_channels,
            out_channels=out_channels,
            bottom_level=bottom_level,
        )
        up_block = self._make_up_block(
            in_channels=out_channels,
            out_channels=in_channels,
            bottom_level=bottom_level,
        )

        self.enc.append(down_block)
        self.dec.insert(0, up_block)

    def _make_down_block(
        self, in_channels: int, out_channels: int, bottom_level: bool = False
    ):
        """
        Creates a downsampling block consisting of a n residual blocks followed by downsampling.
        Note: we are hardcoding 2 residual blocks per downsampling as per [1] (and 3 upsampling blocks per upsampling).
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        Returns:
            nn.Module: A downsampling block.
        """
        if bottom_level:
            return nn.ModuleList(
                [
                    ResBlock(in_channels=in_channels, out_channels=in_channels),
                    ResBlock(in_channels=in_channels, out_channels=in_channels),
                ]
            )
        else:
            return nn.ModuleList(
                [
                    ResBlock(in_channels=in_channels, out_channels=in_channels),
                    ResBlock(in_channels=in_channels, out_channels=in_channels),
                    nn.Conv1d(
                        in_channels, out_channels, kernel_size=3, padding=1, stride=2
                    ),  # Downsampling by factor of 2
                ]
            )

    def _make_up_block(
        self, out_channels: int, in_channels: int, bottom_level: bool = False
    ):
        """
        Creates an upsampling block consisting of an upsampling operation followed by n residual blocks.
        Note: we are hardcoding 3 residual blocks per upsampling as per [1].
        Args:
            out_channels (int): Number of output channels.
            in_channels (int): Number of input channels.
        Returns:
            nn.Module: An upsampling block.
        """
        cat_channels = 2 * in_channels

        if bottom_level:
            return nn.ModuleList(
                [
                    ResBlock(in_channels=cat_channels, out_channels=in_channels),
                    ResBlock(in_channels=in_channels, out_channels=in_channels),
                    ResBlock(in_channels=in_channels, out_channels=out_channels),
                ]
            )
        else:
            return nn.ModuleList(
                [
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv1d(cat_channels, in_channels, 3, padding=1),
                    ResBlock(in_channels=in_channels, out_channels=in_channels),
                    ResBlock(in_channels=in_channels, out_channels=in_channels),
                    ResBlock(in_channels=in_channels, out_channels=out_channels),
                ]
            )

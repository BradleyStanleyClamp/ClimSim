"""
Script that implements the climsim unet found in .climsim_unet, but with hard attention and sparsity regularisation inspired by SPARTAN [1].
The majority of the code is coppied from models/climsim_unet.py.

References:
[1] https://arxiv.org/abs/2411.06890

"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.climsim_unet import make_residual_connection, ResBlock


class SparseAttention(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        tau: float = 1.0,
        num_groups: int = 32,
    ):
        """
        Initilializes the SparseAttention module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            tau (float): Temperature parameter for Gumbel-Softmax.

        """
        super().__init__()
        self.tau = tau

        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels)

        self.Q = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.K = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.V = nn.Conv1d(in_channels, in_channels, kernel_size=1)

        self.residual_connection = make_residual_connection(in_channels, out_channels)

    def forward(self, x):
        """
        x: (torch.Tensor) Input tensor of shape (batch_size, 'variables', 'levels'), where:
            - 'levels' is the abstract representation of the levels + padding dimension after encoder
            -  'variables' is the features dimension after encoder treated as the channels in the 1d conv sense
        """
        batch_size, variables, levels = x.shape
        Q = self.Q(x).permute(0, 2, 1)  # (batch_size, 'levels','variables')
        K = self.K(x).permute(0, 2, 1)  # (batch_size, 'levels','variables')
        V = self.V(x).permute(0, 2, 1)  # (batch_size, 'levels','variables')

        # Compute attention scores
        logits = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(
            K.size(-1)
        )  # (batch_size, 'levels', 'levels')

        att_weights = F.softmax(logits, dim=-1)  # (batch_size, 'levels', 'levels')

        # Gumbel-Softmax for hard attention
        adjacency = F.gumbel_softmax(
            logits, tau=self.tau, hard=True, dim=-1
        )  # (batch_size, 'levels', 'levels')

        # Compute hard attention weights

        sparse_att_weights = att_weights * adjacency  # (batch_size, 'levels', 'levels')
        self.sparse_att_weights = sparse_att_weights  # For inspection
        # Apply attention
        out = torch.matmul(sparse_att_weights, V)  # (batch_size, 'levels', 'variables')
        out = out.permute(0, 2, 1)  # (batch_size, 'variables', 'levels')

        out = self.residual_connection(x) + out / np.sqrt(2.0)

        path_loss = F.sigmoid(logits).mean(dim=(1, 2)).mean().item()
        # num_paths = adjacency.sum(dim=(1, 2)).mean().item()
        pgt0 = (logits > 0).sum(dim=(1, 2)).float().mean().item()

        return out, path_loss, pgt0


class SparseUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 6,
        out_channels: int = 10,
        tau: float = 1.0,
        log_lambda_paths: float = 0.01,
        lambda_update_rate: float = 0.001,
        target_loss: float = 0.1,
    ):
        """
        Initializes the SparseUNet model.
        Args:
            in_channels (int): Number of input channels. Default is 6.
            out_channels (int): Number of output channels. Default is 10.
            tau (float): Temperature parameter for Gumbel-Softmax. Default is 1.0.
            lambda_paths (float): Initial weight for the sparsity regularization term. Default is 0.01.
            lambda_update_rate (float): Update rate for the lambda_paths parameter. Default is 0.001.
            target_loss (float): Target loss value for GECO. Default is 0.1.
        """
        super().__init__()
        self.name = "sparse_unet"
        self.tau = tau
        self.log_lambda_paths = log_lambda_paths
        self.lambda_update_rate = lambda_update_rate
        self.target_loss = target_loss

        self.enc = nn.ModuleList()
        self.dec = nn.ModuleList()

        self.conv_in = nn.Conv1d(6, 128, kernel_size=3, padding=1)
        self._make_levels()
        self.conv_out = nn.Conv1d(128, 10, kernel_size=3, padding=1)

    def forward(self, x):
        """
        Forward pass of the SparseUNet model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, features) or (batch_size, variables, levels)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, features)
            float: Number of paths in the sparse attention adjacency matrix
        """

        if x.dim() == 2:
            x = self._reshape_from_standard_format(x)

        x = self.conv_in(x)

        skips = []
        # Encoder

        for down_block in self.enc:
            for layer in down_block:
                x = layer(x)
            skips.append(x)

        # Mid
        x, num_paths, pgt0 = self.mid[0](x)  # Sparse Attention
        x = self.mid[1](x)  # ResBlock

        # Decoder
        for up_block in self.dec:
            skip = skips.pop()
            x = torch.cat((x, skip), dim=1)  # Concatenate skip connection
            for layer in up_block:
                x = layer(x)

        x = self.conv_out(x)

        x = self._reshape_to_standard_format(x)
        return x, num_paths, pgt0

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
                SparseAttention(in_channels=256, out_channels=256, tau=self.tau),
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

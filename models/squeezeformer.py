"""
Script that implements the ClimSim Kaggle competition SqueezeFormer model.

[second attempt]

[1] Code based on: https://www.kaggle.com/code/shlomoron/leap-training-1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import logging


class ScaleBias(nn.Module):
    def __init__(self, num_features: int):
        """
        Constructs a Scale and Bias module.

        Args:
            num_features: Number of features of the input feature map
        """
        super().__init__()
        self.scale = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        return x * self.scale + self.bias


class ECA(nn.Module):
    """
    Constructs a ECA module.

    code from https://github.com/BangguWu/ECANet/blob/master/models/eca_module.py

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=5):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class Conv1DBlock(nn.Module):
    def __init__(self, embed_dim: int, out_dim: int, expand_ratio: int = 4):
        """
        Block implementing a 1D convolutional layer. Currently based on my unet implementation. The convolution is applied along the 'levels' dimension.
        Args:
            embed_dim (int): Dimension of the embedding space (input channels)
            out_dim (int): Output dimension (output channels)
        """
        super().__init__()

        self.expand_dim = embed_dim * expand_ratio
        self.expand = nn.Conv1d(
            in_channels=embed_dim, out_channels=self.expand_dim, kernel_size=1
        )

        self.glu = nn.GLU(dim=-2)  # GLU activation along the channel dimension
        self.post_glu_dim = self.expand_dim // 2
        self.bn = nn.BatchNorm1d(self.post_glu_dim)
        self.activation = nn.SiLU()
        self.conv = nn.Conv1d(
            self.post_glu_dim,
            self.post_glu_dim,
            groups=self.post_glu_dim,  # depthwise convolution
            kernel_size=3,
            padding=1,
            stride=1,
        )
        self.eca = ECA(channel=self.post_glu_dim, k_size=5)
        self.project = nn.Conv1d(
            in_channels=self.post_glu_dim, out_channels=out_dim, kernel_size=1
        )
        self.scale_bias = ScaleBias(out_dim)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, Levels, embed_dim)
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, Levels, out_dim)
        """
        residual = x
        x = x.transpose(1, 2)  # (batch_size, expand_dim, Levels)

        x = self.expand(x)  # (batch_size, expand_dim, Levels)

        x = self.glu(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.eca(x)

        x = self.project(x)
        x = x.transpose(1, 2)  # (batch_size, Levels, out_dim)
        x = self.scale_bias(x)
        x = x + residual  # Residual connection

        return x


class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        """
        Self-Attention block using PyTorch's MultiheadAttention module. attentio is applied along the level dimension (I think)
        Args:
            embed_dim (int): Dimension of the embedding space
            num_heads (int): Number of attention heads
        """
        super().__init__()

        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
        )

        self.scale_bias = ScaleBias(embed_dim)
        self.ln1 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Args: x (torch.Tensor): Input tensor of shape (batch_size, Levels, embed_dim)
        Returns: torch.Tensor: Output tensor of shape (batch_size, Levels, embed_dim)
        """
        residual = x
        x, att_w = self.mha(x, x, x)
        x = self.scale_bias(x)
        x = x + residual  # Residual connection

        # with torch.no_grad():
        #     print(
        #         "att_out: mean, std, min, max:",
        #         x.mean().item(),
        #         x.std().item(),
        #         x.min().item(),
        #         x.max().item(),
        #     )
        #     print(
        #         "att_w: mean, std, min, max:",
        #         att_w.mean().item(),
        #         att_w.std().item(),
        #         att_w.min().item(),
        #         att_w.max().item(),
        #     )

        x = self.ln1(x)
        return x


class HeadDense(nn.Module):
    def __init__(self, in_dim: int, head_dim: int):
        """
        Linear layer and activation to project the feature dimension to the head size.
        """
        super().__init__()
        self.head_dim = head_dim
        self.dense = nn.Linear(in_dim, head_dim)
        self.activation = nn.SiLU()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, Levels, embed_dim)
        """

        x = self.activation(self.dense(x))
        return x


class GluMlp(nn.Module):
    def __init__(self, in_dim: int, expanded_dim: int):
        """
        two-layer MLP with GLU activation.
        1st layer expands the dimension to expanded_dim
        2nd layer projects back to in_dim
        1st layer uses GLU activation

        Args:
            in_dim (int): Input dimension
            expanded_dim (int): Expanded dimension for the hidden layer

        """
        super().__init__()

        self.fc1 = nn.Linear(in_dim, expanded_dim)
        self.glu = nn.GLU(dim=-1)
        self.fc2 = nn.Linear(expanded_dim // 2, in_dim)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, Levels, embed_dim)
        """
        x = self.fc1(x)
        x = self.glu(x)
        x = self.fc2(x)
        return x


class ResdiualFFN(nn.Module):
    def __init__(self, in_dim: int, expanded_dim: int):
        """
        Residual Feed Forward Network block with two linear layers and RELU activation.
        The first layer expands the dimension to expanded_dim, and the second layer projects back to in_dim.

        Args:
            in_dim (int): Input dimension
            expanded_dim (int): Expanded dimension for the hidden layer

        Note: this will be applied along the 'feature' dimension which by default is dim=1
        """
        super().__init__()

        self.fc1 = nn.Linear(in_dim, expanded_dim)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(expanded_dim, in_dim)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, Levels, embed_dim)
        """
        residual = x
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = x + residual  # Residual connection
        return x


class ResidualGluMlp(nn.Module):
    def __init__(self, in_dim: int, expanded_dim: int):
        """
        Residual Feed Forward Network block with two linear layers and GLU activation.
        The first layer expands the dimension to expanded_dim, and the second layer projects back to in_dim.

        Args:
            in_dim (int): Input dimension
            expanded_dim (int): Expanded dimension for the hidden layer
        """
        super().__init__()

        self.fc1 = nn.Linear(in_dim, expanded_dim)
        self.glu = nn.GLU(dim=-1)
        self.fc2 = nn.Linear(expanded_dim // 2, in_dim)
        self.scale_bias = ScaleBias(in_dim)
        self.ln = nn.LayerNorm(in_dim)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, Levels, embed_dim)
        """
        residual = x
        x = self.fc1(x)
        x = self.glu(x)
        x = self.fc2(x)
        x = self.scale_bias(x)
        x = x + residual  # Residual connection
        x = self.ln(x)
        return x


class SqueezeFormer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        head_dim: int,
        out_dim: int,
        num_heads: int = 8,
        num_encoder_blocks: int = 12,
    ):
        """
        Implementation of the SqueezeFormer model, based on the ClimSim Kaggle competition winnder [1].
        The model consists of
        - Embedding block: Projecting inputs (features dimension) to higher dimensional space
        - Encoder block: Series of 1D convolutional layers with self attention 'sliding' over the levels dimension and attending across 'features' dimension
        - Decoder block: Projecting to high 'head dimension'
        - Prediction block: Projecting to the output dimension to provide the predicted values

        The standard input is (batch_size, levels, features) (e.g (batch_size, 60, 6))

        Args:
            in_dim (int): Input feature dimension (e.g 6)
            embed_dim (int): Dimension of the embedding space e.g 384
            head_dim (int): Dimension of the head space e.g 64
            out_dim (int): Output dimension (e.g 10)
            num_heads (int): Number of attention heads (e.g 8)
            num_encoder_blocks (int): Number of encoder blocks to use (e.g 12)
        """

        super().__init__()
        logging.info(
            f"Initializing SqueezeFormer2 with in_dim={in_dim}, embed_dim={embed_dim}, out_dim={out_dim}"
        )

        # Embedding
        self.embedding = nn.ModuleList(
            [
                nn.Linear(in_dim, embed_dim),
                nn.LayerNorm(embed_dim),
            ]
        )

        # Encoder (dummy for now)
        self.encoder = nn.ModuleList([])
        for _ in range(num_encoder_blocks):
            self.encoder.append(Conv1DBlock(embed_dim, embed_dim))
            self.encoder.append(ResidualGluMlp(embed_dim, expanded_dim=embed_dim * 4))

            self.encoder.append(SelfAttentionBlock(embed_dim, num_heads=num_heads))
            self.encoder.append(ResidualGluMlp(embed_dim, expanded_dim=embed_dim * 4))

        # Decoder
        self.decoder = nn.ModuleList(
            [
                HeadDense(embed_dim, head_dim),
                GluMlp(head_dim, expanded_dim=head_dim * 2),
            ]
        )

        # Prediction
        self.prediction_head = nn.Linear(head_dim, out_dim)

    def forward(self, x):
        """
        Forward pass of the SqueezeFormer model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, Levels, in_dim) .eg (batch_size, 60, 6)
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, variables) .eg (batch_size, 128) (as this format is used for evaluation)
        """
        # If standard input is goven, then we can reshape
        if x.ndim == 2:
            x = self._reshape_from_standard_format(x)

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

    def _reshape_from_standard_format(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshapes the input from standard format (batch, features) to (batch, levels, variables)
        (Note: I think I am interchanging 'variables' and 'features' here)

        Args:
            x: (torch.Tensor) (batch, features) input data in standard format

        Returns:
            reshaped_x: (torch.Tensor) (batch, levels, features) reshaped input data

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
        reshaped_x = reshaped_x.permute(1, 2, 0)  # shape (batch, levels, features)

        print(
            f"Reshaped input from standard format to (batch, levels, features): {reshaped_x.shape}"
        )
        return reshaped_x

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

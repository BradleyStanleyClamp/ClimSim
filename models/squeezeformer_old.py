# """
# Script that implements the ClimSim Kaggle competition SqueezeFormer model.

# Code based on: https://www.kaggle.com/code/shlomoron/leap-training-1
# """

# import logging
# import torch
# import numpy as np
# import torch.nn as nn


# class GLUMlp(nn.Module):
#     def __init__(self, in_dim: int, expanded_dim: int):
#         super().__init__()

#         self.fc1 = nn.Linear(in_dim, expanded_dim)
#         self.glu = nn.GLU(dim=-1)
#         self.fc2 = nn.Linear(expanded_dim // 2, in_dim)

#     def forward(self, x):
#         """
#         Input shape: (batch, levels, in_dim)
#         """
#         x = self.fc1(x)
#         x = self.glu(x)
#         x = self.fc2(x)
#         return x


# class ScaleBiasLayer(nn.Module):
#     def __init__(self, channels: int):
#         super().__init__()
#         self.scale = nn.Parameter(torch.ones(1, 1, channels))
#         self.bias = nn.Parameter(torch.zeros(1, 1, channels))

#     def forward(self, x):
#         """
#         Input shape: (batch, levels, channels)
#         """

#         return x * self.scale + self.bias


# class ECA(nn.Module):
#     def __init__(self, levels: int, kernel_size: int = 5):
#         """
#         Efficient Channel Attention module.
#         """

#         super().__init__()

#         self.avg_pool = nn.AvgPool1d(kernel_size=levels)
#         self.conv = nn.Conv1d(
#             1, 1, kernel_size=kernel_size, stride=1, padding="same", bias=False
#         )
#         self.activation = nn.Sigmoid()

#     def forward(self, x):
#         """
#         Input shape: (batch, levels, channels)
#         [note: I am bad and have been interchanging channels with features and variables!]
#         """
#         B, L, C = x.shape
#         h = x.view(B, C, L)  # (batch, channels, levels)
#         h = self.avg_pool(h)  # (batch, channels, 1)
#         h = h.view(B, 1, C)  # (batch, 1, channels)
#         h = self.conv(h)  # (batch, 1, channels)
#         h = h.squeeze(1)  # (batch, channels)
#         h = self.activation(h)  # (batch, channels)
#         att = h.unsqueeze(1)  # (batch, 1, channels)
#         return att * x


# class Conv1DBlockSqueezeFormer(nn.Module):
#     def __init__(
#         self,
#         in_features: int,
#         levels: int,
#         kernel_size: int,
#         dilation_rate: int = 1,
#         expand_ratio: int = 4,
#         activation=nn.SiLU,
#         norm_momentum: float = 0.95,
#     ):
#         """
#         Convolutional 1D Block used in SqueezeFormer model.

#         Args:
#             in_features (int): Number of input features (last dimension)
#             levels (int): Number of levels (second dimension)
#             kernel_size (int): Size of the convolutional kernel
#             dilation_rate (int): Dilation rate for the convolution
#             expand_ratio (int): Expansion ratio for the features
#         """

#         super().__init__()
#         self.expanded_features = in_features * expand_ratio

#         self.expand = nn.Linear(in_features, self.expanded_features)
#         self.glu = nn.GLU(dim=-1)

#         conv_channel_size = self.expanded_features // 2
#         self.conv1d = nn.Conv1d(
#             in_channels=conv_channel_size,
#             out_channels=conv_channel_size,
#             kernel_size=kernel_size,
#             padding="same",
#             dilation=dilation_rate,
#             groups=conv_channel_size,
#             bias=False,
#         )

#         self.batch_norm = nn.BatchNorm1d(
#             conv_channel_size, eps=0.001, momentum=norm_momentum
#         )

#         self.activation = activation()

#         self.eca = ECA(levels=levels, kernel_size=5)

#         self.project = nn.Linear(conv_channel_size, in_features)

#         self.scale_bias_1 = ScaleBiasLayer(in_features)

#         self.glu_mlp = GLUMlp(in_dim=in_features, expanded_dim=in_features * 4)

#         self.scale_bias_2 = ScaleBiasLayer(in_features)

#         self.layer_norm = nn.LayerNorm(in_features, eps=1e-6)

#     def forward(self, x):
#         """
#         Forward pass of the Conv1DBlockSqueezeFormer.

#         Args:
#             x (torch.Tensor): Input tensor of shape (batch, levels, in_features)
#         """
#         B, L, C = x.shape
#         h = self.expand(x)  # (batch, levels, in_features * 4)
#         h = self.glu(h)  # (batch, levels, in_features * 2)
#         h = h.view(B, C * 2, L)  # (batch, in_features * 2, levels)
#         h = self.conv1d(h)  # (batch, in_features * 2, levels)
#         h = self.batch_norm(h)  # (batch, in_features * 2, levels)
#         h = self.activation(h)  # (batch, in_features * 2, levels)
#         h = h.view(B, L, C * 2)  # (batch, levels, in_features * 2)
#         h = self.eca(h)  # (batch, levels, in_features * 2)
#         h = self.project(h)  # (batch, levels, in_features)
#         h = self.scale_bias_1(h)  # (batch, levels, in_features)

#         h = h + x  # Residual connection
#         residual = x
#         x = self.glu_mlp(x)  # (batch, levels, in_features)
#         x = self.scale_bias_2(x)  # (batch, levels, in_features)
#         x = self.layer_norm(x + residual)  # Residual connection

#         return x


# class TransformerEncoder(nn.Module):
#     def __init__(self, embed_dim: int, num_heads: int, feedforward_dim: int):
#         super().__init__()

#         self.att = nn.MultiheadAttention(
#             embed_dim=embed_dim, num_heads=num_heads, batch_first=True
#         )
#         self.scale_bias1 = ScaleBiasLayer(embed_dim)
#         self.ln1 = nn.LayerNorm(embed_dim, eps=1e-6)
#         self.ffn = GLUMlp(in_dim=embed_dim, expanded_dim=feedforward_dim)
#         self.scale_bias2 = ScaleBiasLayer(embed_dim)
#         self.ln2 = nn.LayerNorm(embed_dim, eps=1e-6)

#     def forward(self, x):
#         """
#         Forward pass of the TransformerEncoder.

#         Args:
#             x (torch.Tensor): Input tensor of shape (batch, levels, embed_dim)
#         """

#         residual = x
#         x, _ = self.att(x, x, x)  # Self-attention
#         x = self.scale_bias1(x)
#         x = self.ln1(x + residual)  # Residual connection

#         residual = x
#         x = self.ffn(x)
#         x = self.scale_bias2(x)
#         x = self.ln2(x + residual)  # Residual connection

#         return x


# class HeadDense(nn.Module):
#     def __init__(self, in_dim: int, head_dim: int):
#         super().__init__()
#         self.head_dim = head_dim
#         self.dense = nn.Linear(in_features=in_dim, out_features=head_dim)
#         self.activation = nn.SiLU()

#     def forward(self, x):
#         """
#         Input shape: (batch, levels, in_dim)
#         """
#         return self.activation(self.dense(x))


# class SqueezeFormer(nn.Module):
#     def __init__(
#         self,
#         in_dim: int,
#         embed_dim: int,
#         levels: int,
#         head_dim: int,
#         out_dim: int,
#         num_heads: int = 4,
#         num_encoder_blocks: int = 12,
#     ):
#         """
#         SqueezeFormer inspured model based on the ClimSim Kaggle competition winner.

#         Model is expected to take in input of shape (batch, levels, variables)

#         Args:
#             in_dim (int): Number of input features (variables)
#             embed_dim (int): Embedding dimension
#             levels (int): Number of levels (second dimension)
#             head_dim (int): Dimension of the head dense layer
#             out_dim (int): Number of output features (variables)
#         """

#         super().__init__()
#         self.in_dim = in_dim
#         self.embed_dim = embed_dim
#         self.levels = levels
#         self.out_dim = out_dim
#         self.head_dim = head_dim
#         self.num_encoder_blocks = num_encoder_blocks
#         self.num_heads = num_heads

#         # Embedding
#         self.embedding = nn.ModuleList(
#             [
#                 nn.Linear(self.in_dim, self.embed_dim),
#                 nn.LayerNorm(self.embed_dim),
#             ]
#         )

#         # Encoder
#         self.encoder = nn.ModuleList([])
#         for _ in range(self.num_encoder_blocks):
#             self.encoder.append(
#                 nn.ModuleList(
#                     [
#                         Conv1DBlockSqueezeFormer(
#                             in_features=self.embed_dim,
#                             levels=self.levels,
#                             kernel_size=15,
#                         ),
#                         TransformerEncoder(
#                             self.embed_dim,
#                             self.num_heads,
#                             feedforward_dim=4 * self.embed_dim,
#                         ),
#                     ]
#                 )
#             )

#         # Decoder
#         self.decoder = nn.ModuleList(
#             [
#                 HeadDense(self.embed_dim, self.head_dim),
#                 GLUMlp(self.head_dim, expanded_dim=self.head_dim * 2),
#             ]
#         )

#         # Prediction head
#         self.prediction_head = nn.Linear(self.head_dim, out_dim)

#         # Confidence head
#         self.confidence_head = nn.Linear(self.head_dim, out_dim)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass of the SqueezeFormer model.

#         Args:
#             x (torch.Tensor): Input tensor of shape (batch, levels, variables)

#         Returns:
#             torch.Tensor: Output tensor of shape (batch, levels, out_dim)
#         """

#         # Embedding
#         for layer in self.embedding:
#             x = layer(x)

#         # Encoder
#         for conv_block, transformer in self.encoder:
#             x = conv_block(x)
#             x = transformer(x)

#         # Decoder
#         for layer in self.decoder:
#             x = layer(x)

#         # Prediction and confidence heads
#         prediction = self.prediction_head(x)
#         confidence = self.confidence_head(x)
#         B, L, D = prediction.shape
#         prediction = prediction.view(B, D, L)
#         out = self._reshape_to_standard_format(prediction)

#         return out

#     def _reshape_to_standard_format(self, x):
#         """
#         Reshapes the output to standard format (n_samples, features) from (n_samples, levels, variables)
#         Args:
#             x: (torch.Tensor) (n_samples, levels, variables) output data from the model
#         Returns:
#             reshaped_x: (torch.Tensor) (n_samples, features) reshaped output data
#         """
#         # slice once (no copy — view)
#         # x0_60 = x[:, :, 0:60]            # shape (n, levels, 60)

#         # flatten the first two levels into (n, 120) in one op
#         first_two = x[:, 0:2, :].reshape(x.shape[0], -1)  # shape (n, 2*60) == (n,120)
#         first_two = first_two[
#             :, :-15
#         ]  # HARDCODED REMOVAL OF TOP SH Levels # shape (n, 105)
#         # compute the per-level means for levels 2..9 in one op
#         means_2_to_9 = x[:, 2:10, :].mean(dim=2)  # shape (n, 8)

#         # concatenate once
#         output = torch.cat([first_two, means_2_to_9], dim=1)  # shape (n, 128)
#         return output

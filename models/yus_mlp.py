"""
Implementation of a Yu, Hannah, et al., 2024's Multi-Layer Perceptron (MLP). This model was optimised through a 8257 sweep and is designed for the standard input case, this model is implemented in PyTorch.

[ClimSim-Online: Dataset and Framework for Hybrid Climate Emulation](https://arxiv.org/abs/2306.08754)
"""

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


class YusMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation=F.leaky_relu,
    ):
        """
        Initializes the MLP model.
        Args:
            input_dim (int): Dimension of the input features.
            hidden_dims (List[int]): Dimensions of the hidden layers.
            output_dim (int): Dimension of the output.
            activation (callable): Activation function to be used. Default is ReLU.

        """
        super().__init__()

        self.nhidden = len(hidden_dims)

        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dims[i], hidden_dims[i + 1]) for i in range(self.nhidden - 1)]
        )

        self.last_hidden_layer = nn.Linear(hidden_dims[-1], output_dim)


        # Functionality to deal with removal of specific humidity levels
        adjusted_linear_output_dim = output_dim - 8  # Adjust output dim for linear branch

        self.output_lin = nn.Linear(output_dim, adjusted_linear_output_dim)  # linear branch
        self.output_relu = nn.Sequential(             # relu branch
            nn.Linear(output_dim, 8),
            nn.ReLU()
        )

        self.activation = activation

    def forward(self, x):
        """
        Forward pass through the MLP.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        x = self.activation(self.last_hidden_layer(x))
        x = torch.cat([self.output_lin(x), self.output_relu(x)], dim=-1)
        return x

"""
Implementation of a simple Multi-Layer Perceptron (MLP) model using PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        nhidden: int,
        activation=F.relu,
    ):
        """
        Initializes the MLP model.
        Args:
            input_dim (int): Dimension of the input features.
            hidden_dim (int): Dimension of the hidden layers, for now all have equal size.
            output_dim (int): Dimension of the output.
            nhidden (int): Number of hidden layers.
            activation (callable): Activation function to be used. Default is ReLU.

        """
        super().__init__()

        self.nhidden = nhidden

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(nhidden)]
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)
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
        x = self.output_layer(x)
        return x

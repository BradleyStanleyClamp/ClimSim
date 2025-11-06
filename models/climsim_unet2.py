"""
Script that implements a U-Net architecture based on description in [1] and code from [2].

[1]: Stable Machine-Learning Parameterization of Subgrid Processes in a Comprehensive Atmospheric Model Learned From Embedded Convection-Permitting Simulations
Zeyuan Hu, Akshay Subramaniam, Zhiming Kuang, Jerry Lin, Sungduk Yu, Walter M. Hannah, Noah D. Brenowitz, Josh Romero, Michael S. Pritchard
https://arxiv.org/abs/2407.00124


[2] https://github.com/g4vrel/sde_ddpm/tree/master
"""

import logging
from torch import nn
import torch
import numpy as np

def make_residual_connection(in_channels:int, out_channels:int):
        """
        Creates a residual connection. If the number of input and output channels differ, a 1x1 convolution is used to match dimensions.
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        Returns:
            nn.Module: A module representing the residual connection.
        """
        if in_channels != out_channels:
            return nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            return nn.Identity()

class ResBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, num_groups:int=32):
        """
        Classical residual block. 
        Note: this may differ slightly from [1]'s implementation, as it is unclear exactly what they choose to do, for example: 
        1. The second silu activation after conv, the paper seems to say 'y =Conv1D(GM(Conv1D(silu(GM(x))))) + x' but the standard approach seems to be two?
        
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            num_groups (int): Number of groups for GroupNorm.
        """

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.block0 = self._make_block(in_channels, out_channels, num_groups)
        self.block1 = self._make_block(out_channels, out_channels, num_groups)
        self.residual_connection = make_residual_connection(in_channels, out_channels)

    def _make_block(self, in_channels:int, out_channels:int, num_groups:int=32):
        """
        Creates a residual block consisting of GroupNorm, SiLU activation, and a 1D convolution.
        Args:
            channels (int): Number of input channels.
            num_groups (int): Number of groups for GroupNorm.
        Returns:
            nn.Sequential: A sequential container of the residual block.
        """
        return nn.Sequential(
            nn.GroupNorm(num_groups=num_groups, num_channels=in_channels),
            nn.SiLU(),
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
        )

    def forward(self, x):
        h = self.block0(x)
        h = self.block1(h)
        x = (self.residual_connection(x) + h) / np.sqrt(2.0)# Residual connection with normalisation of the variance to improve stability
        return x
    
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

        h = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)

        h = h.permute(0, 2, 1)  # (B, C, L)

        x = self.residual_connection(x) + h / np.sqrt(2.0)  # Residual connection with normalisation of the variance to improve stability 
        return x

class ClimSimUNet(nn.Module):
    def __init__(self):
        """
        Initializes the ClimSimUNet model.
        """
        super().__init__()

        self.enc = nn.ModuleList()
        self.dec = nn.ModuleList()

        self.conv_in = nn.Conv1d(48, 128, kernel_size=3, padding=1)
        self._make_levels()
        self.conv_out = nn.Conv1d(128, 13, kernel_size=3, padding=1)

    def forward(self, x):
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
        for layer in self.mid:
            x = layer(x)
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
        return x


    def _make_levels(self):
        """
        Creates all levels of the U-Net architecture, including the base level. For now hardcoded.
        """
        self._make_level(128, 128)  # Level 1
        self._make_level(128, 256)  # Level 2
        self._make_level(256, 256)  # Level 3
        self._make_level(256, 256, bottom_level=True)  # Level 4

        self.mid = nn.ModuleList([
            AttentionBlock(in_channels=256, out_channels=256),
            ResBlock(in_channels=256, out_channels=256),
        ])



    def _make_level(self, in_channels:int, out_channels:int, bottom_level:bool=False):
        """
        Creates a single latent depth of the U-Net architecture (can be seen as a single level / row in the classical U-Net diagram). This is made up of several residual blocks and an down or up sampling operation.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            num_res_blocks (int): Number of residual blocks in this level.
            down (bool): If True, includes a downsampling operation; if False, includes an upsampling operation.
        """

        down_block = self._make_down_block(in_channels=in_channels, out_channels=out_channels, bottom_level=bottom_level)
        up_block = self._make_up_block(in_channels=out_channels, out_channels=in_channels, bottom_level=bottom_level)

        self.enc.append(down_block)
        self.dec.insert(0, up_block)

    def _make_down_block(self, in_channels:int, out_channels:int, bottom_level:bool=False):
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
            return nn.ModuleList([
                ResBlock(in_channels=in_channels, out_channels=in_channels),
                ResBlock(in_channels=in_channels, out_channels=in_channels),
            ])
        else:
            return nn.ModuleList([
                ResBlock(in_channels=in_channels, out_channels=in_channels),
                ResBlock(in_channels=in_channels, out_channels=in_channels),
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, stride=2) # Downsampling by factor of 2
            ])

    def _make_up_block(self, out_channels:int, in_channels:int, bottom_level:bool=False):
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
        print(f"Making up block: cat_channels={cat_channels}, in_channels={in_channels}, out_channels={out_channels}")
        
        if bottom_level:
            return nn.ModuleList([
                ResBlock(in_channels=cat_channels, out_channels=in_channels),
                ResBlock(in_channels=in_channels, out_channels=in_channels),
                ResBlock(in_channels=in_channels, out_channels=out_channels),
            ])
        else:
            return nn.ModuleList([
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(cat_channels, in_channels, 3, padding=1),
                ResBlock(in_channels=in_channels, out_channels=in_channels),
                ResBlock(in_channels=in_channels, out_channels=in_channels),
                ResBlock(in_channels=in_channels, out_channels=out_channels),
            ])
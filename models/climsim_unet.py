"""
Script that implements a U-Net architecture based on [1] tasks using PyTorch.

[1]: Stable Machine-Learning Parameterization of Subgrid Processes in a Comprehensive Atmospheric Model Learned From Embedded Convection-Permitting Simulations
Zeyuan Hu, Akshay Subramaniam, Zhiming Kuang, Jerry Lin, Sungduk Yu, Walter M. Hannah, Noah D. Brenowitz, Josh Romero, Michael S. Pritchard
https://arxiv.org/abs/2407.00124
"""

from torch import nn



class UNetBlockNoAttention(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, use_skip: bool, down_block: bool=False, group_norm_groups=32, activation=nn.SiLU()):
        super().__init__()
        conv0_stride = 2 if down_block else 1 
        
        self.group_norm0 = nn.GroupNorm(num_groups=group_norm_groups, num_channels=in_channels)
        self.conv0 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, stride=conv0_stride)
        self.group_norm1 = nn.GroupNorm(num_groups=group_norm_groups, num_channels=out_channels)
        self.conv1 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)

        self.activation = activation

        self.use_skip = use_skip
        if use_skip:
            self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.group_norm0(x)
        x = self.activation(x)
        x = self.conv0(x)
        x = self.group_norm1(x)
        x = self.conv1(x)

        if self.use_skip:
            skip = self.skip(x)
            return x, skip
        
        return x

class ClimSimUNet(nn.Module):
    def __init__(self ):
        """
        Initilisation for ClimSimUNet model.
        """

        super().__init__()

        # Conv block 
        self.conv_in = nn.Conv1d(48, 128, kernel_size=3, padding=1)

        # Encoder block
        self.enc = nn.ModuleDict()
        self.enc['64_block0'] = UNetBlockNoAttention(in_channels=128, out_channels=128, use_skip=False, group_norm_groups=32, activation=nn.SiLU()) # shape 64x128
        self.enc['64_block1'] = UNetBlockNoAttention(in_channels=128, out_channels=128, use_skip=False, group_norm_groups=32, activation=nn.SiLU())
        self.enc['32_down'] = UNetBlockNoAttention(in_channels=128, out_channels=128, down_block=True, use_skip=True, group_norm_groups=32, activation=nn.SiLU())
        self.enc['32_block0'] = UNetBlockNoAttention(in_channels=128, out_channels=256, use_skip=True, group_norm_groups=32, activation=nn.SiLU())
        self.enc['32_block1'] = UNetBlockNoAttention(in_channels=256, out_channels=256, use_skip=False, group_norm_groups=32, activation=nn.SiLU())
        self.enc['16_down'] = UNetBlockNoAttention(in_channels=256, out_channels=256, down_block=True, use_skip=True, group_norm_groups=32, activation=nn.SiLU())
        self.enc['16_block0'] = UNetBlockNoAttention(in_channels=256, out_channels=256, use_skip=True, group_norm_groups=32, activation=nn.SiLU())
        self.enc['16_block1'] = UNetBlockNoAttention(in_channels=256, out_channels=256, use_skip=False, group_norm_groups=32, activation=nn.SiLU()) 
        self.enc['8_down'] = UNetBlockNoAttention(in_channels=256, out_channels=256, down_block=True, use_skip=True, group_norm_groups=32, activation=nn.SiLU())
        self.enc['8_block0'] = UNetBlockNoAttention(in_channels=256, out_channels=256, use_skip=True, group_norm_groups=32, activation=nn.SiLU())
        # self.enc['8_block1'] = UNetBlockNoAttention(in_channels=256,
        # Base layer 


        # Decoder block

    def forward(self, x):
        assert x.shape == (x.shape[0], 48, 64), f"Input shape must be (batch_size, 48, 64), but got {x.shape}"
        x = self.conv_in(x)
        assert x.shape == (x.shape[0], 128, 64), f"After conv_in, shape must be (batch_size, 128, 64), but got {x.shape}"
        x = self.enc['64_block0'](x)
        assert x.shape == (x.shape[0], 128, 64), f"After 64_block0, shape must be (batch_size, 128, 128), but got {x.shape}"
        x = self.enc['64_block1'](x)
        assert x.shape == (x.shape[0], 128, 64), f"After 64_block1, shape must be (batch_size, 128, 64), but got {x.shape}"
        x, skip_32 = self.enc['32_down'](x)
        assert x.shape == (x.shape[0], 128, 32), f"After 32_down, shape must be (batch_size, 128, 32), but got {x.shape}"
        x, skip_32_block0 = self.enc['32_block0'](x)
        assert x.shape == (x.shape[0], 256, 32), f"After 32_block0, shape must be (batch_size, 256, 32), but got {x.shape}"
        x = self.enc['32_block1'](x)

        return x
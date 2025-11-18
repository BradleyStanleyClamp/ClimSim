"""
Pytorch dataset that loads from raw data files
"""
from torch.utils.data import Dataset
import os
import torch 
import xarray as xr

class
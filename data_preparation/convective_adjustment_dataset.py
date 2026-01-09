"""
Dataset utilising the radiative transfer column model to perform convective adjustment.
Adapted from Brian Rose's Climate Modeling class https://www.atmos.albany.edu/facstaff/brose/classes/ENV480_Spring2014/styled-5/code-2/index.html
"""
import numpy as np
import torch 
from torch.utils.data import Dataset
from radiative_transfer_column_model import column


mycolumn = column()

print(mycolumn)
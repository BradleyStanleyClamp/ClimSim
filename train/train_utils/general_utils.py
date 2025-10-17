"""
General utility functions for training and evaluation.
"""

import os
import random
import numpy as np
import torch
from lightning.pytorch import seed_everything
from hydra.core.hydra_config import HydraConfig


def seed_everything(seed: int):
    """Set random seed for reproducibility, across all libraries used."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



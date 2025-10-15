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


def is_test_run() -> bool:
    """
    A slightly hacky way to determine if this is a test run, by checking if
    "hydra=test" is in the overrides. This requires that the user actually
    specifies this when calling the script, e.g.: hydra=test
    """

    hc = HydraConfig.get()
    task_overrides = getattr(hc.overrides, "task", [])  # other overrides

    if any(
        o.startswith("hydra=") and o.split("=", 1)[1] == "test" for o in task_overrides
    ):
        return True
    else:
        return False

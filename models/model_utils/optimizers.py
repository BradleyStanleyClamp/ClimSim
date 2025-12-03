"""
Script for selecting optimizers for training.
"""

import torch


def select_optimizer(optimizer_name: str):
    """
    Selects and returns an optimizer based on the provided optimizer name.
    Args:
        optimizer_name (str): Name of the optimizer to be selected.
    Returns:
        optimizer (callable): The optimizer class.
    """
    if optimizer_name == "Adam":
        return torch.optim.Adam
    elif optimizer_name == "RAdam":
        return torch.optim.RAdam
    elif optimizer_name == "AdamW":
        return torch.optim.AdamW
    else:
        raise ValueError(f"Optimizer {optimizer_name} not recognized.")

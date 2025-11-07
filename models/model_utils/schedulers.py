"""
Script containing learning rate schedulers for models.
"""
import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf


def select_scheduler(scheduler_name: str, scheduler_cfg: DictConfig, optimizer: torch.optim.Optimizer):
    """
    Selects and returns a scheduler based on the provided scheduler name.
    Args:
        scheduler_name (str): Name of the scheduler to be selected.
    Returns:
        scheduler (callable): The scheduler class.
    """
    if type(scheduler_cfg) is not DictConfig:
        scheduler_cfg = OmegaConf.create(scheduler_cfg)

    if scheduler_name == "cyclic":
        return cyclic_optimizer(optimizer)
    elif scheduler_name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_cfg.step_size, gamma=scheduler_cfg.gamma)
    else:
        raise ValueError(f"Scheduler {scheduler_name} not recognized.")

def cyclic_optimizer(optimizer, base_lr=2.5e-4, max_lr=2.5e-3, step_size=3285, scale_fn=lambda x: 1/(2.**(x-1))):
    """
    Creates a cyclic learning rate scheduler. Hardcoded for Yus MLP for now.

    """

    batch_size = 3072
    data_quantity = 10091520
    step_size = data_quantity // batch_size * 4  # 4 epochs
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=step_size, scale_fn=scale_fn)
    return scheduler
"""
Script containing learning rate schedulers for models.
"""

import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import LambdaLR
import math


def select_scheduler(
    scheduler_name: str, scheduler_cfg: DictConfig, optimizer: torch.optim.Optimizer
):
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
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=scheduler_cfg.step_size, gamma=scheduler_cfg.gamma
        )
    elif scheduler_name == "lambda_lr":
        lr_lambda = make_lambda_lr_scheduler(
            num_warmup_steps=scheduler_cfg.num_warmup_steps,
            warmup_method=scheduler_cfg.warmup_method,
            num_training_steps=scheduler_cfg.num_training_steps,
            num_cycles=scheduler_cfg.num_cycles,
        )
        return LambdaLR(optimizer, lr_lambda=lr_lambda)
    else:
        raise ValueError(f"Scheduler {scheduler_name} not recognized.")


def cyclic_optimizer(
    optimizer,
    base_lr=2.5e-4,
    max_lr=2.5e-3,
    step_size=3285,
    scale_fn=lambda x: 1 / (2.0 ** (x - 1)),
):
    """
    Creates a cyclic learning rate scheduler. Hardcoded for Yus MLP for now.

    """

    batch_size = 3072
    data_quantity = 10091520
    step_size = data_quantity // batch_size * 4  # 4 epochs
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=base_lr,
        max_lr=max_lr,
        step_size_up=step_size,
        scale_fn=scale_fn,
    )
    return scheduler


def make_lambda_lr_scheduler(
    num_warmup_steps, warmup_method, num_training_steps, num_cycles
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            if warmup_method == "log":
                return 0.10 ** (num_warmup_steps - current_step)
            else:
                return 2 ** (-(num_warmup_steps - current_step))
        else:
            progress = float(current_step - num_warmup_steps) / float(
                max(1, num_training_steps - num_warmup_steps)
            )
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))
            return max(0.0, cosine_decay)

    return lr_lambda

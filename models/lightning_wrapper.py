"""
PyTorch Lightning wrapper module for efficient reproduction of training and evaluation of models.

"""

import logging
import lightning as L
import torch
from torch import nn
from models.model_utils.optimizers import select_optimizer
from models.model_utils.schedulers import select_scheduler
from omegaconf import DictConfig
import math


class LightningWrapper(L.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        loss=torch.nn.MSELoss(),
        optimizer="Adam",
        scheduler_cfg=None,
        lr=1e-3,
    ):
        """
        Initializes the LightningWrapper with a PyTorch model.
        Args:
            model (torch.nn.Module): The PyTorch model to be wrapped.
            loss (callable): Loss function to be used during training. Default is Mean Squared Error.
            optimizer (callable):Name of optimizer to use. Default is Adam.
            lr (float): Learning rate for the optimizer. Default is 1e-3.
            scheduler_cfg (DictConfig): Configuration for the learning rate scheduler. Default is None.
        """
        super().__init__()

        self.model = model

        self.loss = loss
        self.optimizer = optimizer
        self.scheduler_cfg = (
            DictConfig(scheduler_cfg) if scheduler_cfg is not None else None
        )
        self.lr = lr

    def forward(self, x):
        """
        Forward pass through the model.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Model output.
            (optional) float: Number of paths in the sparse attention adjacency matrix
        """
        return self.model(x)

    def step(self, batch, batch_idx, stage=None):
        """
        Generic step for training, validation, and testing. That handles different models i.e with/without num_paths output.
        Args:
            batch (tuple): A tuple containing input data and target labels.
            batch_idx (int): Index of the batch.
            stage (str, optional): Stage of the step ('train', 'val', 'test'). Default is None.

        """
        if len(batch) == 2:
            x, y = batch
            output = self(x)
        elif len(batch) == 4:
            xnh, xsh, ynh, ysh = batch
            output_nh = self(xnh)
            output_sh = self(xsh)
            output = (output_nh, output_sh)
            y = (ynh, ysh)

        if hasattr(self.model, "step"):
            loss = self.model.step(output, y, self.log, self.loss, stage)

        else:

            y_hat = output

            loss = self.loss(y_hat, y)

            self.log(f"{stage}/loss", loss)

        return loss

    def training_step(self, batch, batch_idx):
        """
        Training step for a single batch.
        Args:
            batch (tuple): A tuple containing input data and target labels.
            batch_idx (int): Index of the batch.
        Returns:
            torch.Tensor: Loss value for the batch.
        """
        return self.step(batch, batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        """
        Validation step for a single batch.
        Args:
            batch (tuple): A tuple containing input data and target labels.
            batch_idx (int): Index of the batch.
        Returns:
            torch.Tensor: Loss value for the batch.
        """
        return self.step(batch, batch_idx, stage="val")

    def test_step(self, batch, batch_idx):
        """
        Test step for a single batch.
        Args:
            batch (tuple): A tuple containing input data and target labels.
            batch_idx (int): Index of the batch.
        Returns:
            torch.Tensor: Loss value for the batch.
        """
        return self.step(batch, batch_idx, stage="test")

    def configure_optimizers(self):
        """
        Configures the optimizer for training.
        Returns:
            torch.optim.Optimizer: The optimizer to be used for training.
        """
        optimizer = select_optimizer(self.optimizer)(self.parameters(), lr=self.lr)
        logging.info(f"Using optimizer: {self.optimizer}")

        if self.scheduler_cfg is not None and self.scheduler_cfg != "None":
            scheduler = select_scheduler(
                self.scheduler_cfg["name"], self.scheduler_cfg, optimizer
            )
            logging.info(f"Using scheduler: {self.scheduler_cfg['name']}")
            return [optimizer], [scheduler]

        else:
            return optimizer

    def check_gradients(
        self,
        high_threshold: float = 1e3,
        low_threshold: float = 1e-6,
        clip_grad_norm: float | None = None,
        log_prefix: str = "",
    ):
        """
        Inspect gradients for exploding / vanishing behavior.

        Args:
            high_threshold: if total norm or max element-wise grad > this -> possible exploding grads.
            low_threshold: if total norm and max element-wise grad < this -> possible vanishing grads.
            clip_grad_norm: if not None, apply torch.nn.utils.clip_grad_norm_ with this value.
            log_prefix: optional prefix for logged keys (e.g. 'train' / 'val').
        Returns:
            dict of metrics: {'total_norm', 'max_grad', 'min_nonzero_grad', 'zero_frac', 'problem'}
        """
        grads = [p.grad.detach() for p in self.parameters() if p.grad is not None]
        if len(grads) == 0:
            # nothing to do
            return {
                "total_norm": 0.0,
                "max_grad": 0.0,
                "min_nonzero_grad": 0.0,
                "zero_frac": 1.0,
                "problem": None,
            }

        # total norm (sqrt of sum of squared param norms)
        try:
            norms = torch.stack([g.norm() for g in grads])
            total_norm = torch.norm(norms).item()  # scalar
        except Exception:
            # fallback robust computation
            total_norm = math.sqrt(
                sum(float(g.float().norm().item()) ** 2 for g in grads)
            )

        # max absolute element-wise gradient
        max_grad = max(float(g.abs().max().item()) for g in grads)

        # min non-zero element-wise gradient across all params (if exists)
        min_nonzero = None
        for g in grads:
            nonzero = g.abs().view(-1)[g.abs().view(-1) > 0]
            if nonzero.numel() > 0:
                cur_min = float(nonzero.min().item())
                if min_nonzero is None or cur_min < min_nonzero:
                    min_nonzero = cur_min
        if min_nonzero is None:
            min_nonzero = 0.0

        # fraction of zero gradients
        total_elements = sum(g.numel() for g in grads)
        zero_elements = sum(int((g == 0).sum().item()) for g in grads)
        zero_frac = zero_elements / total_elements if total_elements > 0 else 1.0

        # optionally clip gradients to prevent explosions
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), clip_grad_norm)

        # decide whether this looks like exploding / vanishing
        problem = None
        if total_norm > high_threshold or max_grad > high_threshold:
            problem = "exploding"
        elif total_norm < low_threshold and max_grad < low_threshold:
            # also detect very many zeros which can indicate dead/vanishing grads
            if zero_frac > 0.9:
                problem = "vanishing_or_sparse"
            else:
                problem = "vanishing"

        # Log scalars with Lightning so WandB (or other logger) captures them
        # log_prefix can be 'train/' or 'val/' if you want to distinguish
        pref = (
            f"{log_prefix}grad/"
            if log_prefix and not log_prefix.endswith("/")
            else f"{log_prefix}grad/"
        )
        # use self.log so Lightning routes it to the logger (WandB)
        try:
            self.log(
                f"{pref}total_norm",
                total_norm,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                f"{pref}max_grad",
                max_grad,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                f"{pref}min_nonzero_grad",
                min_nonzero,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                f"{pref}zero_frac",
                zero_frac,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
            )
        except Exception:
            # if self.log not available for some reason, fallback to logging
            logging.info(
                f"Grad metrics: total_norm={total_norm:.3e}, max_grad={max_grad:.3e}, min_nonzero={min_nonzero:.3e}, zero_frac={zero_frac:.3f}"
            )

        # For local debugging also emit a warning when problems detected
        if problem is not None:
            logging.warning(
                f"[grad-check] detected '{problem}' grads: total_norm={total_norm:.3e}, max_grad={max_grad:.3e}, zero_frac={zero_frac:.3f}"
            )

        return {
            "total_norm": total_norm,
            "max_grad": max_grad,
            "min_nonzero_grad": min_nonzero,
            "zero_frac": zero_frac,
            "problem": problem,
        }

    def on_after_backward(self) -> None:
        """
        Lightning hook called after backward() has been called.
        We call check_gradients here to detect exploding / vanishing gradients early.
        """
        # choose thresholds appropriate for your model / scale
        # Example: big networks might use high_threshold=1e2..1e3; low_threshold=1e-8..1e-6
        self.check_gradients(
            high_threshold=1e3,
            low_threshold=1e-6,
            clip_grad_norm=None,
            log_prefix="train",
        )

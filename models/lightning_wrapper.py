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

        # GECO parameters
        if hasattr(self.model, "log_lambda_paths"):
            self.log_lambda_paths = self.model.log_lambda_paths
            self.init_log_lambda_paths = self.model.log_lambda_paths
            self.target_loss = self.model.target_loss
            self.lambda_update_rate = self.model.lambda_update_rate
            self.ma_loss = torch.tensor(-1.0)  # Initialize moving average loss

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
        x, y = batch

        output = self(x)
        if isinstance(output, tuple):
            y_hat, num_paths, pgt0 = output

            standard_loss = self.loss(y_hat, y)
            self.log(f"{stage}/num_paths", num_paths)
            self.log(f"{stage}/pgt0", pgt0)
            self.log(f"{stage}/loss", standard_loss)

            if self.model.name == "sparse_unet":
                loss = standard_loss + 10**self.log_lambda_paths * num_paths
                self.log(f"{stage}/total_loss", loss)
                self.geco_update_lambda(standard_loss)
                self.log(f"{stage}/lambda_paths", self.log_lambda_paths)

            else:
                loss = standard_loss

        else:
            y_hat = output

            loss = self.loss(y_hat, y)

        if stage:
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

    def geco_update_lambda(self, loss: torch.Tensor):
        """
        Update the lambda parameter using the loss. Keeps a moving average of the loss.
        If the loss is below the target, the lambda is increased, otherwise it is decreased.
        This allows the model to tighten the sparsity of the attention patterns while maintaining the same prediction loss.
        This should be called after each forward pass.
        """

        # Internal parameters
        mooving_average_alpha = 0.99
        max_step = 1e-5
        max_lambda_paths = 1e6

        if self.ma_loss < 0:
            self.ma_loss = loss
        else:
            self.ma_loss = (
                mooving_average_alpha * self.ma_loss
                + (1 - mooving_average_alpha) * loss
            )
        self.ma_loss = self.ma_loss.detach()
        loss_diff = self.ma_loss - self.target_loss
        step = loss_diff * self.lambda_update_rate
        step = torch.clamp(step, max=max_step)
        self.log_lambda_paths = self.log_lambda_paths - step
        self.log_lambda_paths = torch.clamp(
            self.log_lambda_paths, min=self.init_log_lambda_paths, max=max_lambda_paths
        ).detach()

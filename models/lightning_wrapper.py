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
        normalisation_stats=None,
        debug=False,
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
        # Keep a list of the buffer names so we can create a friendly dict accessor later.
        self._norm_buffer_names = []

        if normalisation_stats is not None:
            for k, v in normalisation_stats.items():
                # convert numpy -> tensor (preserve shape/dtype). Use as_tensor to avoid copy if possible.
                t = torch.as_tensor(v)

                # ensure float32 for floats (optional, but common for stats)
                if t.is_floating_point():
                    t = t.to(torch.float32)

                # sanitize key for a valid attribute name if needed:
                name = f"norm_{k}"
                name = name.replace(".", "_").replace("-", "_")

                # register buffer so Lightning moves it automatically to device
                self.register_buffer(name, t, persistent=True)
                self._norm_buffer_names.append(name)

        self.loss = loss
        self.optimizer = optimizer
        if scheduler_cfg is None or scheduler_cfg == "None":
            self.scheduler_cfg = None
        else:
            self.scheduler_cfg = DictConfig(scheduler_cfg)
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
        # if len(batch) == 2:
        #     x, y = batch
        #     output = self(x)
        #     output = (
        #         self.unnormalize(output, self.normalisation_stats)
        #         if self.unnorm
        #         else output
        #     )
        # elif len(batch) == 4:
        #     xnh, xsh, ynh, ysh = batch
        #     output_nh = self(xnh)
        #     output_nh = (
        #         self.unnormalize(output_nh, self.normalisation_stats)
        #         if self.unnorm
        #         else output_nh
        #     )
        #     output_sh = self(xsh)
        #     output_sh = (
        #         self.unnormalize(output_sh, self.normalisation_stats)
        #         if self.unnorm
        #         else output_sh
        #     )
        #     output = (output_nh, output_sh)
        #     y = (ynh, ysh)

        # if hasattr(self.model, "step"):
        #     loss = self.model.step(output, y, self.log, self.loss, stage)

        # else:

        x, y = batch
        output = self(x)
        
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
        self.unnorm = False
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
        self.unnorm = True if self.normalisation_stats is not None else False
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
        self.unnorm = True if self.normalisation_stats is not None else False
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

    def unnormalize(self, data, normalisation_stats):
        """
        Unnormalizes the output data using the provided normalization statistics.
        Args:
            data (torch.Tensor): Normalized data.
            normalisation_stats (dict): Dictionary containing 'mean' and 'std' for unnormalization.
        Returns:
            torch.Tensor: Unnormalized data.
        """

        unnormalized_data = (
            data
            * (normalisation_stats["target_max"] - normalisation_stats["target_min"])
            + normalisation_stats["target_mean"]
        )
        return unnormalized_data

    @property
    def normalisation_stats(self):
        """Return a dict view of the registered buffers."""
        return {
            name[len("norm_") :]: getattr(self, name)
            for name in self._norm_buffer_names
        }

"""
PyTorch Lightning wrapper module for efficient reproduction of training and evaluation of models.

"""


import logging
import lightning as L
import torch 
from torch import nn
from models.model_utils.optimizers import select_optimizer
from models.model_utils.schedulers import select_scheduler

class LightningWrapper(L.LightningModule):
    def __init__(self, model: torch.nn.Module, loss=torch.nn.MSELoss(), optimizer='Adam', scheduler_cfg=None, lr=1e-3):
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
        self.scheduler_cfg = scheduler_cfg
        self.lr = lr
    


    def forward(self, x):
        """
        Forward pass through the model.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Model output.
        """
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        """
        Training step for a single batch.
        Args:
            batch (tuple): A tuple containing input data and target labels.
            batch_idx (int): Index of the batch.
        Returns:
            torch.Tensor: Loss value for the batch.
        """
        x, y = batch

        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('train/loss', loss)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step for a single batch.
        Args:
            batch (tuple): A tuple containing input data and target labels.
            batch_idx (int): Index of the batch.
        Returns:
            torch.Tensor: Loss value for the batch.
        """
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('val/loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        """
        Test step for a single batch.
        Args:
            batch (tuple): A tuple containing input data and target labels.
            batch_idx (int): Index of the batch.
        Returns:
            torch.Tensor: Loss value for the batch.
        """
        x, y = batch

        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('test/loss', loss)
        return loss

    def configure_optimizers(self):
        """
        Configures the optimizer for training.
        Returns:
            torch.optim.Optimizer: The optimizer to be used for training.
        """
        optimizer = select_optimizer(self.optimizer)(self.parameters(), lr=self.lr)
        logging.info(f"Using optimizer: {self.optimizer}")

        if self.scheduler_cfg is not None and self.scheduler_cfg != 'None':
            scheduler = select_scheduler(self.scheduler_cfg['name'], self.scheduler_cfg, optimizer)
            logging.info(f"Using scheduler: {self.scheduler_cfg['name']}")
            return [optimizer], [scheduler]
        
        else:
            return optimizer


"""
PyTorch Lightning wrapper module for efficient reproduction of training and evaluation of models.

"""


import lightning as L
import torch 
from torch import nn

class LightningWrapper(L.LightningModule):
    def __init__(self, model: torch.nn.Module, loss=torch.nn.MSELoss(), optimizer=torch.optim.Adam, lr=1e-3, multi_env=False, discrepancy_factor=None):
        """
        Initializes the LightningWrapper with a PyTorch model.
        Args:
            model (torch.nn.Module): The PyTorch model to be wrapped.
            loss (callable): Loss function to be used during training. Default is Mean Squared Error.
            optimizer (callable): Optimizer class to be used for training. Default is Adam.
            lr (float): Learning rate for the optimizer. Default is 1e-3.
            multi_env (bool): Flag indicating if the model is trained across multiple environments. Default is False.
        """
        super().__init__()

        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.lr = lr
        self.multi_env = multi_env
    


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
        optimizer = self.optimizer(self.parameters(), lr=1e-3)
        return optimizer
    

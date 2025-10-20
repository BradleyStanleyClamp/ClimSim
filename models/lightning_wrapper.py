"""
PyTorch Lightning wrapper module for efficient reproduction of training and evaluation of models.

"""


import lightning as L
import torch 
from torch import nn

class LightningWrapper(L.LightningModule):
    def __init__(self, model: torch.nn.Module, loss=torch.nn.MSELoss(), optimizer=torch.optim.Adam, scheduler=None, lr=1e-3):
        """
        Initializes the LightningWrapper with a PyTorch model.
        Args:
            model (torch.nn.Module): The PyTorch model to be wrapped.
            loss (callable): Loss function to be used during training. Default is Mean Squared Error.
            optimizer (callable): Optimizer class to be used for training. Default is Adam.
            lr (float): Learning rate for the optimizer. Default is 1e-3.
        """
        super().__init__()

        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
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
        optimizer = self.optimizer(self.parameters(), lr=self.lr)

        if self.scheduler == 'cyclic':
            scheduler = cyclic_optimizer(optimizer)
            return [optimizer], [scheduler]
        
        else:
            return optimizer

def cyclic_optimizer(optimizer, base_lr=2.5e-4, max_lr=2.5e-3, step_size=3285, scale_fn=lambda x: 1/(2.**(x-1))):
    """
    Creates a cyclic learning rate scheduler.

    """

    batch_size = 3072
    data_quantity = 10091520
    step_size = data_quantity // batch_size * 4  # 4 epochs
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=step_size, scale_fn=scale_fn)
    return scheduler
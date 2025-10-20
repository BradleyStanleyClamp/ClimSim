"""
Utility functions for selecting and managing models. The key component is we use a single config + model name for dealing with different models, e.g loading, training, optimizing different architectures.
"""

import lightning as L
import models
import torch


def select_model(model_name: str, model_params: dict, data_params: dict):
    """
    Selects and returns a model class based on the provided model name.
    Args:
        model_name (str): Name of the model to be selected.
        model_params (dict): Dictionary of parameters to initialize the model.
        data_params (dict): Dictionary of data-related parameters.
    Returns:
        model (nn.Module): An instance of the selected model class.
    Raises:
        ValueError: If the model name is not recognized.
    """
    if model_name == "mlp":
        mlp = models.mlp.MLP(hidden_dims=model_params.hidden_dims, input_dim=data_params.input_dim, output_dim=data_params.output_dim)
        return models.LightningWrapper(mlp)
    
    elif model_name == "yus_mlp":
        yus_mlp = models.yus_mlp.YusMLP(hidden_dims=model_params.hidden_dims, input_dim=data_params.input_dim, output_dim=data_params.output_dim)
        optimizer = select_optimizer(model_params.optimizer)
        return models.LightningWrapper(yus_mlp, optimizer=optimizer, lr=model_params.lr, scheduler=model_params.scheduler)

    else:
        raise ValueError(f"Model {model_name} not recognized.")
    
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
    else:
        raise ValueError(f"Optimizer {optimizer_name} not recognized.")
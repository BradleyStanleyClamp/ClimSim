"""
Utility functions for selecting and managing models. The key component is we use a single config + model name for dealing with different models, e.g loading, training, optimizing different architectures.
"""

import lightning as L
import models


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
        mlp = models.mlp.MLP(hidden_dim=model_params.hidden_dim, nhidden=model_params.nhidden, input_dim=data_params.input_dim, output_dim=data_params.output_dim)

        return models.LightningWrapper(mlp)
    
    else:
        raise ValueError(f"Model {model_name} not recognized.")
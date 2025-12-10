"""
Utility functions for selecting and managing models. The key component is we use a single config + model name for dealing with different models, e.g loading, training, optimizing different architectures.
"""

import lightning as L
import models
import torch


def select_base_model(
    model_name: str, model_params: dict, data_params: dict
) -> torch.nn.Module:
    """
    Selects and returns a base model (nn.Module) class based on the provided model name.
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
        mlp = models.mlp.MLP(
            hidden_dims=model_params.hidden_dims,
            input_dim=data_params.input_dim,
            output_dim=data_params.output_dim,
        )
        return mlp

    elif model_name == "yus_mlp":
        yus_mlp = models.yus_mlp.YusMLP(
            hidden_dims=model_params.hidden_dims,
            input_dim=data_params.input_dim,
            output_dim=data_params.output_dim,
        )
        return yus_mlp
    elif model_name == "climsim_unet":
        unet = models.ClimSimUNet()
        return unet
    elif model_name == "squeezeformer":

        squeezeformer = models.SqueezeFormer(
            in_dim=6,  # data_params.input_dim,
            embed_dim=model_params.embed_dim,
            out_dim=10,  # data_params.output_dim,
            head_dim=model_params.head_dim,
            num_heads=model_params.num_heads,
            num_encoder_blocks=model_params.num_encoder_blocks,
        )
        return squeezeformer
    elif model_name == "sparse_unet":
        sparse_unet = models.SparseUNet(
            in_channels=data_params.input_dim,
            out_channels=data_params.output_dim,
            tau=model_params.tau,
            lambda_paths=model_params.lambda_paths,
            lambda_update_rate=model_params.lambda_update_rate,
        )
        return sparse_unet
    else:
        raise ValueError(f"Model {model_name} not recognized.")


def select_model(
    model_name: str, model_params: dict, data_params: dict
) -> L.LightningModule:
    """
    Selects and returns a lightning wrapped model class based on the provided model name.
    Args:
        model_name (str): Name of the model to be selected.
        model_params (dict): Dictionary of parameters to initialize the model.
        data_params (dict): Dictionary of data-related parameters.
    Returns:
        model (L.LightningModule): An instance of the selected model class.
    """

    base_model = select_base_model(model_name, model_params, data_params)

    lightning_model = models.LightningWrapper(
        base_model,
        optimizer=model_params.optimizer,
        lr=model_params.lr,
        scheduler_cfg=model_params.scheduler,
    )

    return lightning_model


def load_model_from_checkpoint(
    checkpoint_path: str, model_name: str, model_params: dict, data_params: dict
):
    """
    Loads a model from a checkpoint file.
    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model_name (str): Name of the model to be loaded.
        model_params (dict): Dictionary of parameters to initialize the model.
        data_params (dict): Dictionary of data-related parameters.
    Returns:
        model (nn.Module): An instance of the loaded model class.
    """
    base_model = select_base_model(model_name, model_params, data_params)
    model = models.LightningWrapper.load_from_checkpoint(
        checkpoint_path,
        model=base_model,
        optimizer=model_params.optimizer,
        lr=model_params.lr,
        scheduler=model_params.scheduler,
    )
    return model

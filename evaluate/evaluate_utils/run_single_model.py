"""
Script to extract model from logs setup and run evaluation on dataset.
"""
import os
import numpy as np
import lightning as L
from omegaconf import DictConfig
import torch
from tqdm import tqdm
import yaml
import models
from pathlib import Path

def run_single_model_from_log_results(model_name: str, training_log_path: str, dataset_cfg, dataloader) -> np.ndarray:
   """
   Loads model using information from a log folder path and runs evaluation on the given dataset.
   
    Args:
        model_name (str): Name of the model.
        training_log_path (str): Path to the log folder for the trained model.
        data_cfg (DictConfig): Configuration for the dataset, needed for model loading containing
            input_dim, output_dim, etc.
        dataloader (DataLoader): DataLoader for the dataset to evaluate on.
    Returns:
        np.ndarray: Predictions from the model on the dataset.
   """

   model = get_model_from_config(model_name, training_log_path, dataset_cfg)
   output = evaluate_model_on_dataset(model, dataloader)
   return output

def get_model_from_config(model_name: str, training_log_path: str, dataset_cfg: DictConfig) -> L.LightningModule:
    """
   Loads model from log folder path using models package 

    Args:
            model_name (str): Name of the model.
            training_log_path (str): Path to the log folder for the trained model.
            data_cfg (DictConfig): Configuration for the dataset, needed for model loading containing
            input_dim, output_dim, etc.
    """

    # Check if path is valid
    base_dir = Path(__file__).resolve().parents[2]
    full_path = Path(os.path.join(base_dir, training_log_path, model_name))
    if not full_path.exists():
        raise FileNotFoundError(f"Model log path {full_path} does not exist.")


    # Find checkpoint file and run_config file 
    # Find checkpoint file and run_config file
    ckpt_files = list(full_path.rglob("*.ckpt"))
    if len(ckpt_files) == 0:
        raise FileNotFoundError(f"No .ckpt files found in {full_path}")
    if len(ckpt_files) > 1:
        raise FileExistsError(f"Expected exactly one .ckpt file in {full_path}, found {len(ckpt_files)}: {[str(p) for p in ckpt_files]}")
    checkpoint_path = ckpt_files[0]

    # Try to find a run config YAML (optional)
    run_config_files = list(full_path.rglob("*.yml")) + list(full_path.rglob("*.yaml"))
    run_config_path = run_config_files[0] if run_config_files else None

 

    # Get config containing model and run parameters
    with open(run_config_path, "r") as f:
        run_config_dict = yaml.safe_load(f)
    run_config = DictConfig(run_config_dict)

    

    # Load model from checkpoint
    model = models.load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        model_name=model_name,
        model_params=run_config,
        data_params=dataset_cfg,
    )

    return model

def evaluate_model_on_dataset(model: L.LightningModule, dataloader) -> np.ndarray:
    """
    Evaluates given model on given dataset and returns predictions as numpy array.

    Args:
        model (L.LightningModule): Trained model to be evaluated.
        dataloader (DataLoader): DataLoader for the dataset to evaluate on.
    Returns:
        np.ndarray: Predictions from the model on the dataset.
    """
    model.eval()
    model.freeze()
    outputs_list = []
    for batch in tqdm(dataloader):
        input, _ = batch
        outputs = model(input)
        outputs_list.append(outputs)
    
    outputs_all = torch.cat(outputs_list, dim=0)

    return outputs_all.detach().numpy()
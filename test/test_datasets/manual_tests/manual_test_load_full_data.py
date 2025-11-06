"""
Script that trains the models used for sanity testing process. These models are:
- Constant prediction model: always predicts the mean of the training set
- Multiple linear regression model: linear regression from inputs to outputs
"""

import warnings
import netCDF4 # Another weird import issue that is only triggered if netCDF4 imported after wandb 
with (
    warnings.catch_warnings()
):  # To catch annoying pydantic x wandb warning - looks like it should be adressed soon: https://github.com/wandb/wandb/issues/10662
    warnings.filterwarnings("ignore")
    import wandb
import logging
import string
from omegaconf import DictConfig
import hydra
from train import seed_everything
import torch
import data_preparation
import models
import yaml
import lightning as L
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from pathlib import Path
import train
import numpy as np
from evaluate.evaluate_utils import save_evaluation_results_to_json

@hydra.main(version_base=None, config_path="../../../config", config_name="train_general")
def main(cfg: DictConfig):
       # Seeding everything
    train.seed_everything(cfg.project.seed)
    
    torch.set_float32_matmul_precision('medium')

    # trainset, valset, testset = data_preparation.get_all_datasets(cfg.dataset, cfg.testing.dataset_testing_type)

    logging.info(f'model: {cfg.model.name}')
    trainset = data_preparation.get_dataset(
        cfg.dataset, "train", cfg.testing.dataset_testing_type, model=cfg.model.name
    )

    x, y = trainset[0]

    logging.info(f"Input shape: {x.shape}, Target shape: {y.shape}")
    logging.info(f'dataset lengths: train {len(trainset)}')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    main()    
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
import time

@hydra.main(version_base=None, config_path="../../../config", config_name="train_general")
def main(cfg: DictConfig):
       # Seeding everything
    train.seed_everything(cfg.project.seed)
    
    torch.set_float32_matmul_precision('medium')

    trainset = data_preparation.ClimSimFromRawDataset(mode="train", dataset_testing_type=cfg.testing.dataset_testing_type, dataset_cfg=cfg.dataset, model=cfg.model.name)
    logging.info(f'dataset lengths: train {len(trainset)}')

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    main()    
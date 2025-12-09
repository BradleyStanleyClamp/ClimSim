"""
Script that trains the models used for sanity testing process. These models are:
- Constant prediction model: always predicts the mean of the training set
- Multiple linear regression model: linear regression from inputs to outputs
"""

import warnings
import netCDF4  # Another weird import issue that is only triggered if netCDF4 imported after wandb

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


import dask
from dask.distributed import Client, LocalCluster
import xarray as xr


@hydra.main(
    version_base=None, config_path="../../../config", config_name="evaluate_data_groups"
)
def main(cfg: DictConfig):
    logging.info("Remember to set dataset to climsim_from_npy")
    # Seeding everything
    train.seed_everything(cfg.project.seed)

    torch.set_float32_matmul_precision("medium")

    mode = "train"
    start_time = time.time()
    trainset = data_preparation.ClimSimNpyDataset(
        dataset_cfg=cfg.dataset,
        group_idx=cfg.target_group,
        normalisation_stats=None,
    )
    logging.info(f"Trainset loading time: {time.time() - start_time} seconds")

    logging.info(f"Trainset length: {len(trainset)}")
    logging.info(trainset.input.shape)
    logging.info(trainset.target.shape)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    main()

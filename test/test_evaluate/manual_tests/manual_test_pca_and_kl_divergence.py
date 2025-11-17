import torch
import evaluate

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


@hydra.main(
    version_base=None, config_path="../../../config", config_name="evaluate_data_groups"
)
def main(cfg: DictConfig):
    # Seeding everything
    train.seed_everything(cfg.project.seed)

    torch.set_float32_matmul_precision("medium")
    cfg.dataset.group_method = "group_by_year"
    cfg.dataset.group_by_year.target_group = 0

    # trainset, valset, testset = data_preparation.get_all_datasets(cfg.dataset, cfg.testing.dataset_testing_type)
    trainset = data_preparation.get_dataset(
        cfg.dataset, "train", cfg.testing.dataset_testing_type
    )
    testset = data_preparation.get_dataset(
        cfg.dataset, "test", cfg.testing.dataset_testing_type
    )

    X = trainset.input
    Y = testset.input

    n_components = 3
    start_time = time.perf_counter()
    X_proj, Y_proj = evaluate.pca_gpu(X, Y, n_components)
    elapsed = time.perf_counter() - start_time
    logging.info(f"PCA GPU projection time: {elapsed:0.4f} seconds")
    logging.info(f"X_proj shape: {X_proj.shape}, Y_proj shape: {Y_proj.shape}")

    start_time = time.perf_counter()
    est_kl = evaluate.KLdivergence(X_proj.numpy(), Y_proj.numpy())
    elapsed_kl = time.perf_counter() - start_time
    logging.info(f"Estimated KL divergence: {est_kl:0.4f}")
    logging.info(f"KL divergence computation time: {elapsed_kl:0.4f} seconds")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    main()

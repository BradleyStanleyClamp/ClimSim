"""
High level script for running hyperparameter sweeps, configured using config/train_general.yaml
"""

import json
import warnings


import hydra
from omegaconf import DictConfig
import omegaconf

import yaml
import netCDF4  # Another weird import issue that is only triggered if netCDF4 imported after wandb

with (
    warnings.catch_warnings()
):  # To catch annoying pydantic x wandb warning - looks like it should be adressed soon: https://github.com/wandb/wandb/issues/10662
    warnings.filterwarnings("ignore")
    import wandb

import lightning as L
import numpy as np
import xarray as xr
import logging
import os
import train
import models
import torch
import data_preparation


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="train_general")
def main(cfg: DictConfig):

    # Seeding everything
    train.seed_everything(cfg.project.seed)

    torch.set_float32_matmul_precision("medium")

    # Wandb login
    if cfg.wandb.wkey.wkey is not None:
        wandb.login(key=cfg.wandb.wkey.wkey)
    else:
        raise ValueError("Error: fill wkey.yaml file with API key")

    logging.info("Setup complete, starting training")
    datasets = data_preparation.get_all_datasets(
        cfg.dataset, cfg.testing.dataset_testing_type, model=cfg.model.name
    )

    sweep_config_dict = omegaconf.OmegaConf.to_container(
        cfg.model.sweep_configuration, resolve=True, throw_on_missing=True
    )

    sweep_id = wandb.sweep(
        sweep=sweep_config_dict,
        project=f"{cfg.project.project}_{cfg.project.task}",
    )
    run_counter = 0

    def train_sweep():
        nonlocal run_counter
        train.standard_training_from_cfg(
            cfg,
            datasets,
            sweep_config_dict,
            f"{cfg.multirun_dir_name}_{cfg.project.timestamp}_{run_counter}",
            enable_checkpointing=False,
        )
        run_counter += 1

    sweep_agent = wandb.agent(
        sweep_id,
        function=train_sweep,
        count=cfg.sweep,
    )

    # Get best run parameters
    api = wandb.Api()
    sweep = api.sweep(sweep_id)
    best_run = sweep.best_run()
    best_config_name = os.path.join(f"{cfg.multirun_dir_name}_best_config.yaml")
    logger.info(f"Saving best config as {best_config_name}")
    # epoch_run_duration = best_run.summary.epoch
    best_run_cfg = json.loads(best_run.config)
    cfg.model.single_run_configuration = best_run_cfg
    # best_run_cfg["epochs"] = epoch_run_duration

    with open(best_config_name, "w") as f:
        omegaconf.OmegaConf.save(best_run_cfg, f.name)
    # logger.info(f"Best run summary: {best_run.summary}")
    logger.info(f"training and saving best model")

    # close the sweep
    wandb.finish()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    main()

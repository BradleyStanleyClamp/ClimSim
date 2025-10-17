"""
High level training script that is configured using config/train_general.yaml
"""

import warnings


import hydra
from omegaconf import DictConfig
import omegaconf

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

import data_preparation


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="train_general")
def main(cfg: DictConfig):

    # Seeding everything
    train.seed_everything(cfg.project.seed)

    # TODO: plotting init

    # Wandb login
    if cfg.wandb.wkey.wkey is not None:
        wandb.login(key=cfg.wandb.wkey.wkey)
    else:
        raise ValueError("Error: fill wkey.yaml file with API key")

    wandb_config = omegaconf.OmegaConf.to_container(
        cfg.model.single_run_configuration, resolve=True, throw_on_missing=True
    )

    logging.info("Setup complete, starting training")

    model, test_result = train.standard_training_from_cfg(
        cfg,
        wandb_config,
        f"{cfg.project.name}_{cfg.project.timestamp}",
        cfg.model.single_run_configuration,
        enable_checkpointing=False,
    )

    # Save model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    main()

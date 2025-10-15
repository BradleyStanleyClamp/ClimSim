"""
High level training script that is configured using config/train_general.yaml
"""
import warnings


import hydra
from omegaconf import DictConfig

with warnings.catch_warnings(): # To catch annoying pydantic x wandb warning - looks like it should be adressed soon: https://github.com/wandb/wandb/issues/10662
    warnings.filterwarnings("ignore")
    import wandb

import lightning as L
import numpy as np
import xarray as xr
import logging
import os
import train_utils
import models


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="train_general")
def main(cfg: DictConfig):

    train_utils.seed_everything(cfg.project.seed)


    # raw overrides lists
    test_run = train_utils.is_test_run()


    # TODO: plotting init 

    if cfg.wandb.wkey.wkey is not None:
        wandb.login(key=cfg.wandb.wkey.wkey)
    else:
        raise ValueError("Error: fill wkey.yaml file with API key")

    model_name = cfg.model_name
    model_cfg = models.model_selection.get_model_config(cfg, model_name)

    logging.info(f"working with model: {model_name}, for a test run: {test_run}")


    

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    main()

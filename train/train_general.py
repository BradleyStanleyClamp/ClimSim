"""
High level training script that is configured using config/train_general.yaml
"""

import hydra
from omegaconf import DictConfig
import wandb
import lightning as L
from lightning.pytorch import seed_everything
import numpy as np
import xarray as xr
import logging
import os
import warnings


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="train_general")
def main(cfg: DictConfig):

    logging.info("test 1")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)


    main()

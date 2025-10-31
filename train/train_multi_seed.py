"""
High level training script that is configured using config/train_general.yaml
"""

import json
import warnings


import hydra
from omegaconf import DictConfig
import omegaconf

import yaml 
import netCDF4 # Another weird import issue that is only triggered if netCDF4 imported after wandb 
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
    
    torch.set_float32_matmul_precision('medium')

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

    datasets = data_preparation.get_all_datasets(cfg.dataset, cfg.testing.dataset_testing_type)

    for seed in cfg.train_seeds:
        train.seed_everything(seed)

        test_result, run_cfg = train.standard_training_from_cfg(
            cfg,
            datasets,
            wandb_config,
            f"{cfg.multirun_dir_name}_{cfg.project.timestamp}",
            enable_checkpointing=False,
            seed=seed,
            )

        logging.info(f"Completed training for seed {seed}")

        # write the run config for this seed
        with open(f"run_config_{seed}.yaml", "w") as f:
            yaml.safe_dump(run_cfg.as_dict(), f)

        # aggregate test results into a single JSON file
        results_file = "test_results.json"
        existing = None
        try:
            if os.path.exists(results_file):
                with open(results_file, "r") as rf:
                    existing = json.load(rf)
        except Exception:
            # if loading fails, we'll overwrite with a fresh structure below
            existing = None

        # normalize existing content into a dict or list and merge
        if isinstance(existing, list):
            # append a mapping for this seed
            existing.append({str(seed): test_result})
            new_content = existing
        elif isinstance(existing, dict):
            # add/replace the seed entry
            existing[str(seed)] = test_result
            new_content = existing
        else:
            # no usable existing content -> create dict keyed by seed
            new_content = {str(seed): test_result}

        with open(results_file, 'w') as f:
            json.dump(new_content, f, indent=4, default=str)
    


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    main()

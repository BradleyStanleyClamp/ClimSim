"""
Script to load .nc files, process them into datasets, and save as .npy files for faster loading later.

"""

import json
import warnings
from dask.distributed import Client


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


@hydra.main(
    version_base=None, config_path="../config", config_name="process_and_save_data"
)
def main(cfg: DictConfig):

    # Seeding everything
    train.seed_everything(cfg.project.seed)

    torch.set_float32_matmul_precision("medium")
    num_workers = cfg.dataset.general_dataset_config.num_workers
    climsim_from_raw = data_preparation.ClimSimFromRawDataset(
        mode="all",
        dataset_testing_type=cfg.testing.dataset_testing_type,
        dataset_cfg=cfg.dataset,
        model=None,
        normalisation_stats=None,
        unit_test_specific_methods=True,
        num_workers=num_workers,
    )

    target_years, target_months = climsim_from_raw._select_target_years_months(
        mode="train", dataset_cfg=cfg.dataset
    )

    input_filenames, target_filenames = climsim_from_raw._get_dataset_filenames(
        cfg.dataset.base_folder_path,
        target_years,
        target_months,
    )
    logging.info(f"Total files found: {len(input_filenames)}")

    # manual sampling to reduce data size for testing
    start_index = 0
    end_index = len(input_filenames)
    sample_rate = cfg.sample_rate
    sampled_input_filenames = sorted(input_filenames)[start_index:end_index:sample_rate]
    sampled_target_filenames = sorted(target_filenames)[
        start_index:end_index:sample_rate
    ]
    assert data_preparation.check_matching_files(
        sampled_input_filenames, sampled_target_filenames
    )
    logging.info(
        f"Files after sampling ({cfg.sample_rate}): {len(sampled_input_filenames)}"
    )

    if num_workers > 1:
        client = Client(n_workers=num_workers, threads_per_worker=1, memory_limit="8GB")
        parallel = True
        logging.info(f"Using Dask with {num_workers} workers for parallel processing.")
    else:
        parallel = False
        logging.info("Using single worker for processing.")

    input_ds, target_ds = climsim_from_raw._combine_datasets(
        sampled_input_filenames,
        sampled_target_filenames,
        v1_inputs=cfg.dataset.v1_inputs,
        v1_targets=cfg.dataset.v1_targets,
        parallel=parallel,
    )

    input_ds, target_ds = climsim_from_raw._level_selection(
        input_ds,
        target_ds,
        levels=cfg.dataset.levels,
    )
    logging.info(f"Selected {input_ds.sizes['lev']} levels from dataset.")

    input, target = climsim_from_raw._prepare_data(
        'no_stack',
        input_ds,
        target_ds,
    )

    save_folder = cfg.save_processed_data_path
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    np.save(os.path.join(save_folder, "input.npy"), input)
    np.save(os.path.join(save_folder, "target.npy"), target)
    logging.info(f"Processed data saved to {save_folder}")



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    main()

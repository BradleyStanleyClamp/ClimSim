"""
Working script for ongoing analysis of model performance.
"""

import json
from typing import Tuple
from omegaconf import DictConfig, OmegaConf
import logging
import hydra
import os
import re
import torch
import yaml
import numpy as np
import numbers
from tqdm import tqdm
import lightning as L
import train
import data_preparation
import models
import evaluate
import plotting


@hydra.main(version_base=None, config_path="../../config", config_name="train_general")
def main(cfg: DictConfig):
    train.seed_everything(cfg.project.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    path_to_model = "/home/users/bradlesc/projects/ClimSim/logs/p2.1.4/7/squeeze_former_group_MAM_2025-12-29-09-41-08/squeezeformer_group_MAM/squeezeformer_None_2025-12-29-09-41-08.ckpt"
    # path_to_model = "/gws/nopw/j04/iecdt/bstanleyclamp/checkpoints/p2.1.3/11/squeezeformer_from_npy_multiseed_2025-12-10-11-35-13/squeezeformer_group_SON/squeezeformer_0_2025-12-10-11-35-13.ckpt"
    model = models.load_model_from_checkpoint(
        path_to_model, cfg.model.name, cfg.model.single_run_configuration, cfg.dataset
    )
    model.to(device)

    # Datasets
    batch_size = 4096


    # SON normalisation stats
    # cfg.dataset.group_by_months.target_group = "SON"
    # son_trainset = data_preparation.ClimSimNpyDataset(
    #     cfg.dataset,
    #     cfg.testing.dataset_testing_type,
    #     "train",
    #     normalisation_stats=None,
    #     model=cfg.model.name,
    #     seed=0,
    # )
    # normalisation_stats = son_trainset.normalisation_stats
    
    # logging.info("---")


    # MAM 'in distribution' training set
    cfg.dataset.group_by_months.target_group = "MAM"
    mam_trainset = data_preparation.ClimSimNpyDataset(
        cfg.dataset,
        cfg.testing.dataset_testing_type,
        "train",
        normalisation_stats=None,
        model=cfg.model.name,
        seed=0,
    )
    mam_trainloader = torch.utils.data.DataLoader(
        mam_trainset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(cfg.dataset.general_dataset_config.num_workers),
        persistent_workers=cfg.dataset.general_dataset_config.persistent_workers,
        prefetch_factor=cfg.dataset.general_dataset_config.prefetch_factor,
        pin_memory=True,
    )
    normalisation_stats = mam_trainset.normalisation_stats
    logging.info("---")

    # MAM 'in distribution' test set
    cfg.dataset.group_by_months.test_group = {"MAM": []}
    mam_testset = data_preparation.ClimSimNpyDataset(
        cfg.dataset,
        cfg.testing.dataset_testing_type,
        "test",
        normalisation_stats=normalisation_stats,
        model=cfg.model.name,
        seed=0,
    )
    mam_testloader = torch.utils.data.DataLoader(
        mam_testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(cfg.dataset.general_dataset_config.num_workers),
        persistent_workers=cfg.dataset.general_dataset_config.persistent_workers,
        prefetch_factor=cfg.dataset.general_dataset_config.prefetch_factor,
        pin_memory=True,
    )
    logging.info("---")

    # JJA 'out of distribution'
    cfg.dataset.group_by_months.test_group = {"JJA": []}
    jja_testset = data_preparation.ClimSimNpyDataset(
        cfg.dataset,
        cfg.testing.dataset_testing_type,
        "test",
        normalisation_stats=normalisation_stats,
        model=cfg.model.name,
        seed=0,
    )
    jja_testloader = torch.utils.data.DataLoader(
        jja_testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(cfg.dataset.general_dataset_config.num_workers),
        persistent_workers=cfg.dataset.general_dataset_config.persistent_workers,
        prefetch_factor=cfg.dataset.general_dataset_config.prefetch_factor,
        pin_memory=True,
    )


    # SON 'out of distribution'
    cfg.dataset.group_by_months.test_group = {"SON": []}
    son_testset = data_preparation.ClimSimNpyDataset(
        cfg.dataset,
        cfg.testing.dataset_testing_type,
        "test",
        normalisation_stats=normalisation_stats,
        model=cfg.model.name,
        seed=0,
    )
    son_testloader = torch.utils.data.DataLoader(
        son_testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(cfg.dataset.general_dataset_config.num_workers),
        persistent_workers=cfg.dataset.general_dataset_config.persistent_workers,
        prefetch_factor=cfg.dataset.general_dataset_config.prefetch_factor,
        pin_memory=True,
    )

    dataloaders = {
        # "MAM_train": mam_trainloader,
        "MAM_test": mam_testloader,
        "JJA_test": jja_testloader,
        "SON_test": son_testloader,
    }

    # trainer = L.Trainer(
    #     max_epochs=cfg.testing.epochs,
    #     accelerator="auto",
    #     devices="auto",
    #     enable_checkpointing=False,
    #     log_every_n_steps=5,
    # )

    # for key, loader in dataloaders.items():
    #     logging.info(f"{key} has {len(loader.dataset)} samples.")
    #     test_results = trainer.test(model, dataloaders=loader)
    #     logging.info(f"{key} test results: {test_results}")

    model.eval()
    save_location = '/work/scratch-pw5/bradlesc/climsim/temp/p2.1.4.9_analysis'
    model_name = 'train_group_MAM'
    save_location = os.path.join(save_location, model_name)
    os.makedirs(save_location, exist_ok=True)
    for key, loader in dataloaders.items():
        with torch.no_grad():
            all_outputs = []
            all_targets = []
            all_inputs = []
            for batch in loader:
                inputs, targets = batch
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)

                all_outputs.append(outputs.cpu())
                all_targets.append(targets.cpu())
                all_inputs.append(inputs.cpu())

            all_outputs = torch.cat(all_outputs, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            all_inputs = torch.cat(all_inputs, dim=0)
            np.save(os.path.join(save_location, f"{key}_inputs.npy"), all_inputs.numpy())
            np.save(os.path.join(save_location, f"{key}_outputs.npy"), all_outputs.numpy())
            np.save(os.path.join(save_location, f"{key}_targets.npy"), all_targets.numpy())

            logging.info(f"Saved outputs and targets for {key} to {save_location}")

            mse = torch.mean((all_outputs - all_targets) ** 2).item()
            logging.info(f"{key} MSE: {mse}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    torch.backends.cudnn.benchmark = True

    main()

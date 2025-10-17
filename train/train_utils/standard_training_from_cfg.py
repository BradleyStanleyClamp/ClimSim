"""
Script that wraps all model loading, data loading and training/testing in a single file, to be used for standard training from a config file.
"""

import logging
from typing import Dict
import wandb
from lightning.pytorch.loggers import WandbLogger
import lightning as L
from omegaconf import DictConfig
import data_preparation
import models
from typing import Dict

def standard_training_from_cfg(
    cfg: DictConfig, wandb_cfg: Dict, run_name: str, model_cfg: Dict, enable_checkpointing: bool
):
    """
    Function that peforms standard pytorch-lightning training from config files. Its two main usecases are for single runs and sweeps.

    Args:
        cfg: (DictConfig) main config file, e.g. config/train_general.yaml
        wandb_cfg: (DictConfig) wandb config file, only used for single runs, and is just the single_run_configuration of the model
        run_name: (str) name of the run, to be used in wandb logging
        model_cfg: (Dict) model config file predetermined i.e from sweep or single run
        enable_checkpointing: (bool) whether to enable checkpointing in the trainer. Should be False for sweeps, True for single runs.
    """

    with wandb.init(
        project=f"{cfg.project.project}_{cfg.project.task}",
        mode=cfg.testing.wandb_mode,
        name=run_name,
        config=wandb_cfg,
    ) as run:
        # Setting up wandb logger
        wandb_logger = WandbLogger(
            project=f"{cfg.project.project}_{cfg.project.task}",
            name=f"{cfg.project.name}_{cfg.project.timestamp}",
        )
        logging.info(f"Running training with run name: {run_name}")

        # Get data
        trainloader, valloader, testloader = data_preparation.get_all_dataloaders(
            cfg.dataset, run.config.batch_size, cfg.testing.dataset_testing_type
        )
        logging.info("Data loaders obtained")

        # Load model
        model = models.select_model(cfg.model.name, run.config, cfg.dataset)
        logging.info("Model loaded")

        # Get call backs
        callbacks=[]

        logging.info("Initializing trainer")
        # Initialize trainer
        trainer = L.Trainer(
                max_epochs=cfg.testing.epochs, 
                accelerator="auto",
                devices="auto",
                logger=wandb_logger,
                enable_checkpointing=enable_checkpointing,
                callbacks=callbacks,
            )

        logging.info("Starting training")
        # Train model
        trainer.fit(
                model,
                train_dataloaders=trainloader,
                val_dataloaders=valloader,
            )

        # Test model
        test_results = trainer.test(model, dataloaders=testloader)

        return model, test_results

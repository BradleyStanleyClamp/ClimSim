"""
Script that wraps all model loading, data loading and training/testing in a single file, to be used for standard training from a config file.
"""

import logging
from typing import Dict, Tuple
import wandb
from lightning.pytorch.loggers import WandbLogger
import lightning as L
from omegaconf import DictConfig
import data_preparation
import models
from torch.utils.data import Dataset

from pytorch_lightning.profilers import SimpleProfiler as pl_profiler
import os

def standard_training_from_cfg(
    cfg: DictConfig,
    datasets: Tuple[Dataset, Dataset, Dataset],
    wandb_cfg: Dict,
    run_name: str,
    enable_checkpointing: bool,
    seed: int = None,
) -> Tuple[Dict, Dict]:
    """
    Function that peforms standard pytorch-lightning training from config files. Its two main usecases are for single runs and sweeps.

    Args:
        cfg: (DictConfig) main config file, e.g. config/train_general.yaml
        datasets: (Tuple[Dataset, Dataset, Dataset]) tuple containing the train, val and test datasets
        wandb_cfg: (DictConfig) wandb config file, only used for single runs, and is just the single_run_configuration of the model
        run_name: (str) name of the run, to be used in wandb logging
        enable_checkpointing: (bool) whether to enable checkpointing in the trainer. Should be False for sweeps, True for single runs.
        seed: (int) random seed to use for the run. If None, no seeding is done.

    Returns:
        test_results: (Dict) results from testing the model on the test set
        run.config: (Dict) configuration used for the run
    """

    with wandb.init(
        project=f"{cfg.project.project}_{cfg.project.task}",
        mode=cfg.testing.wandb_mode,
        name=run_name,
        config=wandb_cfg,
    ) as run:
        # Setting up wandb logger
        wandb_logger = WandbLogger(experiment=run)
        logging.info(f"Running training with run name: {run_name}")

        # Get data
        trainloader, valloader, testloader = data_preparation.get_all_dataloaders(
            cfg.dataset,
            run.config.batch_size,
            cfg.testing.dataset_testing_type,
            datasets,
            model=cfg.model.name,
        )
        logging.info("Data loaders obtained")

        # Load model
        sample_input, sample_output = next(iter(trainloader))
        if cfg.model.name == "squeezeformer":
            cfg.dataset.input_dim, cfg.dataset.output_dim = (
                6,
                10,
            )
        else:
            cfg.dataset.input_dim, cfg.dataset.output_dim = (
                sample_input.shape[-1],
                sample_output.shape[-1],
            )
        logging.info(
            f"Input dim: {cfg.dataset.input_dim}, Output dim: {cfg.dataset.output_dim}"
        )
        model = models.select_model(cfg.model.name, run.config, cfg.dataset)
        logging.info(f"Model {cfg.model.name} loaded")

        # Get call backs
        callbacks = []

        if cfg.testing.time_analysis:
            profiler = pl_profiler(filename="profiler_output.txt")
        else:
            profiler = None

        # Initialize trainer
        trainer = L.Trainer(
            max_epochs=cfg.testing.epochs,
            accelerator="auto",
            devices="auto",
            logger=wandb_logger,
            enable_checkpointing=enable_checkpointing,
            callbacks=callbacks,
            log_every_n_steps=5,
            profiler=profiler,
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
        logging.info(f"Testing complete: {test_results}")

        # Optional Saving the model if single run
        if not cfg.sweep:
            model_folder_path = cfg.testing.checkpoint_folder_path
            model_save_path = f"{cfg.model.name}_{seed}_{cfg.project.timestamp}.ckpt"
            full_path = os.path.join(model_folder_path, model_save_path)
            trainer.save_checkpoint(full_path)
            logging.info(f"Model saved at: {full_path}")


        return test_results, run.config

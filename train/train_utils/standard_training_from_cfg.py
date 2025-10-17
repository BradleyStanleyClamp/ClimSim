"""
Script that wraps all model loading, data loading and training/testing in a single file, to be used for standard training from a config file. 
"""

from typing import Dict
import wandb
from lightning.pytorch.loggers import WandbLogger
import lightning as L
from omegaconf import DictConfig
import data_preparation


def standard_training_from_cfg(cfg: DictConfig, wandb_cfg: Dict, run_name: str, enable_checkpointing: bool):
    """
    Function that peforms standard pytorch-lightning training from config files. Its two main usecases are for single runs and sweeps.

    Args:
        cfg: (DictConfig) main config file, e.g. config/train_general.yaml
        wandb_cfg: (DictConfig) wandb config file, only used for single runs, and is just the single_run_configuration of the model
        run_name: (str) name of the run, to be used in wandb logging
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



    # # Get data 
    # trainloader = ...
    # valloader = ...
    # testloader = ...

    # # Load model 
    # model = ...

    # # Get call backs
    # callbacks=[]

    # # Initialize trainer
    # trainer = L.Trainer(
    #         max_epochs=cfg.project.max_epochs,
    #         accelerator="auto",
    #         devices="auto",
    #         logger=wandb_logger,
    #         enable_checkpointing=enable_checkpointing,
    #         callbacks=callbacks,

    #     )
    
    # # Train model
    # trainer.fit(
    #         model,
    #         train_dataloaders=trainloader,
    #         val_dataloaders=valloader,
    #     )
    
    # # Test model
    # test_results = trainer.test(model, dataloaders=testloader)

    # return model, test_results
    

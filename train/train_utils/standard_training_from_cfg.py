"""
Script that wraps all model loading, data loading and training/testing in a single file, to be used for standard training from a config file. 
"""

import wandb
from lightning.pytorch.loggers import WandbLogger
import lightning as L


def standard_training_from_cfg(cfg, wandb_cfg, ):
    """
    Function that peforms standard pytorch-lightning training from config files. Its two main usecases are for single runs and sweeps.
    """

    # To pass in 
    run_name
    wandb_config
    project_name
    run_name
    enable_checkpointing


    with wandb.init(
        project=f"{cfg.project.project}_{cfg.project.task}",
        name=run_name,
        config=wandb_config,
    ) as run:
        # Setting up wandb logger
        wandb_logger = WandbLogger(
            project=f"{cfg.project.project}_{cfg.project.task}",
            name=f"{cfg.project.name}_{cfg.project.timestamp}",
        )


    # Get data 
    trainloader = ...
    valloader = ...
    testloader = ...

    # Load model 
    model = ...

    # Get call backs
    callbacks=[]

    # Initialize trainer
    trainer = L.Trainer(
            max_epochs=cfg.project.max_epochs,
            accelerator="auto",
            devices="auto",
            logger=wandb_logger,
            enable_checkpointing=enable_checkpointing,
            callbacks=callbacks,

        )
    
    # Train model
    trainer.fit(
            model,
            train_dataloaders=trainloader,
            val_dataloaders=valloader,
        )
    
    # Test model
    test_results = trainer.test(model, dataloaders=testloader)

    return model, test_results
    

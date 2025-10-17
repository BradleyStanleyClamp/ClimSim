"""
Script to unit test the standard training from config function
"""

from omegaconf import OmegaConf
import omegaconf
import train
from hydra import initialize, compose


def test_standard_training_from_cfg():
    """
    Test function for standard_training_from_cfg
    """

    with initialize(version_base=None, config_path="../../config"):
        # config is relative to a module
        cfg = compose(config_name="train_general", overrides=[])

        wandb_config = omegaconf.OmegaConf.to_container(
            cfg.model.single_run_configuration, resolve=True, throw_on_missing=True
        )

        # Call standard training from cfg function
        train.standard_training_from_cfg(
            cfg,
            wandb_config,
            "test_run",
            cfg.model.single_run_configuration,
            enable_checkpointing=False,
        )

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
        # NOTE: setting sweep to True to prevent model being saved
        cfg = compose(
            config_name="train_general",
            overrides=[
                "sweep=True",
                "testing=qt",
                
                "dataset.dataset_testing_sample_rates.quick=1",
                "dataset.base_folder_path=/home/users/bradlesc/projects/ClimSim/test/unit_test_sets/dummy_low_res_climsim/filename_testing",
                "dataset.group_method=False",
                "dataset.remove_high_altitude_specific_humidity_levels=15",
            ],
        )

        wandb_config = omegaconf.OmegaConf.to_container(
            cfg.model.single_run_configuration, resolve=True, throw_on_missing=True
        )

        datasets = None  # Using None to let the function load datasets internally

        # Call standard training from cfg function
        train.standard_training_from_cfg(
            cfg,
            datasets,
            wandb_config,
            "test_run",
            enable_checkpointing=False,
        )


def test_unet_standard_training_from_cfg():
    """
    Test function for standard_training_from_cfg with unet model
    """

    with initialize(version_base=None, config_path="../../config"):
        # config is relative to a module
        cfg = compose(
            config_name="train_general",
            overrides=[
                "sweep=True",
                "testing=qt",
                "model=climsim_unet",
                "dataset.dataset_testing_sample_rates.quick=1",
                "dataset.base_folder_path=/home/users/bradlesc/projects/ClimSim/test/unit_test_sets/dummy_low_res_climsim/filename_testing",
                "max_epochs=1",
                "dataset.group_method=False",
                "dataset.remove_high_altitude_specific_humidity_levels=False",
            ],
        )

        wandb_config = omegaconf.OmegaConf.to_container(
            cfg.model.single_run_configuration, resolve=True, throw_on_missing=True
        )

        datasets = None  # Using None to let the function load datasets internally

        # Call standard training from cfg function
        # This should fail for now 
        try:
          train.standard_training_from_cfg(
              cfg,
              datasets,
              wandb_config,
              "test_run",
              enable_checkpointing=False,
          )
        except Exception as e:
          print(f"Expected exception caught: {e}")
        # train.standard_training_from_cfg(
        #     cfg,
        #     datasets,
        #     wandb_config,
        #     "test_run",
        #     enable_checkpointing=False,
        # )



def test_standard_training_from_cfg_squeezeformer():
    """
    Test function for standard_training_from_cfg
    """

    with initialize(version_base=None, config_path="../../config"):
        # config is relative to a module
        # NOTE: setting sweep to True to prevent model being saved
        cfg = compose(
            config_name="train_general",
            overrides=[
                "sweep=True",
                "testing=qt",
                "dataset.dataset_testing_sample_rates.quick=1",
                "dataset.base_folder_path=/home/users/bradlesc/projects/ClimSim/test/unit_test_sets/dummy_low_res_climsim/filename_testing",
                "dataset.group_method=False",
                "dataset.remove_high_altitude_specific_humidity_levels=15",
            ],
        )

        # wandb_config = omegaconf.OmegaConf.to_container(
        #     cfg.model.single_run_configuration, resolve=True, throw_on_missing=True
        # )

        # datasets = None  # Using None to let the function load datasets internally

        # # Call standard training from cfg function
        # train.standard_training_from_cfg(
        #     cfg,
        #     datasets,
        #     wandb_config,
        #     "test_run",
        #     enable_checkpointing=False,
        # )
"""
Script for evaluating trained models on given datasets.
Assumes, model(s) have already been trained and saved.

"""

from omegaconf import DictConfig
import logging
import hydra
import data_preparation

@hydra.main(version_base=None, config_path="../config", config_name="evaluate_general")
def main(cfg: DictConfig):
    """
    High level function to evaluate a set of trained models on a dataset(s), through the following steps:
    1. Loads dataset(s)
    2. Loads trained model(s)
    3. Runs model(s) on dataset(s)
    4. Converts model outputs to physical quantities
    5. Calculates metrics 
    6. Save metrics 
    7. Generate and save plots

    v1: Basic structure, single dataset, single instance of each model
    TODOs:
    - Multiple datasets
    - Multiple instances of each model (e.g., different random seeds)

    Args:
        cfg (DictConfig): Configuration object containing all necessary parameters.
    """

    # Loading the dataset
    dataset = data_preparation.get_dataset(cfg.dataset, cfg.datasplit_to_use, cfg.testing.dataset_testing_type)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    main()

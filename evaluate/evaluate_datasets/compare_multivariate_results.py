import warnings

import logging
import string
from omegaconf import DictConfig
import hydra
from train import seed_everything
import torch
import data_preparation
import models
import yaml
import lightning as L
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from pathlib import Path
import plotting 
import os

def load_energy_distance_results(path: Path | str = None) -> dict:
    """
    Load the energy_distance results JSON saved by the script.
    If path is None, defaults to cwd()/energy_distance.json.
    Returns an empty dict if the file does not exist or fails to load.
    """
    results_path = Path(path) if path is not None else Path.cwd() / "energy_distance.json"
    if not results_path.exists():
        logging.warning(f"energy_distance results file not found: {results_path}")
        return {}
    try:
        with results_path.open("r") as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load energy_distance results from {results_path}: {e}")
        return {}

@hydra.main(version_base=None, config_path="../../config", config_name="evaluate_data_groups")
def main(cfg: DictConfig):

        # Seeding everything
    seed_everything(cfg.project.seed)

    torch.set_float32_matmul_precision("medium")

    plotting.init_plotting_settings()

    distance_metric = ["energy_distance", 'kl_divergence']
    dataset_types = ['standard', 'high_sh_altitude_removed']
    standard_dataset_results_path = '/home/users/bradlesc/projects/ClimSim/logs/p2.1.3/4/standard_sub_samples_low_res_sub_group_evaluation_2025-11-17-15-49-22/'
    adapted_dataset_results_path = '/home/users/bradlesc/projects/ClimSim/logs/p2.1.3/4/removed_top_13_sh_altiude_values_2025-11-17-15-34-09/'

    for metric in distance_metric:
        logging.info(f"Loading results for metric: {metric}")
        standard_path = os.path.join(standard_dataset_results_path, f'{metric}_multivariate', f'multivariate_{metric}.json')
        standard_dataset_results = load_energy_distance_results(standard_path)
        adapted_path = os.path.join(adapted_dataset_results_path, f'{metric}_multivariate', f'multivariate_{metric}.json')
        adapted_dataset_results = load_energy_distance_results(adapted_path)
        data_dict = {
            dataset_types[0]: standard_dataset_results,
            dataset_types[1]: adapted_dataset_results
        }
        plotting.plot_compare_multivariate_results(
            data_dict,
            metric,
            save_path=f'comparison_multivariate_{metric}.png'
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    main()
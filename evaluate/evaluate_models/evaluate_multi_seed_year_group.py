"""
Script that plots mse test results for each trained group on the same figure.
"""

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

    fig, ax = plt.subplots(figsize=(8, 4))


    # path_to_results = '/home/users/bradlesc/projects/ClimSim/logs/p2.1.3/1/multi_seed_mlp_performance_on_year_grouped_data_2025-10-30-10-25-04'
    # path_to_results = '/home/users/bradlesc/projects/ClimSim/logs/p2.1.3/4/yus_mlp_multi_seed_reduced_sh_001_2025-11-18-14-58-24'
    # path_to_results = '/home/users/bradlesc/projects/ClimSim/logs/p2.1.1/7/unet_year_groups_multiseed_003_2025-11-07-09-16-56'
    # path_to_results = '/home/users/bradlesc/projects/ClimSim/logs/p2.1.3/6/multiseed_monthly_first_3_years_2025-11-25-15-20-48'
    path_to_results = '/home/users/bradlesc/projects/ClimSim/logs/p2.1.3/6/unet_multiseed_2025-11-25-22-11-05'
    for subfolder in os.listdir(path_to_results):
        if subfolder == 'climsim_unet_group_2':
            logging.warning(f"Skipping folder: {subfolder}")
            continue
        full_path = os.path.join(path_to_results, subfolder)
        if os.path.isdir(full_path) and (subfolder.startswith('yus') or subfolder.startswith('climsim')):
            print(f"Processing folder: {subfolder}")
            results_file = os.path.join(full_path, 'test_results.json')
            with open(results_file, 'r') as f:
                results = json.load(f)
            
                test_losses = [results[str(i)][0]['test/loss'] for i in range(len(results))]
                year_group = int(subfolder[-1])
                x = [year_group] * len(test_losses)
                ax.scatter(x, test_losses)

    # draw baseline horizontal line and add to legend
    # path_to_baseline = '/home/users/bradlesc/projects/ClimSim/logs/p2.1.3/2/marginal_distribution_shifts_001_2025-11-05-10-19-36/yus_mlp_year_group_False/test_results.json'
    # path_to_baseline = '/home/users/bradlesc/projects/ClimSim/logs/p2.1.1/7/unet_full_data_multiseed_001_2025-11-06-22-22-34/climsim_unet_year_group_False/test_results.json'

    # with open(path_to_baseline, 'r') as f:
    #     baseline_results = json.load(f)
    #     baseline_losses = [baseline_results[str(i)][0]['test/loss'] for i in range(len(baseline_results))]


    # # baseline_y = 0.003736531361937523
    # for i in range(len(baseline_losses)):
    #     if i==0:
    #         ax.axhline(y=baseline_losses[i], color='grey', linestyle='--', linewidth=1.5, label='baseline performance')
    #     else:
    #         ax.axhline(y=baseline_losses[i], color='grey', linestyle='--', linewidth=1.5)
    # ax.legend(loc='best', fontsize='small')

    ax.set_xlabel("Training data group index")
    ax.set_ylabel("Test MSE Loss")
    ax.set_title("Test MSE Loss vs Training Data Group Index")
    ax.grid(True)
    fig.tight_layout()


    save_path = 'multi_seed_year_group.png'


    # ensure save_path doesn't collide; append _1, _2, ... before the suffix until an unused filename is found
    save_path = Path(save_path)
    if save_path.exists():
        parent = save_path.parent
        stem = save_path.stem
        suffix = save_path.suffix
        i = 1
        while True:
            candidate = parent / f"{stem}_{i}{suffix}"
            if not candidate.exists():
                save_path = candidate
                logging.info(f"save_path exists; using new path: {save_path}")
                break
            i += 1

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, transparent=True)
    logging.info(f"Saved plot to {save_path}")
    plt.close(fig)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    main()
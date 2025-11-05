"""
Script that is used to evaluate different groups of data for 'distribution shifts'. A dataset with quantifiable distribution shifts is very useful for identifying how much of a generalisation tasks a model must achieve.
"""
import warnings
import netCDF4 # Another weird import issue that is only triggered if netCDF4 imported after wandb 
with (
    warnings.catch_warnings()
):  # To catch annoying pydantic x wandb warning - looks like it should be adressed soon: https://github.com/wandb/wandb/issues/10662
    warnings.filterwarnings("ignore")
    import wandb
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
from evaluate import evaluate_data_groups
import time
import plotting
import numpy as np

@hydra.main(version_base=None, config_path="../config",config_name="evaluate_data_groups")
def main(cfg: DictConfig):

    
    # Seeding everything
    seed_everything(cfg.project.seed)

    plotting.init_plotting_settings()

    torch.set_float32_matmul_precision("medium")


    # Load trainset and testset 
    trainset = data_preparation.get_dataset(cfg.dataset, "train", cfg.testing.dataset_testing_type)
    testset = data_preparation.get_dataset(cfg.dataset, "test", cfg.testing.dataset_testing_type)

    train_input = trainset.input
    testset_input = testset.input

    data_group_sample_size, num_data_groups = data_preparation.calc_sub_sampled_low_res_yearly_group_sample_size_and_num_groups(cfg.dataset, len(trainset), cfg.testing.dataset_testing_type)


    # Iterate through each data group  
    full_results = {option_name: {} for option_name in cfg.evaluation_options}
    for group_idx in range(num_data_groups):

        # get data group 
        train_input_group = train_input[group_idx * data_group_sample_size:(group_idx + 1) * data_group_sample_size]


        # start_time = time.perf_counter()
        results = evaluate_data_groups(cfg, train_input_group, testset_input, evaluation_options=cfg.evaluation_options)
        for option_name in cfg.evaluation_options:
            full_results[option_name][group_idx] = results[option_name]

        # elapsed = time.perf_counter() - start_time

        # logging.info(f"evaluate_data_groups took {elapsed:.3f} seconds for group {group_idx}")

        full_results[group_idx] = results


    # Save results to json file 
    if 'energy_distance' in cfg.evaluation_options: 
        results_path = Path.cwd() / "energy_distance.json"
        with open(results_path, "w") as f:
            json.dump(full_results['energy_distance'], f, indent=2)
        logging.info(f"Saved energy_distance results to {results_path}")
    if 'marginal_distributions' in cfg.evaluation_options:
        results_path = Path.cwd() / "marginal_distribution_distances.json"
        with open(results_path, "w") as f:
            json.dump(full_results['marginal_distributions'], f, indent=2)
        logging.info(f"Saved marginal distribution distances results to {results_path}")
    
    # Plotting results 
    if 'vis_marginals' in cfg.evaluation_options:
        marginal_data = full_results['vis_marginals'][0]
        for var in marginal_data.keys():
            plotting.plot_multiple_marginal_distributions_on_single_plot(
                data_dict=full_results['vis_marginals'],
                variable_name=var, save_path=f"{var}_distributions.png", groups_to_plot=cfg.vis_marginals.groups_to_plot)

    
    if 'energy_distance' in cfg.evaluation_options:
        plotting.plot_energy_distance_results(full_results['energy_distance'], save_path="energy_distance.png")

     
    if 'kl_divergence' in cfg.evaluation_options:
        plotting.plot_kl_divergence_results(full_results['kl_divergence'], save_path="kl_divergence.png")
  
    
    if 'marginal_distributions' in cfg.evaluation_options:
        plotting.plot_standard_feature_marginals(full_results['marginal_distributions'], save_path="marginal_distribution_distances.png")

    


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    main()

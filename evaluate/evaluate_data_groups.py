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

    torch.set_float32_matmul_precision("medium")


    # Load trainset and testset 
    # trainset, _, _ = data_preparation.get_all_datasets(cfg.dataset, cfg.testing.dataset_testing_type)
    trainset = data_preparation.get_dataset(cfg.dataset, "train", cfg.testing.dataset_testing_type)
#     trainset.data_class.set_pressure_grid(data_split="train")

    train_input = trainset.input
#     # train_target = trainset.target
#     # testset_input = testset.input
#     # testset_target = testset.target

#     # Get size of each group 
    data_group_sample_size = (cfg.dataset.samples_per_day * 365 // cfg.dataset.subsample_factors.train
) * cfg.dataset.num_spatial_points

    
    data_group_sample_size = 384 if cfg.testing.dataset_testing_type == "quick" else data_group_sample_size
    data_group_sample_size = 38400 if cfg.testing.dataset_testing_type == "reduced" else data_group_sample_size
    
    logging.info(f'Total trainset size: {len(trainset)}, data group sample size: {data_group_sample_size}')
    rng = np.random.default_rng(cfg.project.seed)
    i_idx = rng.integers(0, data_group_sample_size, size=1000)
    
    num_data_groups = len(trainset) // data_group_sample_size

    logging.info(f'Data group sample size: {data_group_sample_size}, number of data groups: {num_data_groups}')

    # Iterate through each data group  
    ed_values = {}
    near_surface_values = {}
    for group_idx in [0, 6]:  # tqdm(range(num_data_groups), desc="Evaluating data groups"):

        # get data group 
        train_input_group = train_input[group_idx * data_group_sample_size:(group_idx + 1) * data_group_sample_size]
        # train_target_group = train_target[group_idx * data_group_sample_size:(group_idx + 1) * data_group_sample_size]

        # extract near surface variables only for evaluation
        near_surface_air_temp = train_input_group[:,59] #[i_idx]
        near_surface_specific_humidity = train_input_group[:,119] #[i_idx]

        logging.info(f'near_surface_air_temp shape: {near_surface_air_temp.shape}, near_surface_specific_humidity shape: {near_surface_specific_humidity.shape}')

        near_surface_values[group_idx] = {
            "air_temperature": near_surface_air_temp,
            "specific_humidity": near_surface_specific_humidity
        }

        # print(type(train_input_group), train_input_group.shape)


        # # call evaluation functions on data groups and save results
        # evaluation_options = ['energy_distance']  # Options for evaluation metrics
        # logging.info(f'Evaluating data group {group_idx + 1}/{num_data_groups}')

        # start_time = time.perf_counter()
        # ed_value = evaluate_data_groups(train_input_group, train_target_group, testset_input, testset_target, evaluation_options, chunk=cfg.chunk_size)
        # elapsed = time.perf_counter() - start_time

        # logging.info(f"evaluate_data_groups took {elapsed:.3f} seconds for group {group_idx}")
        # ed_values[group_idx] = {"energy_distance": ed_value} #, "p-value": pval}


    plotting.plot_multiple_distributions_on_single_plot(
        data_dict=near_surface_values,
        variable_name="specific_humidity", save_path="near_surface_specific_humidity_distributions.png")

    plotting.plot_multiple_distributions_on_single_plot(
        data_dict=near_surface_values,
        variable_name="air_temperature", save_path="near_surface_air_temperature_distributions.png")    
    
    # Prepare numeric ED values
    # processed_ed = []
    # for v in ed_values.values():
    #     if isinstance(v, dict):
    #         if "energy_distance" in v:
    #             processed_ed.append(float(v["energy_distance"]))
    #         else:
    #             # fallback to first numeric value in dict
    #             first_val = next(iter(v.values()))
    #             processed_ed.append(float(first_val))
    #     elif isinstance(v, (list, tuple)):
    #         processed_ed.append(float(v[0]))
    #     else:
    #         processed_ed.append(float(v))

    # # Save results to json file
    # results_path = Path.cwd() / "ed_values.json"
    # with open(results_path, "w") as f:
    #     json.dump({"ed_values": ed_values}, f, indent=2)
    # logging.info(f"Saved ED values to {results_path}")

    # # Plot results
    # plt.figure(figsize=(8, 4))
    # x = list(range(len(processed_ed)))
    # plt.plot(x, processed_ed, marker="o", linestyle="-")
    # plt.xlabel("Data group index")
    # plt.ylabel("Energy distance")
    # plt.title("Energy distance vs Data group index")
    # plt.grid(True)
    # plt.tight_layout()

    # plot_path = Path.cwd() / "ed_values.png"
    # plt.savefig(plot_path, dpi=200)
    # logging.info(f"Saved plot to {plot_path}")
    # plt.savefig('test_ed_plot.png', dpi=200)
    # Save results to json file

    # plot results

    


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    main()

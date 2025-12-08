"""
Script that is used to evaluate different groups of data for 'distribution shifts'. A dataset with quantifiable distribution shifts is very useful for identifying how much of a generalisation tasks a model must achieve.
"""

import warnings
import netCDF4  # Another weird import issue that is only triggered if netCDF4 imported after wandb

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
from evaluate import evaluate_data_group
import time
import plotting
import numpy as np


@hydra.main(
    version_base=None, config_path="../../config", config_name="evaluate_data_groups"
)
def main(cfg: DictConfig):

    # Seeding everything
    seed_everything(cfg.project.seed)

    plotting.init_plotting_settings()

    torch.set_float32_matmul_precision("medium")

    group_method = cfg.dataset.group_method
    if not group_method:
        raise ValueError(
            "cfg.dataset.group_method must be set to a valid grouping method, cannot be False for evaluating data groups."
        )
    groups = cfg.dataset[group_method].groups.keys()
    test_group = list(cfg.dataset[group_method].test_group.keys())[0]

    # Load trainset and testset
    # trainset = data_preparation.get_dataset(cfg.dataset, "train", cfg.testing.dataset_testing_type)

    # Iterate through each data group
    full_results = {option_name: {} for option_name in cfg.evaluation_options}
    for group_idx in groups:
        logging.info(f"Evaluating data group {group_idx} compared to {test_group}.")

        # Get data groups
        cfg.dataset[group_method].target_group = group_idx

        start_time = time.perf_counter()
        trainset = data_preparation.get_dataset(
            cfg.dataset, "train", cfg.testing.dataset_testing_type
        )
        if hasattr(trainset, "normalisation_stats"):
            normalisation_stats = trainset.normalisation_stats
        else:
            normalisation_stats = None

        testset = data_preparation.get_dataset(
            cfg.dataset,
            "test",
            cfg.testing.dataset_testing_type,
            normalisation_stats=normalisation_stats,
        )

        elapsed = time.perf_counter() - start_time
        logging.info(f"Loaded trainset for group {group_idx} in {elapsed:.3f} seconds")

        logging.info(f"Trainset size for group {group_idx}: {len(trainset)}")

        start_time = time.perf_counter()
        results = evaluate_data_group(
            cfg,
            trainset,
            testset,
            evaluation_options=cfg.evaluation_options,
        )
        for option_name in cfg.evaluation_options:
            full_results[option_name][group_idx] = results[option_name]

        elapsed = time.perf_counter() - start_time
        logging.info(
            f"evaluate_data_groups took {elapsed:.3f} seconds for group {group_idx}"
        )

        full_results[group_idx] = results

    # Save results to json file
    if "multivariate" in cfg.evaluation_options:
        results_path = Path.cwd() / f"multivariate_{cfg.metric_name}.json"
        with open(results_path, "w") as f:
            json.dump(full_results["multivariate"], f, indent=2)
        logging.info(f"Saved multivariate results to {results_path}")
    if "marginals" in cfg.evaluation_options:
        results_path = Path.cwd() / f"marginal_distribution_{cfg.metric_name}.json"
        with open(results_path, "w") as f:
            json.dump(full_results["marginals"], f, indent=2)
        logging.info(f"Saved marginal distribution distances results to {results_path}")

    # Plotting results
    if "vis_marginals" in cfg.evaluation_options:
        marginal_data = full_results["vis_marginals"][0]
        for var in marginal_data.keys():
            plotting.plot_multiple_marginal_distributions_on_single_plot(
                data_dict=full_results["vis_marginals"],
                variable_name=var,
                save_path=f"{var}_distributions.png",
                groups_to_plot=cfg.vis_marginals.groups_to_plot,
            )

    if "multivariate" in cfg.evaluation_options:
        plotting.plot_multivariate_results(
            full_results["multivariate"],
            metric_name=cfg.metric_name,
            save_path=f"multivariate_{cfg.metric_name}.png",
        )

    if "marginals" in cfg.evaluation_options:
        plotting.plot_standard_feature_marginals(
            full_results["marginals"],
            levels=cfg.dataset.levels,
            save_path="marginal_distribution_distances.png",
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    main()

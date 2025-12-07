"""
Script for evaluating trained models on given datasets.
Assumes, model(s) have already been trained and saved.

"""

import json
from typing import Tuple
from omegaconf import DictConfig, OmegaConf
import logging
import hydra
import os
import re
import torch
import yaml
import numpy as np
import numbers

import data_preparation
import models
import evaluate
import plotting


def get_trained_models(path: str) -> Tuple:
    """
    Collects all trained model checkpoints and their configurations from the specified directory.
    Args:
        path (str): The directory path containing the trained models.
    Returns:
        Tuple: A tuple containing the checkpoint path and configuration parameters.
    """
    for subfolder in os.listdir(path):
        full_path = os.path.join(path, subfolder)
        if os.path.isdir(full_path) and (
            subfolder.startswith("yus")
            or subfolder.startswith("climsim")
            or subfolder.startswith("squeezeformer")
        ):
            logging.info(f"Processing folder: {subfolder}")

            # --- Find config files ---
            config_files = [
                f
                for f in os.listdir(full_path)
                if re.match(r"run_config_\d+\.yaml$", f)
            ]
            config_files.sort()  # Picks "run_config_0.yaml" first

            # --- Find checkpoint files ---
            ckpt_files = [
                f
                for f in os.listdir(full_path)
                if re.match(r"(squeezeformer|climsim|yus)_.*\.ckpt$", f)
            ]
            ckpt_files.sort()

            # config = OmegaConf.load(os.path.join(full_path, first_config))

            return ckpt_files, config_files, full_path


@hydra.main(
    version_base=None, config_path="../../config", config_name="evaluate_trained_models"
)
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

    # For now, whilst data groups vary a lot, I wont hardcode store the normalisation stats, hopefully in the future I will be a bit more sure of what data grouping/selection I am using, then I can store my normalisation stats with the data. But for now, we can get the normalisation stats without doing too much of the heavy lifting of loading the data. Note: this means a lot of things are hardcoded!
    logging.info("Getting normalisation stats from training dataset...")
    trainset = data_preparation.ClimSimFromRawDataset(
        mode="train",
        dataset_testing_type=cfg.testing.dataset_testing_type,
        dataset_cfg=cfg.dataset,
        model=None,
        num_workers=int(cfg.dataset.general_dataset_config.num_workers),
        get_normalisation_stats_only=True,
    )

    logging.info("Loading the dataset...")
    dataset = data_preparation.get_dataset(
        cfg.dataset,
        cfg.datasplit_to_use,
        cfg.testing.dataset_testing_type,
        normalisation_stats=trainset.normalisation_stats,
    )
    dataloader = data_preparation.get_dataloader(
        cfg.dataset,
        cfg.datasplit_to_use,
        cfg.testing.dataset_testing_type,
        batch_size=1024,
        dataset=dataset,
    )

    loaded_models = {}

    for model_name, model_path in cfg.models.items():
        logging.info(f"Evaluating model: {model_name} from path: {model_path}")
        ckpt_files, config_files, full_path = get_trained_models(model_path)
        for seed, (ckpt_file, config_file) in enumerate(zip(ckpt_files, config_files)):
            logging.info(
                f"  Loading model checkpoint: {ckpt_file} with config: {config_file}"
            )
            checkpoint_path = os.path.join(full_path, ckpt_file)
            config_path = os.path.join(full_path, config_file)

            model_config = OmegaConf.load(config_path)

            model = models.load_model_from_checkpoint(
                checkpoint_path, model_name, model_config, cfg.dataset
            )
            loaded_models[(model_name, seed)] = {"model": model}

    logging.info("All models loaded successfully.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for model_name, model_dict in loaded_models.items():
        model_dict["model"].to(device)
        model_dict["model"].eval()

    with torch.no_grad():
        targets = []
        for batch in dataloader:
            x, y = batch
            x = x.to(device)

            for name, model_dict in loaded_models.items():
                logging.info(f"Evaluating model: {name}")
                y_hat = model_dict["model"](x)
                if "preds" not in model_dict:
                    model_dict["preds"] = []
                model_dict["preds"].append(y_hat)
            targets.append(y)

        targets = torch.cat(targets, dim=0).detach().to("cpu")

    output_weighting = evaluate.OutputWeighting(cfg=cfg)
    weighted_targets = output_weighting.weight(targets, dataset)

    metrics_calculator = evaluate.MetricsCalculator(num_latlon=dataset.num_latlon)

    results_dict = {}
    for name, model_dict in loaded_models.items():

        preds_obj = model_dict.get("preds", None)
        if preds_obj is None:
            logging.warning(f"No 'preds' found for model {name}; skipping.")
            continue

        model_dict["preds"] = torch.cat(model_dict["preds"], dim=0).detach().to("cpu")
        logging.info(
            f"Concatenated predictions for model: {name} into shape {model_dict['preds'].shape}"
        )
        model_dict["weighted_preds"] = output_weighting.weight(
            model_dict["preds"], dataset
        )

        results = {}
        for metric_name, metric_func in metrics_calculator.metrics_dict.items():
            logging.info(f"Calculating {metric_name}...")
            results[metric_name] = {}
            for var in model_dict["weighted_preds"].data_vars:
                # print(f"  Variable: {var}")
                pred = model_dict["weighted_preds"][var].values
                target = weighted_targets[var].values

                metric_result = metric_func(pred, target).mean()

                results[metric_name][var] = metric_result
                logging.info(f"{metric_name} for {var}: {metric_result.mean():.2f}")

        if name[0] not in results_dict:
            results_dict[name[0]] = {}

        results_dict[name[0]][name[1]] = results

    plotting_values = {}
    for model_name, seeds_dict in results_dict.items():
        n = len(seeds_dict)

        for metric_name in metrics_calculator.metrics_dict.keys():
            for var in weighted_targets.data_vars:

                values = [seeds_dict[seed][metric_name][var] for seed in seeds_dict]

                # sample SD → SE
                standard_error = np.std(values, ddof=1) / np.sqrt(n)
                mean = np.mean(values)

                # logging.info(
                #     f"Model: {model_name}, Metric: {metric_name}, Variable: {var}, "
                #     f"Standard Error across seeds: {standard_error:.4f}"
                # )
                if model_name not in plotting_values:
                    plotting_values[model_name] = {}
                if metric_name not in plotting_values[model_name]:
                    plotting_values[model_name][metric_name] = {}
                if var not in plotting_values[model_name][metric_name]:
                    plotting_values[model_name][metric_name][var] = {}
                plotting_values[model_name][metric_name][var][
                    "standard_error"
                ] = standard_error
                plotting_values[model_name][metric_name][var]["mean"] = mean

    with open(f"results.json", "w") as f:
        json.dump((results_dict), f, indent=4)

    plotting.init_plotting_settings()
    plotting.plot_trained_model_evaluations(plotting_values)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    main()

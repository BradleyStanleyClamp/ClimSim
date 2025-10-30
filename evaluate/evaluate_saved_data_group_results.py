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

@hydra.main(version_base=None, config_path="../config", config_name="evaluate_data_groups")
def main(cfg: DictConfig):

        # Seeding everything
    seed_everything(cfg.project.seed)

    torch.set_float32_matmul_precision("medium")

    plotting.init_plotting_settings()

    # load results 
    energy_distance_dict = load_energy_distance_results(cfg.energy_distance_results_path)

    p = Path(cfg.energy_distance_results_path) if cfg.energy_distance_results_path else Path.cwd()
    base_dir = p.parent if p.suffix else p
    save_path = base_dir / "energy_distance_plot.png"

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

    plotting.plot_energy_distance_results_with_p_values(energy_distance_dict, str(save_path), cmap="viridis")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    main()
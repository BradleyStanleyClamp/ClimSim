"""
Script for comapring the distribution shift and model error across different models trained on different data groups. That also loads the data on the fly 
"""

import logging
from omegaconf import DictConfig
import hydra
from train import seed_everything
import plotting
import json
import os
import models

# Path to models results

models_results_paths = {
    "climsim_unet": "p2.1.4/3/unet_from_npy_multiseed_2025-12-22-10-58-16",
    "vib_unet": "p2.1.4/3/vib_unet_from_npy_multiseed_2025-12-22-10-58-42",
    # "yus_mlp": "p2.1.3/11/yus_mlp_from_npy_multiseed_2025-12-10-09-37-03",
    "squeezeformer": "p2.1.3/11/squeezeformer_from_npy_multiseed_2025-12-10-11-35-13",
    "mlp": "p2.1.3/11/mlp_from_npy_multiseed_2025-12-10-15-08-48",
    # "old_climsim_unet": "p2.1.3/11/unet_from_npy_multiseed_2025-12-10-09-35-41",
}

checkpoint_path = '/gws/nopw/j04/iecdt/bstanleyclamp/checkpoints/' #${project.project}/${project.task}/${project.name}_${project.timestamp}/${multirun_dir_name}'
log_path = '/home/users/bradlesc/projects/ClimSim/logs/'

model_paths = {
    "climsim_unet": "p2.1.4/3/unet_from_npy_multiseed_2025-12-22-10-58-16",
    "vib_unet": "p2.1.4/3/vib_unet_from_npy_multiseed_2025-12-22-10-58-42",
    # "yus_mlp": "p2.1.3/11/yus_mlp_from_npy_multiseed_2025-12-10-09-37-03",
    "squeezeformer": "p2.1.3/11/squeezeformer_from_npy_multiseed_2025-12-10-11-35-13",
    "mlp": "p2.1.3/11/mlp_from_npy_multiseed_2025-12-10-15-08-48",
    # "old_climsim_unet": "p2.1.3/11/unet_from_npy_multiseed_2025-12-10-09-35-41",

}



distribution_shift_path = "p2.1.3/10/subsample_7_seasonality_energy_distance_2025-12-28-11-07-18/eval_data_groups/multivariate_energy_distance.json"

group_to_int = {"DJF": 0, "MAM": 1, "JJA": 2, "SON": 3}


def load_results(path: str) -> dict:
    """
    Load the results JSON saved by the script.
    Returns an empty dict if the file does not exist or fails to load.
    """
    output_results = {}
    for subfolder in os.listdir(path):
        full_path = os.path.join(path, subfolder)
        if os.path.isdir(full_path) and (
            subfolder.startswith("yus")
            or subfolder.startswith("climsim")
            or subfolder.startswith("squeezeformer")
            or subfolder.startswith("mlp")
            or subfolder.startswith("vib")
        ):
            print(f"processing folder: {subfolder}")
            group = subfolder.split("_")[-1]
            output_results[group] = {}
            results_file = os.path.join(full_path, "test_results.json")
            with open(results_file, "r") as f:
                results = json.load(f)

                test_losses = [
                    results[str(i)][0]["test/loss"] for i in range(len(results))
                ]
                mean_test_loss = sum(test_losses) / len(test_losses)
                max_test_loss = max(test_losses)
                min_test_loss = min(test_losses)
                output_results[group]["mean_test_loss"] = mean_test_loss
                output_results[group]["max_test_loss"] = max_test_loss
                output_results[group]["min_test_loss"] = min_test_loss
    return output_results


def load_distribution_shift_results(path: str) -> dict:
    """
    Load the distribution shift results JSON saved by the script.
    Returns an empty dict if the file does not exist or fails to load.
    """
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load distribution shift results from {path}: {e}")
        return {}


def load_target_group_models(model_name: str, model_path: str, target_group):
    """
    Load the model given its name.
    """
    model_folder_path = os.path.join(checkpoint_path, model_path, f"{model_name}_group_{target_group}.ckpt")
    model_paths = os.listdir(model_folder_path)
    print(model_paths)
    models = []
    for path in model_paths:
        if path.endswith(".ckpt"):
            full_model_path = os.path.join(model_folder_path, path)
            model = models.load_from_checkpoint(full_model_path)
            models.append(model)
    
    print(f"Loaded {len(models)} models for {model_name} group {target_group} from {model_folder_path}")

    return models

@hydra.main(
    version_base=None, config_path="../../config", config_name="evaluate_data_groups"
)
def main(cfg: DictConfig):

    # Seeding everything
    seed_everything(cfg.project.seed)

    plotting.init_plotting_settings()

    # Load results for each model
    models_results = {}
    for target_group in group_to_int.keys():
        print(f"Processing target group: {target_group}")
        for model_name, model_path in model_paths.items():
            models = load_target_group_models(model_name, model_path, target_group)
            # models_results[model_name] = load_results(results_path)

    # # Get distribution shift results
    # distribution_shift_results = load_distribution_shift_results(
    #     distribution_shift_path
    # )

    # # Plot distribution shift vs error
    # plotting.plot_distribution_shift_vs_error(
    #     distribution_shift_results,
    #     models_results,
    # )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    main()

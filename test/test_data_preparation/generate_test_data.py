"""
Script for generating test data for unit tests
"""

from pathlib import Path
import yaml
import os
import numpy as np


def load_sub_sampled_low_res_data_and_save_reduced_versions(
    sub_sampled_low_res_config_path: str = "../../config/datasets/sub_sampled_low_res.yaml",
    save_repository_path: str = "test/unit_test_sets/sub_sampled_low_res/",
):
    """
    Function for providing test data for unit tests, but it may also be useful for saving reduced versions of the data for quick experiments.
    It will load in the full .npy files, subsample them and then save them in the testing repository as .npy files.

    Args:
        sub_sampled_low_res_config_path: (str) path to the config file for the sub_sampled_low_res dataset
        save_repository_path: (str) path to the repository where the unit test sets will be saved
    """

    # Resolve config path relative to this file when given as a relative path
    config_path = Path(sub_sampled_low_res_config_path)
    if not config_path.is_absolute():
        config_path = (Path(__file__).resolve().parent / config_path).resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r") as f:
        config = yaml.safe_load(f)

    modes = ["train", "val", "scoring"]
    options = ["input", "target"]
    data_repository_path = config["data_path"]

    subsample_value = config["dataset_testing_fractions"]["unit_test"]

    for mode in modes:
        for option in options:
            # Load the data
            data_path = os.path.join(data_repository_path, f"{mode}_{option}.npy")

            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data file not found: {data_path}")

            data = np.load(data_path)

            # Sub-sample the data
            subsampled_data = subsample_data(data, subsample_value)

            # Save the subsampled data
            save_path = os.path.join(save_repository_path, f"{mode}_{option}.npy")
            # Ensure target directory exists and avoid overwriting existing files
            save_dir = Path(save_path).parent
            save_dir.mkdir(parents=True, exist_ok=True)

            if Path(save_path).exists():
                raise FileExistsError(
                    f"File already exists and will not be overwritten: {save_path}"
                )

            np.save(save_path, subsampled_data)



def subsample_data(data: np.ndarray, subsample_value: int) -> np.ndarray:
    """
    Sub-sample the data by taking every nth sample based on the subsample value. Also check that subsample value is proportion or quantity

    Args:
        data: (np.ndarray) The original data array, with first dimension being the sample dimension.
        subsample_value: (int) The subsampling factor.

    Returns:
        np.ndarray: The subsampled data array.
    """
    if subsample_value > 1:
        return data[:subsample_value]
    else:
        data_length = data.shape[0]
        subsample_count = int(data_length * subsample_value)
        return data[:subsample_count]


if __name__ == "__main__":
    load_sub_sampled_low_res_data_and_save_reduced_versions()

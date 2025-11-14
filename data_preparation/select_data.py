"""
Script for selecting datasets and data loaders
"""

import logging
from typing import List, Tuple
from torch.utils.data import DataLoader, Dataset
from omegaconf import DictConfig
import data_preparation


def get_dataset(
    dataset_cfg, mode: str, dataset_testing_type: str, model: str = None
) -> Dataset:
    """
    Function that gets you the specified dataset

    Args:
        dataset_cfg: configuration for the dataset to be selected containing:
            dataset_name: (str) name of the dataset to be selected
            dataset_testing_fractions: (DictConfig) configuration containing the fractions for quick and reduced datasets

        mode: (str) one of 'train', 'val' or 'test', specifying which dataset split to return
        dataset_testing_type: (str) size of dataset to be used, related to the type of testing e.g quick, reduced, full
        model: (str) name of the model to be used, e.g if mlp then data can be used as is, but if unet then further processing is required, and if none then no further processing is required

    Returns:
        Dataset: the specified dataset
    """
    assert mode in [
        "train",
        "val",
        "test",
    ], "mode must be one of 'train', 'val' or 'test'"
    assert dataset_testing_type in [
        "quick",
        "reduced",
        "full",
    ], "dataset_testing_type must be one of 'quick', 'reduced' or 'full'"

    if dataset_cfg.dataset_name == "subsampled_low_res":

        return data_preparation.SubSampledLowResDataset(
            mode, dataset_testing_type, dataset_cfg, model=model
        )


def get_dataloader(dataset_cfg, mode, dataset_testing_type, batch_size) -> DataLoader:
    """
    Function that gets you the specified dataloader, and only shuffles if in training mode
    Args:
        dataset_cfg: configuration for the dataset to be selected containing:
            dataset_name: (str) name of the dataset to be selected
            dataset_testing_fractions: (DictConfig) configuration containing the fractions for quick and reduced datasets
        mode: (str) one of 'train', 'val' or 'test', specifying which dataset split to return
        dataset_testing_type: (str) size of dataset to be used, related to the type of testing e.g quick, reduced, full

    Returns:
        DataLoader: the specified dataloader
    """
    dataset = get_dataset(dataset_cfg, mode, dataset_testing_type)
    return DataLoader(
        dataset,
        batch_size=dataset_cfg.batch_size,
        shuffle=(mode == "train"),
        num_workers=int(dataset_cfg.general_dataset_config.num_workers),
        persistent_workers=dataset_cfg.general_dataset_config.persistent_workers,
        prefetch_factor=dataset_cfg.general_dataset_config.prefetch_factor,
    )


def get_all_datasets(
    dataset_cfg, dataset_testing_type: str, model: str = None
) -> tuple[Dataset, Dataset, Dataset]:
    """
    Function that gets you train, val and test datasets

    Args:
        dataset_cfg: configuration for the dataset to be selected containing:
            dataset_name: (str) name of the dataset to be selected
            dataset_testing_fractions: (DictConfig) configuration containing the fractions for quick and reduced datasets
        dataset_testing_type: (str) size of dataset to be used, related to the type of testing e.g quick, reduced, full

    Returns:
        tuple[Dataset, Dataset, Dataset]: train, val and test datasets
    """
    train_dataset = get_dataset(dataset_cfg, "train", dataset_testing_type, model=model)
    logging.info(f"trainset loaded with {len(train_dataset)} samples")
    val_dataset = get_dataset(dataset_cfg, "val", dataset_testing_type, model=model)
    logging.info(f"valset loaded with {len(val_dataset)} samples")
    test_dataset = get_dataset(dataset_cfg, "test", dataset_testing_type, model=model)
    logging.info(f"testset loaded with {len(test_dataset)} samples")

    return train_dataset, val_dataset, test_dataset


def get_all_dataloaders(
    dataset_cfg,
    batch_size: int,
    dataset_testing_type: str,
    datasets: Tuple[Dataset, Dataset, Dataset],
    model: str = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Function that gets you train, val and test dataloaders, with the option of precomputed datasets
    Args:
        dataset_cfg: configuration for the dataset to be selected containing:
            dataset_name: (str) name of the dataset to be selected
            dataset_testing_fractions: (DictConfig) configuration containing the fractions for quick and reduced datasets
        batch_size: (int) batch size for the dataloaders
        dataset_testing_type: (str) size of dataset to be used, related to the type of testing e.g quick, reduced, full
        datasets: (Tuple[Dataset, Dataset, Dataset]) tuple containing the train, val and test datasets

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: train, val and test dataloaders
    """
    if datasets is not None:
        trainset, valset, testset = datasets
    else:
        trainset, valset, testset = get_all_datasets(
            dataset_cfg, dataset_testing_type, model=model
        )

    logging.info(
        f"Creating dataloaders with batch size {batch_size} using {dataset_cfg.general_dataset_config.num_workers} workers"
    )
    train_dataloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(dataset_cfg.general_dataset_config.num_workers),
        persistent_workers=dataset_cfg.general_dataset_config.persistent_workers,
        prefetch_factor=dataset_cfg.general_dataset_config.prefetch_factor,
    )
    val_dataloader = DataLoader(
        valset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(dataset_cfg.general_dataset_config.num_workers),
        persistent_workers=dataset_cfg.general_dataset_config.persistent_workers,
        prefetch_factor=dataset_cfg.general_dataset_config.prefetch_factor,
    )
    test_dataloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(dataset_cfg.general_dataset_config.num_workers),
        persistent_workers=dataset_cfg.general_dataset_config.persistent_workers,
        prefetch_factor=dataset_cfg.general_dataset_config.prefetch_factor,
    )

    return train_dataloader, val_dataloader, test_dataloader


def sample_data_based_on_testing_type(
    data: tuple, dataset_testing_type: str, dataset_testing_fraction_cfg: DictConfig
) -> tuple:
    """
    Function that subsamples data based on the dataset_testing_type

    Args:
        data: (tuple) datasets to be subsampled, usually input and target data
        dataset_testing_type: (str) size of dataset to be used, related to the type of testing e.g quick, reduced, full
        dataset_testing_fraction_cfg: (DictConfig) configuration containing the fractions for quick and reduced datasets

    Returns:
        tuple: subsampled input and target data
    """
    testing_value = dataset_testing_fraction_cfg[dataset_testing_type]

    if testing_value > 1.0:
        return select_first_n_samples(data, [int(testing_value)])

    elif testing_value == 1.0:
        return data
    elif isinstance(testing_value, str):
        raise ValueError(
            f"testing value is set to {testing_value}, cannot sample real data"
        )
    else:
        return select_first_n_samples(
            data, [int(d.shape[0] * testing_value) for d in data]
        )


def select_first_n_samples(data: tuple, n_samples: List[int]) -> tuple:
    """
    Function that selects the first n samples from the data

    Args:
        data: (tuple) datasets to select samples from, usually input and target data
        n_samples: (int) number of samples to select

    Returns:
        tuple: the selected samples from the data
    """
    if len(n_samples) == 1:
        return tuple(d[: n_samples[0]] for d in data)
    else:
        return tuple(d[: n_samples[i]] for i, d in enumerate(data))


def select_year_of_data(
    data: tuple,
    year_index: int,
    dataset_cfg: DictConfig,
    dataset_testing_type: str,
    days_per_year=365,
) -> tuple:
    """
    Function that selects data corresponding to a specific year, used for the sub_sampled_low_res dataset

    Args:
        data: (tuple) datasets to select samples from, usually input and target data
        year_index: (int) index of the year to select
        dataset_cfg: (DictConfig) configuration for the dataset
        dataset_testing_type: (str) size of dataset to be used, related to the type of testing e.g quick, reduced, full
        days_per_year: (int) number of days in a year, defaults to 365

    Returns:
        tuple: the selected samples from the data corresponding to the specified year
    """

    data_group_sample_size, num_data_groups = (
        calc_sub_sampled_low_res_yearly_group_sample_size_and_num_groups(
            dataset_cfg, len(data[0]), dataset_testing_type, days_per_year
        )
    )
    assert (
        year_index < num_data_groups
    ), f"year_index {year_index} out of range, only {num_data_groups} data groups available"

    output = tuple(
        d[
            year_index
            * data_group_sample_size : (year_index + 1)
            * data_group_sample_size
        ]
        for d in data
    )

    assert all(
        o.shape[0] == data_group_sample_size for o in output
    ), "Selected data does not have the correct number of samples"
    return output


def calc_sub_sampled_low_res_yearly_group_sample_size_and_num_groups(
    dataset_cfg: DictConfig,
    dataset_length: int,
    dataset_testing_type: str,
    days_per_year=365,
):
    """
    Calculate the sample size and number of groups for the sub-sampled low resolution dataset.

    Args:
        dataset_cfg: (DictConfig) configuration for the dataset
        dataset_length: (int) length of the dataset
        dataset_testing_type: (str) type of testing to be performed
        days_per_year: (int) number of days in a year

    Returns:
        tuple: (data_group_sample_size, num_data_groups)
    """
    samples_per_day = dataset_cfg.samples_per_day
    subsample_factor = dataset_cfg.subsample_factors.train
    num_spatial_points = dataset_cfg.num_spatial_points

    data_group_sample_size = (
        samples_per_day * days_per_year // subsample_factor
    ) * num_spatial_points

    data_group_sample_size = (
        384 if dataset_testing_type == "quick" else data_group_sample_size
    )
    data_group_sample_size = (
        38400 if dataset_testing_type == "reduced" else data_group_sample_size
    )

    num_data_groups = dataset_length // data_group_sample_size

    logging.info(
        f"Data group sample size: {data_group_sample_size}, number of data groups: {num_data_groups}"
    )
    return data_group_sample_size, num_data_groups

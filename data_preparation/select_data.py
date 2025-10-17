"""
Script for selecting datasets and data loaders 
"""

import logging
from typing import List
from torch.utils.data import DataLoader, Dataset
from omegaconf import DictConfig
import data_preparation

def get_dataset(dataset_cfg, mode:str, dataset_testing_type: str) -> Dataset:
    """
    Function that gets you the specified dataset

    Args:
        dataset_cfg: configuration for the dataset to be selected containing:
            dataset_name: (str) name of the dataset to be selected
            dataset_testing_fractions: (DictConfig) configuration containing the fractions for quick and reduced datasets

        mode: (str) one of 'train', 'val' or 'test', specifying which dataset split to return
        dataset_testing_type: (str) size of dataset to be used, related to the type of testing e.g quick, reduced, full

    Returns:
        Dataset: the specified dataset
    """
    assert mode in ["train", "val", "test"], "mode must be one of 'train', 'val' or 'test'"
    assert dataset_testing_type in [
            "quick",
            "reduced",
            "full",
        ], "dataset_testing_type must be one of 'quick', 'reduced' or 'full'"

    if dataset_cfg.dataset_name == "subsampled_low_res":
        return data_preparation.SubSampledLowResDataset(mode, dataset_testing_type, dataset_cfg)

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
    return DataLoader(dataset, batch_size=dataset_cfg.batch_size, shuffle=(mode=="train"))

def get_all_datasets(dataset_cfg, dataset_testing_type: str) -> tuple[Dataset, Dataset, Dataset]:
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
    logging.info("loading trainset")
    train_dataset = get_dataset(dataset_cfg, "train", dataset_testing_type)
    logging.info(f'trainset loaded with {len(train_dataset)} samples')
    val_dataset = get_dataset(dataset_cfg, "val", dataset_testing_type)
    logging.info(f'valset loaded with {len(val_dataset)} samples')
    test_dataset = get_dataset(dataset_cfg, "test", dataset_testing_type)
    logging.info(f'testset loaded with {len(test_dataset)} samples')

    return train_dataset, val_dataset, test_dataset
    
    

def get_all_dataloaders(dataset_cfg, batch_size: int, dataset_testing_type: str) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Function that gets you train, val and test dataloaders
    Args:
        dataset_cfg: configuration for the dataset to be selected containing:
            dataset_name: (str) name of the dataset to be selected
            dataset_testing_fractions: (DictConfig) configuration containing the fractions for quick and reduced datasets
        batch_size: (int) batch size for the dataloaders
        dataset_testing_type: (str) size of dataset to be used, related to the type of testing e.g quick, reduced, full

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: train, val and test dataloaders
    """
    logging.info("Loading datasets...")
    trainset, valset, testset = get_all_datasets(dataset_cfg, dataset_testing_type)

    train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(valset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader

def sample_data_based_on_testing_type(data: tuple, dataset_testing_type: str, dataset_testing_fraction_cfg: DictConfig) -> tuple:
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
        return select_first_n_samples(data, int(testing_value))

    elif testing_value == 1.0:
        return data
    elif isinstance(testing_value, str):
        raise ValueError(f"testing value is set to {testing_value}, cannot sample real data")
    else:
        return select_first_n_samples(data, [int(d.shape[0] * testing_value) for d in data])

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
        return tuple(d[:n_samples[0]] for d in data)
    else:
        return tuple(d[:n_samples[i]] for i, d in enumerate(data))
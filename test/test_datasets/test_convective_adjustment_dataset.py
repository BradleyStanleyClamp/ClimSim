import numpy as np
import torch 
from data_preparation import ConvectiveAdjustmentDataset
import data_preparation
from omegaconf import OmegaConf, DictConfig

def test_seeding_consistency():
    """
    Test that setting the random seed produces consistent datasets.
    """

    dataset_cfg = {
        "factor_ranges": {
            "abs_coeff": {
                "range_target": (1.229e-4, 2.259e-4),
                "range_support": (1.229e-4, 1.744e-4),
            },
            "albedo": {"range_target": (0.001, 0.999), "range_support": (0.001, 0.5)},
            "Q": {"range_target": (341.3, 500), "range_support": (341.3, 420.65)},
        },
        "num_samples_per_factor_group": 100,
        "num_levels": 3,
        "ood_percent": 1.05,
    }

    train_dataset1 = ConvectiveAdjustmentDataset(
        dataset_mode="train",
        dataset_type="in_domain",
        dataset_cfg=dataset_cfg,
    )

    
    train_dataset2 = ConvectiveAdjustmentDataset(
        dataset_mode="train",
        dataset_type="in_domain",
        dataset_cfg=dataset_cfg,
    )

    # Check that the datasets are identical
    assert torch.equal(train_dataset1.input, train_dataset2.input), "Inputs differ"
    assert torch.equal(train_dataset1.target, train_dataset2.target), "Targets differ"
    for p1, p2 in zip(train_dataset1.params, train_dataset2.params):
        for key in p1.keys():
            assert np.isclose(p1[key], p2[key]), f"Params differ for key {key}"


def test_sampling_composition_training_set():
    
    x = [1, 0.5, 3, 5]
    y = [1, 5, 3, 0.5]
    # in should be: [(1, 1), (0.5, 5), (5, 0.5)]
    # z = np.arange(10)
    a = {'x': x, 'y': y} #, 'z': z}

    for i in range(len(x)):
        for b in a.keys():
            keep = True
            for c in a.keys():
                if b != c:
                    if a[c][i] > 2:
                        keep = False    
            if keep:
                print(x[i], y[i]) 
                break


def test_select_convective_adjustment_dataset():

    dataset_cfg = {
        "dataset_name": "convective_adjustment",
        "dataset_type": "in_domain",
        "factor_ranges": {
            "abs_coeff": {
                "range_target": (1.229e-4, 2.259e-4),
                "range_support": (1.229e-4, 1.744e-4),
            },
            "albedo": {"range_target": (0.001, 0.999), "range_support": (0.001, 0.5)},
            "Q": {"range_target": (341.3, 500), "range_support": (341.3, 420.65)},
        },
        "num_samples_per_factor_group": 1000,
        "num_levels": 3,
        "ood_percent": 1.05,
    }
    dataset_cfg = OmegaConf.create(dataset_cfg)
    mode = 'train'
    dataset_testing_type = 'quick'
    dataset = data_preparation.get_dataset(dataset_cfg, mode, dataset_testing_type)


if __name__ == "__main__":
    test_sampling_composition_training_set()
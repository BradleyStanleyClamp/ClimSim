import numpy as np
import torch 
from data_preparation import ConvectiveAdjustmentDataset


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
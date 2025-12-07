import torch

import data_preparation
from omegaconf import DictConfig
from pathlib import Path
import os
import pytest
import xarray as xr
import numpy as np

v1_inputs = [
    "state_t",
    "state_q0001",
    "state_ps",
    "pbuf_SOLIN",
    "pbuf_LHFLX",
    "pbuf_SHFLX",
]
v1_targets = [
    "ptend_t",
    "ptend_q0001",
    "cam_out_NETSW",
    "cam_out_FLWDS",
    "cam_out_PRECSC",
    "cam_out_PRECC",
    "cam_out_SOLS",
    "cam_out_SOLL",
    "cam_out_SOLSD",
    "cam_out_SOLLD",
]

output_scale_file_path = os.path.join(
    Path(__file__).resolve().parents[2],
    "preprocessing",
    "normalizations",
    "outputs",
    "output_scale.nc",
)


def test_climsim_from_raw_dataset_unit_test_select_target_years_months():
    dataset = data_preparation.ClimSimFromRawDataset(
        mode="train",
        dataset_testing_type="qt",
        dataset_cfg=None,
        model=None,
        unit_test_specific_methods=True,
    )

    dataset_cfg = DictConfig(
        {
            "target_years": [
                "0001",
                "0002",
                "0003",
                "0004",
                "0005",
                "0006",
                "0007",
                "0008",
                "0009",
            ],
            "group_method": "group_by_months",  # Whether to select a year of data to train on, if not false should be the year [0, 6] to select
            "group_by_months": {
                "num_groups": 3,  # (02, 03, 04)
                "target_group": 1,
                "groups": [["02"], ["03"], ["04"]],
                "test_group": ["05"],
            },
        }
    )

    years, months = dataset._select_target_years_months(
        mode="train", dataset_cfg=dataset_cfg
    )
    assert years == [
        "0001",
        "0002",
        "0003",
        "0004",
        "0005",
        "0006",
        "0007",
        "0008",
        "0009",
    ]
    assert months == ["03"]

    years, months = dataset._select_target_years_months(
        mode="test", dataset_cfg=dataset_cfg
    )
    assert years == [
        "0001",
        "0002",
        "0003",
        "0004",
        "0005",
        "0006",
        "0007",
        "0008",
        "0009",
    ]
    assert months == ["05"]


def test_climsim_from_raw_dataset_unit_test_get_dataset_filenames_working():
    base_dir = Path(__file__).resolve().parents[1]
    data_path = os.path.join(
        base_dir, "unit_test_sets", "dummy_low_res_climsim/", "filename_testing"
    )

    dataset_config = DictConfig(
        {
            "base_folder_path": data_path,
            "target_years": ["0001"],
            "target_months": ["02"],
            "sample_rate": 1,
        }
    )
    dataset = data_preparation.ClimSimFromRawDataset(
        mode="train",
        dataset_testing_type="qt",
        dataset_cfg=dataset_config,
        model=None,
        unit_test_specific_methods=True,
    )
    input_filenames, target_filenames = dataset._get_dataset_filenames(
        base_folder_path=dataset_config.base_folder_path,
        target_years=dataset_config.target_years,
        target_months=dataset_config.target_months,
    )
    assert len(input_filenames) == 3
    assert len(target_filenames) == 3
    for input_file, target_file in zip(input_filenames, target_filenames):
        assert "E3SM-MMF.mli" in input_file
        assert "E3SM-MMF.mlo" in target_file


def test_climsim_from_raw_dataset_unit_test_get_dataset_filenames_failing():
    base_dir = Path(__file__).resolve().parents[1]
    data_path = os.path.join(
        base_dir, "unit_test_sets", "dummy_low_res_climsim/", "filename_testing"
    )

    dataset_config = DictConfig(
        {
            "base_folder_path": data_path,
            "target_years": ["0001"],
            "target_months": ["02_should_fail"],
            "sample_rate": 1,
        }
    )
    dataset = data_preparation.ClimSimFromRawDataset(
        mode="train",
        dataset_testing_type="qt",
        dataset_cfg=dataset_config,
        model=None,
        unit_test_specific_methods=True,
    )

    input_filenames, target_filenames = dataset._get_dataset_filenames(
        base_folder_path=dataset_config.base_folder_path,
        target_years=dataset_config.target_years,
        target_months=dataset_config.target_months,
    )

    assert len(input_filenames) == 2
    assert len(target_filenames) == 2


def test_climsim_from_raw_dataset_unit_test_sample_filenames():
    # For reproducibility

    dataset = data_preparation.ClimSimFromRawDataset(
        mode="train",
        dataset_testing_type="qt",
        dataset_cfg=None,
        model=None,
        unit_test_specific_methods=True,
    )
    input_filenames = [f"E3SM-MMF.mli.{i}.nc" for i in range(10)]
    target_filenames = [f"E3SM-MMF.mlo.{i}.nc" for i in range(10)]
    sampled_input_filenames, sampled_target_filenames = dataset._sample_filenames(
        input_filelist=input_filenames,
        target_filelist=target_filenames,
        mode="train",
        num_files=3,
    )
    assert len(sampled_input_filenames) == 3
    assert len(sampled_target_filenames) == 3

    assert sampled_input_filenames == [
        "E3SM-MMF.mli.2.nc",
        "E3SM-MMF.mli.8.nc",
        "E3SM-MMF.mli.4.nc",
    ]
    assert sampled_target_filenames == [
        "E3SM-MMF.mlo.2.nc",
        "E3SM-MMF.mlo.8.nc",
        "E3SM-MMF.mlo.4.nc",
    ]
    sampled_input_filenames, sampled_target_filenames = dataset._sample_filenames(
        input_filelist=input_filenames,
        target_filelist=target_filenames,
        mode="val",
        num_files=3,
    )
    assert len(sampled_input_filenames) == 3
    assert len(sampled_target_filenames) == 3

    print(sampled_input_filenames)
    print(sampled_target_filenames)

    assert sampled_input_filenames == [
        "E3SM-MMF.mli.9.nc",
        "E3SM-MMF.mli.1.nc",
        "E3SM-MMF.mli.6.nc",
    ]
    assert sampled_target_filenames == [
        "E3SM-MMF.mlo.9.nc",
        "E3SM-MMF.mlo.1.nc",
        "E3SM-MMF.mlo.6.nc",
    ]
    sampled_input_filenames, sampled_target_filenames = dataset._sample_filenames(
        input_filelist=input_filenames,
        target_filelist=target_filenames,
        mode="test",
        num_files=3,
    )
    assert len(sampled_input_filenames) == 3
    assert len(sampled_target_filenames) == 3

    print(sampled_input_filenames)
    print(sampled_target_filenames)

    assert sampled_input_filenames == [
        "E3SM-MMF.mli.7.nc",
        "E3SM-MMF.mli.3.nc",
        "E3SM-MMF.mli.0.nc",
    ]
    assert sampled_target_filenames == [
        "E3SM-MMF.mlo.7.nc",
        "E3SM-MMF.mlo.3.nc",
        "E3SM-MMF.mlo.0.nc",
    ]


def test_climsim_from_raw_dataset_unit_test_combine_datasets():
    base_dir = Path(__file__).resolve().parents[1]
    data_path = os.path.join(
        base_dir, "unit_test_sets", "dummy_low_res_climsim/", "filename_testing"
    )

    dataset_config = DictConfig(
        {
            "base_folder_path": data_path,
            "target_years": ["0001"],
            "target_months": ["02"],
            "sample_rate": 1,
        }
    )
    dataset = data_preparation.ClimSimFromRawDataset(
        mode="train",
        dataset_testing_type="qt",
        dataset_cfg=dataset_config,
        model=None,
        unit_test_specific_methods=True,
    )
    input_filenames, target_filenames = dataset._get_dataset_filenames(
        base_folder_path=dataset_config.base_folder_path,
        target_years=dataset_config.target_years,
        target_months=dataset_config.target_months,
    )
    input_ds, target_ds = dataset._combine_datasets(
        input_filenames,
        target_filenames,
        v1_inputs=v1_inputs,
        v1_targets=v1_targets,
    )
    assert "state_t" in input_ds
    assert "state_q0001" in input_ds
    assert "ptend_t" in target_ds
    assert "ptend_q0001" in target_ds
    assert input_ds.sizes["sample"] == 3
    assert target_ds.sizes["sample"] == 3


def test_climsim_from_raw_dataset_unit_test_spatial_selection_none():
    dataset = data_preparation.ClimSimFromRawDataset(
        mode="train",
        dataset_testing_type="qt",
        dataset_cfg=None,
        model=None,
        unit_test_specific_methods=True,
    )

    # Create dummy input and target datasets
    input_data = np.random.rand(1000, 384, 60)
    target_data = np.random.rand(1000, 384, 60)

    input_dataset = xr.Dataset({"state_t": (("sample", "ncol", "lev"), input_data)})
    target_dataset = xr.Dataset({"ptend_t": (("sample", "ncol", "lev"), target_data)})

    path_to_grid_info = (
        "/home/users/bradlesc/projects/ClimSim/grid_info/ClimSim_low-res_grid-info.nc"
    )

    input_dataset_out, output_dataset_out = dataset._spatial_selection(
        input_dataset, target_dataset, path_to_grid_info, spatial_selection_method=False
    )

    assert np.array_equal(input_dataset.to_array(), input_dataset_out.to_array())
    assert np.array_equal(target_dataset.to_array(), output_dataset_out.to_array())

def test_climsim_from_raw_dataset_unit_test_spatial_selection_northern_hemisphere():
    dataset = data_preparation.ClimSimFromRawDataset(
        mode="train",
        dataset_testing_type="qt",
        dataset_cfg=None,
        model=None,
        unit_test_specific_methods=True,
    )

    # Create dummy input and target datasets
    input_data = np.random.rand(1000, 384, 60)
    target_data = np.random.rand(1000, 384, 60)

    input_dataset = xr.Dataset({"state_t": (("sample", "ncol", "lev"), input_data)})
    target_dataset = xr.Dataset({"ptend_t": (("sample", "ncol", "lev"), target_data)})

    path_to_grid_info = (
        "/home/users/bradlesc/projects/ClimSim/grid_info/ClimSim_low-res_grid-info.nc"
    )

    input_dataset_out, target_dataset_out = dataset._spatial_selection(
        input_dataset, target_dataset, path_to_grid_info, spatial_selection_method='northern_hemisphere'
    )
    # Check that only northern hemisphere latitudes are selected
    grid_info = xr.open_dataset(path_to_grid_info)
    latitudes = grid_info["lat"]
    northern_hemisphere_indices = np.where(latitudes.values > 0)[0]
    assert input_dataset_out.sizes["ncol"] == len(northern_hemisphere_indices)
    assert target_dataset_out.sizes["ncol"] == len(northern_hemisphere_indices)





def test_climsim_from_raw_dataset_unit_test_normalise_dataset():

    dataset = data_preparation.ClimSimFromRawDataset(
        mode="train",
        dataset_testing_type="qt",
        dataset_cfg=None,
        model=None,
        unit_test_specific_methods=True,
    )

    # Create dummy input and target datasets
    singleinput_data = np.random.normal(loc=300, scale=10, size=(1000, 384))
    # repeate along ncol dimension to create 3D
    input_data = np.repeat(singleinput_data[:, :, np.newaxis], 60, axis=2)
    taget_data = np.ones((1000, 384, 60))
    input_dataset = xr.Dataset({"state_t": (("sample", "ncol", "lev"), input_data)})
    target_dataset = xr.Dataset({"ptend_t": (("sample", "ncol", "lev"), taget_data)})

    normalised_input_ds, normalised_target_ds, normalisation_stats = (
        dataset._normalise_datasets(
            normalisation_stats=None,
            input_ds=input_dataset,
            target_ds=target_dataset,
            output_scale_file_path=output_scale_file_path,
            v1_targets=["ptend_t"],
        )
    )

    # Check that normalisation stats is returned
    assert normalisation_stats is not None
    # Check that normalised datasets have same variables
    assert set(normalised_input_ds.data_vars) == set(input_dataset.data_vars)

    # check mean of each variable in normalised_input_ds is approximately 0
    for var in normalised_input_ds.data_vars:
        mean_val = normalised_input_ds[var].mean(dim=["sample", "ncol"])
        assert np.all(
            abs(mean_val) < 1e-6
        ), f"Mean of {var} is not approximately 0 after normalisation, got {mean_val}"


def test_climsim_from_raw_dataset_unit_test_normalisation_real_data():
    base_dir = Path(__file__).resolve().parents[1]
    data_path = os.path.join(
        base_dir, "unit_test_sets", "dummy_low_res_climsim/", "filename_testing"
    )

    dataset_config = DictConfig(
        {
            "base_folder_path": data_path,
            "target_years": ["0001"],
            "target_months": ["02"],
            "sample_rate": 1,
        }
    )
    dataset = data_preparation.ClimSimFromRawDataset(
        mode="train",
        dataset_testing_type="qt",
        dataset_cfg=dataset_config,
        model=None,
        unit_test_specific_methods=True,
    )
    input_filenames, target_filenames = dataset._get_dataset_filenames(
        base_folder_path=dataset_config.base_folder_path,
        target_years=dataset_config.target_years,
        target_months=dataset_config.target_months,
    )
    input_ds, target_ds = dataset._combine_datasets(
        input_filenames,
        target_filenames,
        v1_inputs=v1_inputs,
        v1_targets=v1_targets,
    )
    normalised_input_ds, normalised_target_ds, normalisation_stats = (
        dataset._normalise_datasets(
            normalisation_stats=None,
            input_ds=input_ds,
            target_ds=target_ds,
            v1_targets=v1_targets,
            output_scale_file_path=output_scale_file_path,
        )
    )
    # check mean of each variable in normalised_input_ds is approximately 0
    for var in normalised_input_ds.data_vars:
        mean_val = normalised_input_ds[var].mean(dim=["sample", "ncol"])
        assert np.all(
            abs(mean_val) < 1e-6
        ), f"Mean of {var} is not approximately 0 after normalisation, got {mean_val}"


def test_climsim_from_raw_dataset_unit_test_prepare_data():
    base_dir = Path(__file__).resolve().parents[1]
    data_path = os.path.join(
        base_dir, "unit_test_sets", "dummy_low_res_climsim/", "filename_testing"
    )

    dataset_config = DictConfig(
        {
            "base_folder_path": data_path,
            "target_years": ["0001"],
            "target_months": ["02"],
            "sample_rate": 1,
        }
    )
    dataset = data_preparation.ClimSimFromRawDataset(
        mode="train",
        dataset_testing_type="qt",
        dataset_cfg=dataset_config,
        model=None,
        unit_test_specific_methods=True,
    )
    input_filenames, target_filenames = dataset._get_dataset_filenames(
        base_folder_path=dataset_config.base_folder_path,
        target_years=dataset_config.target_years,
        target_months=dataset_config.target_months,
    )
    input_ds, target_ds = dataset._combine_datasets(
        input_filenames,
        target_filenames,
        v1_inputs=v1_inputs,
        v1_targets=v1_targets,
    )
    input_tensor, target_tensor = dataset._prepare_data(
        model_name=None,
        input_ds=input_ds,
        target_ds=target_ds,
        v1_inputs=v1_inputs,
        v1_targets=v1_targets,
    )
    assert isinstance(input_tensor, torch.Tensor)
    assert isinstance(target_tensor, torch.Tensor)
    assert input_tensor.shape[0] == target_tensor.shape[0]
    assert input_tensor.shape[1] == 124
    assert target_tensor.shape[1] == 128


def test_climsim_from_raw_dataset_init_sample_data():
    base_dir = Path(__file__).resolve().parents[1]
    data_path = os.path.join(
        base_dir, "unit_test_sets", "dummy_low_res_climsim/", "filename_testing"
    )
    dataset_config = DictConfig(
        {
            "base_folder_path": data_path,
            "target_years": ["0001"],
            "target_months": ["02"],
            "dataset_testing_num_files": {"full": 1},
            "v1_inputs": v1_inputs,
            "v1_targets": v1_targets,
            "output_scale_file_path": output_scale_file_path,
            "group_method": False,
            "path_to_grid_info": "/home/users/bradlesc/projects/ClimSim/grid_info/ClimSim_low-res_grid-info.nc",
            "spatial_selection_method": False,
        }
    )

    dataset = data_preparation.ClimSimFromRawDataset(
        mode="train",
        dataset_testing_type="full",
        dataset_cfg=dataset_config,
        model=None,
        unit_test_specific_methods=False,
    )

    assert isinstance(dataset.input, torch.Tensor)
    assert isinstance(dataset.target, torch.Tensor)
    assert dataset.input.shape[0] == dataset.target.shape[0]
    assert dataset.input.shape[1] == 124
    assert dataset.target.shape[1] == 128

    x, y = dataset[0]
    assert torch.equal(x, dataset.input[0])
    assert torch.equal(y, dataset.target[0])

    assert len(x) == 124
    assert len(y) == 128


def test_climsim_from_raw_dataset_init_sample_data_train_and_test():
    base_dir = Path(__file__).resolve().parents[1]
    data_path = os.path.join(
        base_dir, "unit_test_sets", "dummy_low_res_climsim/", "filename_testing"
    )
    dataset_config = DictConfig(
        {
            "base_folder_path": data_path,
            "target_years": ["0001"],
            "target_months": ["02"],
            "dataset_testing_num_files": {"full": 1},
            "v1_inputs": v1_inputs,
            "v1_targets": v1_targets,
            "output_scale_file_path": output_scale_file_path,
            "group_method": False,
            "path_to_grid_info": "/home/users/bradlesc/projects/ClimSim/grid_info/ClimSim_low-res_grid-info.nc",
            "spatial_selection_method": False,
        }
    )

    dataset = data_preparation.ClimSimFromRawDataset(
        mode="train",
        dataset_testing_type="full",
        dataset_cfg=dataset_config,
        model=None,
        unit_test_specific_methods=False,
    )

    test_dataset = data_preparation.ClimSimFromRawDataset(
        mode="test",
        dataset_testing_type="full",
        dataset_cfg=dataset_config,
        model=None,
        unit_test_specific_methods=False,
        normalisation_stats=dataset.normalisation_stats,
    )

    assert isinstance(dataset.input, torch.Tensor)
    assert isinstance(dataset.target, torch.Tensor)
    assert isinstance(test_dataset.input, torch.Tensor)
    assert isinstance(test_dataset.target, torch.Tensor)
    assert dataset.input.shape[0] == dataset.target.shape[0]
    assert test_dataset.input.shape[0] == test_dataset.target.shape[0]
    assert dataset.input.shape[1] == 124
    assert dataset.target.shape[1] == 128
    assert test_dataset.input.shape[1] == 124
    assert test_dataset.target.shape[1] == 128

    with pytest.raises(ValueError):
        _ = data_preparation.ClimSimFromRawDataset(
            mode="test",
            dataset_testing_type="full",
            dataset_cfg=dataset_config,
            model=None,
            unit_test_specific_methods=False,
            normalisation_stats=None,
        )


def test_climsim_from_raw_dataset_init_sample_data_remove_high_altitude_specific_humidity_levels():
    base_dir = Path(__file__).resolve().parents[1]
    data_path = os.path.join(
        base_dir, "unit_test_sets", "dummy_low_res_climsim/", "filename_testing"
    )
    dataset_config = DictConfig(
        {
            "base_folder_path": data_path,
            "target_years": ["0001"],
            "target_months": ["02"],
            "dataset_testing_num_files": {"full": 1},
            "v1_inputs": v1_inputs,
            "v1_targets": v1_targets,
            "output_scale_file_path": output_scale_file_path,
            "group_method": False,
            "remove_high_altitude_specific_humidity_levels": 2,
            "path_to_grid_info": "/home/users/bradlesc/projects/ClimSim/grid_info/ClimSim_low-res_grid-info.nc",
            "spatial_selection_method": False,
        }
    )

    dataset = data_preparation.ClimSimFromRawDataset(
        mode="train",
        dataset_testing_type="full",
        dataset_cfg=dataset_config,
        model=None,
        unit_test_specific_methods=False,
    )

    assert isinstance(dataset.input, torch.Tensor)
    assert isinstance(dataset.target, torch.Tensor)
    assert dataset.input.shape[0] == dataset.target.shape[0]
    assert (
        dataset.input.shape[1]
        == 124 - dataset_config.remove_high_altitude_specific_humidity_levels
    )
    assert (
        dataset.target.shape[1]
        == 128 - dataset_config.remove_high_altitude_specific_humidity_levels
    )

    x, y = dataset[0]
    assert torch.equal(x, dataset.input[0])
    assert torch.equal(y, dataset.target[0])

    assert len(x) == 124 - dataset_config.remove_high_altitude_specific_humidity_levels
    assert len(y) == 128 - dataset_config.remove_high_altitude_specific_humidity_levels

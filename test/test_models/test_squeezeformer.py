import torch
import torch.nn as nn
import pytest
import models
from omegaconf import DictConfig
from pathlib import Path
import os
import data_preparation


def test_squeezeformer_initialization():
    in_dim = 10
    embed_dim = 32
    out_dim = 5
    head_dim = 16
    levels = 50

    model = models.SqueezeFormer(
        in_dim=in_dim,
        embed_dim=embed_dim,
        head_dim=head_dim,
        levels=levels,
        out_dim=out_dim,
    )
    assert model.in_dim == in_dim
    assert model.embed_dim == embed_dim
    assert model.out_dim == out_dim


def test_squeezeformer_forward_pass():
    in_dim = 6
    embed_dim = 384
    out_dim = 8
    batch_size = 4
    levels = 60
    head_dim = 128

    model = models.SqueezeFormer(
        in_dim=in_dim,
        embed_dim=embed_dim,
        head_dim=head_dim,
        out_dim=out_dim,
        levels=levels,
    )
    input_tensor = torch.randn(batch_size, levels, in_dim)
    output_tensor = model(input_tensor)

    assert output_tensor[0].shape == (batch_size, levels, out_dim)


def test_Conv1DBlockSqueezeFormer():
    block = models.Conv1DBlockSqueezeFormer(
        in_features=44, levels=60, kernel_size=15, dilation_rate=1, expand_ratio=4
    )
    input_tensor = torch.randn(2, 60, 44)  # (batch, levels, features)
    output_tensor = block(input_tensor)

    assert output_tensor.shape == (2, 60, 44)  # (batch, levels, in_features)


def test_ECA():
    input_tensor = torch.randn(2, 60, 768)  # (batch, levels, channels)
    eca = models.ECA(levels=60, kernel_size=5)
    output_tensor = eca(input_tensor)

    assert output_tensor.shape == (2, 60, 768)


def test_TransformerEncoder():
    embed_dim = 128
    num_heads = 8
    feedforward_dim = 512
    batch_size = 2
    levels = 30

    encoder = models.TransformerEncoder(
        embed_dim=embed_dim, num_heads=num_heads, feedforward_dim=feedforward_dim
    )
    input_tensor = torch.randn(batch_size, levels, embed_dim)
    output_tensor = encoder(input_tensor)

    assert output_tensor.shape == (batch_size, levels, embed_dim)


def test_load_squeezeformer():
    model_params = DictConfig(
        {
            "embed_dim": 384,
            "head_dim": 2048,
            "lr": 0.0005,
            "batch_size": 1024,
            "optimizer": "AdamW",
            "scheduler": {
                "name": "lambda_lr",  # or 'lambda_lr' if it's a string
                "num_warmup_steps": 0,
                "warmup_method": "exp",
                "num_training_steps": 10,
                "num_cycles": 0.5,
            },
        }
    )
    data_params = DictConfig(
        {
            "in_dim": 6,
            "out_dim": 8,
            "levels": 60,
        }
    )

    model = models.select_model(
        model_name="squeezeformer",
        model_params=model_params,
        data_params=data_params,
    )
    assert isinstance(model, models.LightningWrapper)
    assert isinstance(model.model, models.SqueezeFormer)


def test_squeezeformer_proper_setup():
    model_params = DictConfig(
        {
            "embed_dim": 384,
            "head_dim": 2048,
            "lr": 0.0005,
            "batch_size": 4,
            "optimizer": "AdamW",
            "scheduler": {
                "name": "lambda_lr",  # or 'lambda_lr' if it's a string
                "num_warmup_steps": 0,
                "warmup_method": "exp",
                "num_training_steps": 10,
                "num_cycles": 0.5,
            },
        }
    )
    data_params = DictConfig(
        {
            "input_dim": 6,
            "output_dim": 10,
            "levels": 60,
        }
    )

    model = models.select_model(
        model_name="squeezeformer",
        model_params=model_params,
        data_params=data_params,
    )

    data = torch.randn(
        model_params.batch_size, data_params.levels, data_params.input_dim
    )

    output = model(data)

    assert output.shape == (model_params.batch_size, 113)


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


def test_squeezeformer_on_climsim_from_raw():
    base_dir = Path(__file__).resolve().parents[1]
    data_path = os.path.join(
        base_dir, "unit_test_sets", "dummy_low_res_climsim/", "filename_testing"
    )
    output_scale_file_path = "/home/users/bradlesc/projects/ClimSim/preprocessing/normalizations/outputs/output_scale.nc"
    dataset_config = DictConfig(
        {
            "base_folder_path": data_path,
            "target_years": ["0001"],
            "target_months": ["02"],
            "dataset_testing_sample_rates": {"full": 1},
            "v1_inputs": v1_inputs,
            "v1_targets": v1_targets,
            "output_scale_file_path": output_scale_file_path,
            "group_method": False,
            "remove_high_altitude_specific_humidity_levels": 15,
            "path_to_grid_info": "/home/users/bradlesc/projects/ClimSim/grid_info/ClimSim_low-res_grid-info.nc",
        }
    )

    dataset = data_preparation.ClimSimFromRawDataset(
        mode="train",
        dataset_testing_type="full",
        dataset_cfg=dataset_config,
        model='squeezeformer',
        unit_test_specific_methods=False,
    )

    model_params = DictConfig(
        {
            "embed_dim": 384,
            "head_dim": 2048,
            "lr": 0.0005,
            "batch_size": 4,
            "optimizer": "AdamW",
            "scheduler": {
                "name": "lambda_lr",  # or 'lambda_lr' if it's a string
                "num_warmup_steps": 0,
                "warmup_method": "exp",
                "num_training_steps": 10,
                "num_cycles": 0.5,
            },
        }
    )
    data_params = DictConfig(
        {
            "input_dim": 6,
            "output_dim": 10,
            "levels": 60,
        }
    )

    model = models.select_model(
        model_name="squeezeformer",
        model_params=model_params,
        data_params=data_params,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=model_params.batch_size, shuffle=True
    )
    sample_input, sample_output = next(iter(data_loader))
    print(f"Sample input shape: {sample_input.shape}")
    output = model(sample_input)
    assert output.shape == (model_params.batch_size, 113)

    
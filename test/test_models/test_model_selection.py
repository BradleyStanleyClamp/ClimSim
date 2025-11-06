"""
Script to test model selection functionalities
"""

import models
from omegaconf import DictConfig
import torch
import pytest
from pathlib import Path

def test_select_base_model_mlp():
    
    model_name = "mlp"
    model_params = DictConfig({
        "lr": 0.001,
        "batch_size": 512,
        "hidden_dims": [768, 640, 512, 640, 640],
        "activation": "relu"
    })

    data_params = DictConfig({"input_dim": 100, "output_dim": 10})

    model = models.select_base_model(model_name, model_params, data_params)

    assert isinstance(model, models.mlp.MLP)
    assert model.activation == torch.nn.functional.relu

def test_select_base_model_yus_mlp():
    
    model_name = "yus_mlp"
    model_params = DictConfig({"lr": 0.001,
                    "batch_size": 512,
                    "hidden_dims": [768, 640, 512, 640, 640],
                    "activation": "leaky_relu"})

    data_params = DictConfig({"input_dim": 100, "output_dim": 10})

    model = models.select_base_model(model_name, model_params, data_params)

    assert isinstance(model, models.yus_mlp.YusMLP)
    assert model.activation == torch.nn.functional.leaky_relu


def test_select_model_mlp():
    
    model_name = "mlp"
    model_params = DictConfig({
        "lr": 0.001,
        "batch_size": 512,
        "hidden_dims": [768, 640, 512, 640, 640],
        "activation": "relu",
        "optimizer": "Adam",
        "scheduler": None
    })

    data_params = DictConfig({"input_dim": 100, "output_dim": 10})

    model = models.select_model(model_name, model_params, data_params)

    assert isinstance(model, models.LightningWrapper)
    assert isinstance(model.model, models.mlp.MLP)
    assert model.optimizer == 'Adam'
    assert model.scheduler_cfg is None

def test_select_model_yus_mlp():
    
    model_name = "yus_mlp"
    model_params = DictConfig({
        "lr": 0.001,
        "batch_size": 512,
        "hidden_dims": [768, 640, 512, 640, 640],
        "activation": "leaky_relu",
        "optimizer": "RAdam",
        "scheduler": {'name': 'cyclic'}
    })

    data_params = DictConfig({"input_dim": 100, "output_dim": 10})

    model = models.select_model(model_name, model_params, data_params)

    assert isinstance(model, models.LightningWrapper)
    assert isinstance(model.model, models.yus_mlp.YusMLP)
    assert model.optimizer == 'RAdam'
    assert model.scheduler_cfg.name == 'cyclic'


def test_select_base_model_invalid():
    model_name = "invalid_model"
    model_params = DictConfig({})
    data_params = DictConfig({})

    with pytest.raises(ValueError) as excinfo:
        models.select_base_model(model_name, model_params, data_params)
    
    assert "Model invalid_model not recognized." in str(excinfo.value)

def test_select_optimizer():
    adam_optimizer = models.select_optimizer("Adam")
    radam_optimizer = models.select_optimizer("RAdam")

    assert adam_optimizer == torch.optim.Adam
    assert radam_optimizer == torch.optim.RAdam
    with pytest.raises(ValueError) as excinfo:
        models.select_optimizer("InvalidOptimizer")
    assert "Optimizer InvalidOptimizer not recognized." in str(excinfo.value)

def test_load_model_from_checkpoint(mlp_checkpoint_path: str = "../unit_test_sets/trained_model_log/mlp/test_mlp_checkpoint.ckpt"):
    model_name = "mlp"
    model_params = DictConfig({
        "lr": 0.001,
        "batch_size": 512,
        "hidden_dims": [256, 128],
        "activation": "relu",
        "optimizer": "Adam",
        "scheduler": 'None'
    })

    data_params = DictConfig({"input_dim": 124, "output_dim": 128})



    mlp_checkpoint_path = Path(mlp_checkpoint_path)
    if not mlp_checkpoint_path.is_absolute():
        mlp_checkpoint_path = (Path(__file__).resolve().parent / mlp_checkpoint_path).resolve()

    # Load the model from the checkpoint
    loaded_model = models.load_model_from_checkpoint(mlp_checkpoint_path, model_name, model_params, data_params)

    assert isinstance(loaded_model, models.LightningWrapper)
    assert isinstance(loaded_model.model, models.mlp.MLP)

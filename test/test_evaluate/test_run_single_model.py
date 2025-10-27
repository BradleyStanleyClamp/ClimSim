

import evaluate
from omegaconf import DictConfig
import models
import lightning as L
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np

def test_get_model_from_config():


    model_name = "mlp"
    model_path = "test/unit_test_sets/trained_model_log/"
    

    dataset_cfg = DictConfig({
        "input_dim": 124,
        "output_dim": 128
    })

    model = evaluate.get_model_from_config(model_name, model_path, dataset_cfg)
    assert model is not None
    assert isinstance(model, models.LightningWrapper)
    assert isinstance(model.model, models.MLP)

def test_evaluate_model_on_dataset():
    fake_lightning_module = models.LightningWrapper(
        model=models.MLP(input_dim=10, hidden_dims=[20, 20], output_dim=5)
    )
    class FakeDataset(Dataset):
        def __init__(self, n_samples=50, input_dim=10, output_dim=5):
            super().__init__()
            self.x = torch.randn(n_samples, input_dim)
            self.y = torch.randn(n_samples, output_dim)

        def __len__(self):
            return self.x.size(0)

        def __getitem__(self, idx):
            return self.x[idx], self.y[idx]

    fake_dataset = FakeDataset(n_samples=50, input_dim=10, output_dim=5)
    dataloader = DataLoader(fake_dataset, batch_size=10, shuffle=False)
    predictions = evaluate.evaluate_model_on_dataset(fake_lightning_module, dataloader)
    assert isinstance(predictions, np.ndarray)

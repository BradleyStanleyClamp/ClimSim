
from models import ClimSimUNet, select_model
import torch
import lightning as L
from omegaconf import DictConfig
def test_clim_sim_unet_init():
    model = ClimSimUNet()
    assert model is not None
    assert isinstance(model, ClimSimUNet)

def test_clim_sim_unet_forward():
    model = ClimSimUNet()
    batch_size = 2
    input_tensor = torch.randn(batch_size, 6, 64)  # (batch_size, features, length)
    output = model(input_tensor)
    print(output.shape)
    assert output is not None
    assert output.shape[0] == batch_size
    assert output.shape[1] == 128
    # assert output.shape[1] == 10
    # assert output.shape[2] == 60

def test_select_unet_model():
    model_params = DictConfig({'lr': 1e-4,
        'batch_size': 1024,
        'optimizer': 'Adam',
        'scheduler': {
            'name': 'step',
            'step_size': 3,  # epochs
            'gamma': 0.5 },
    })
    data_params = {}

    model = select_model('climsim_unet', model_params, data_params)

    assert model is not None
    assert isinstance(model, L.LightningModule)
    assert isinstance(model.model, ClimSimUNet)
    assert model.optimizer == 'Adam'
    assert model.scheduler_cfg['name'] == 'step'




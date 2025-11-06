
from models import ClimSimUNet
import torch


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
    # assert output.shape[2] == input_tensor.shape[2]
    # Add more specific assertions about output shape if known



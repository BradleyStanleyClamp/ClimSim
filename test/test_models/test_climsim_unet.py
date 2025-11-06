
from models import ClimSimUNet
import torch


def test_clim_sim_unet_init():
    model = ClimSimUNet()
    assert model is not None
    assert isinstance(model, ClimSimUNet)

def test_clim_sim_unet_forward():
    model = ClimSimUNet()
    batch_size = 2
    input_tensor = torch.randn(batch_size, 48, 64)  # (batch_size, features, length)
    output = model(input_tensor)
    assert output is not None
    assert output.shape[0] == batch_size
    assert output.shape[1] == 13
    assert output.shape[2] == input_tensor.shape[2]
    # Add more specific assertions about output shape if known


# def test_unet_block_no_attention():
#     block = UNetBlockNoAttention(in_channels=128, out_channels=128, use_skip=True)
#     batch_size = 2
#     input_tensor = torch.randn(batch_size, 128, 64)
#     output, skip = block(input_tensor)
#     assert output is not None
#     assert skip is not None
#     assert output.shape[1] == 128
#     assert skip.shape[1] == 128
#     assert output.shape[2] == input_tensor.shape[2]
#     assert skip.shape[2] == input_tensor.shape[2]
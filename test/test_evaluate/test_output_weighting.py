import torch

import evaluate
import xarray as xr


def test_output_weighting_reshape_outputs():
    output_weighting = evaluate.OutputWeighting(cfg=None)

    x = torch.randn(1920, 119)  # Example input tensor
    reshaped_ds = output_weighting._reshape_outputs(x)

    assert isinstance(reshaped_ds, xr.Dataset)
    assert "ptend_t" in reshaped_ds
    assert "ptend_q0001" in reshaped_ds
    assert "netsw" in reshaped_ds
    assert "flwds" in reshaped_ds
    assert reshaped_ds["ptend_t"].shape == (
        1920 // 192,
        192,
        45,
    )  # Adjusted for top levels removed
    assert reshaped_ds["ptend_q0001"].shape == (
        1920 // 192,
        192,
        45,
    )  # Adjusted for top levels removed
    assert reshaped_ds["netsw"].shape == (1920 // 192, 192)
    assert reshaped_ds["flwds"].shape == (1920 // 192, 192)

    assert x[0, 0].item() == reshaped_ds["ptend_t"].values.flatten()[0]

def test_output_weighting_undo_output_scaling(tmp_path):

    output_weighting = evaluate.OutputWeighting(cfg=None)

    # Create a mock scaling dataset
    scale_data = {
        "ptend_t": (("lev"), [1.0] * 45),
        "ptend_q0001": (("lev"), [2.0] * 45),
        "netsw": ((), 3.0),
        "flwds": ((), 4.0),
    }
    scale_ds = xr.Dataset(scale_data)
    scale_file = tmp_path / "output_scale.nc"
    scale_ds.to_netcdf(scale_file)

    # Mock the config to point to the temporary scale file
    output_weighting.cfg = type("cfg", (), {})()
    output_weighting.cfg.dataset = type("dataset", (), {})()
    output_weighting.cfg.dataset.output_scale_file_path = str(scale_file)
    output_weighting.cfg.dataset.v1_targets = [
        "ptend_t",
        "ptend_q0001",
        "netsw",
        "flwds",
    ]

    # Create a mock scaled dataset
    scaled_data = {
        "ptend_t": (("sample", "lat", "lev"), [[[10.0] * 45] * 192]),
        "ptend_q0001": (("sample", "lat", "lev"), [[[20.0] * 45] * 192]),
        "netsw": (("sample", "lat"), [[30.0] * 192]),
        "flwds": (("sample", "lat"), [[40.0] * 192]),
    }
    scaled_ds = xr.Dataset(scaled_data)

    unscaled_ds = output_weighting.undo_output_scaling(scaled_ds)

    assert isinstance(unscaled_ds, xr.Dataset)
    assert unscaled_ds["ptend_t"].values[0, 0, 0] == 10.0 / 1.0
    assert unscaled_ds["ptend_q0001"].values[0, 0, 0] == 20.0 / 2.0
    assert unscaled_ds["netsw"].values[0, 0] == 30.0 / 3.0
    assert unscaled_ds["flwds"].values[0, 0] == 40.0 / 4.0

def test_vertical_weighting():
    # TODO
    pass
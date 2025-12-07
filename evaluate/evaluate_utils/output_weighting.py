"""
Script to transform the output data such that it is ready for evaluation.
The code is based on the `output_weighting` function in ClimSim's
`data_utils.py`.
"""

import logging
import xarray as xr
import torch
from omegaconf import DictConfig
import numpy as np


class OutputWeighting:
    def __init__(
        self, cfg: DictConfig, num_latlon: int = 192, top_levels_to_remove: int = 15, unit_test: bool = False
    ):
        """
        Initialize the OutputWeighting class.
        """
        self.cfg = cfg
        self.num_latlon = num_latlon
        self.top_levels_to_remove = top_levels_to_remove

        self.max_t_level_index = 45
        self.min_sh_level_index = 45
        self.max_sh_level_index = 90

        if unit_test:
            return
        self.grid_info = xr.open_dataset(self.cfg.dataset.path_to_grid_info)
    def weight(
        self, data: torch.Tensor, testset: torch.utils.data.Dataset
    ) -> xr.Dataset:
        """
        Apply output weighting to the dataset.

        Args:
            ds (xr.Dataset): The input dataset.
            testset: The test dataset containing normalization stats and input data.

        Uses:
            self._reshape_outputs
            self.undo_output_scaling
            self._vertical_weighting
            self._weight_by_area
            self._unit_conversion

        Returns:
            xr.Dataset: The weighted dataset.
        """
        logging.info("Applying output weighting...")
        ds = self._reshape_outputs(data)
        logging.info(f"Reshaped outputs to dataset")
        ds = self.undo_output_scaling(ds)
        logging.info("Undid output scaling")
        ds = self._vertical_weighting(ds, testset)
        logging.info("Applied vertical weighting")
        ds = self._weight_by_area(ds)
        logging.info("Applied area weighting")
        ds = self._unit_conversion(ds)
        logging.info("Applied unit conversion")

        return ds

    def _reshape_outputs(self, data: torch.Tensor) -> xr.Dataset:
        """
        Reshape the model outputs to match the original xr dataset structure for ease of multivariable operations.

        Args:
            data (torch.Tensor): The model output tensor (batch_size, num_features).

        Uses:
            self.max_t_level_index (int): Maximum index for temperature levels.
            self.min_sh_level_index (int): Minimum index for specific humidity levels.
            self.max_sh_level_index (int): Maximum index for specific humidity levels.
            self.num_latlon (int): Number of latitude-longitude points.


        Returns:
            xr.Dataset: The reshaped dataset.
        """
        num_samples = data.shape[0]

        ptend_t = data[:, : self.max_t_level_index].reshape(
            (
                int(num_samples / self.num_latlon),
                self.num_latlon,
                self.max_t_level_index,
            )
        )
        ptend_q0001 = data[
            :, self.min_sh_level_index : self.max_sh_level_index
        ].reshape(
            (
                int(num_samples / self.num_latlon),
                self.num_latlon,
                self.max_sh_level_index - self.min_sh_level_index,
            )
        )
        netsw = data[:, self.max_sh_level_index].reshape(
            (int(num_samples / self.num_latlon), self.num_latlon)
        )
        flwds = data[:, self.max_sh_level_index + 1].reshape(
            (int(num_samples / self.num_latlon), self.num_latlon)
        )
        precsc = data[:, self.max_sh_level_index + 2].reshape(
            (int(num_samples / self.num_latlon), self.num_latlon)
        )
        precc = data[:, self.max_sh_level_index + 3].reshape(
            (int(num_samples / self.num_latlon), self.num_latlon)
        )
        sols = data[:, self.max_sh_level_index + 4].reshape(
            (int(num_samples / self.num_latlon), self.num_latlon)
        )
        soll = data[:, self.max_sh_level_index + 5].reshape(
            (int(num_samples / self.num_latlon), self.num_latlon)
        )
        solsd = data[:, self.max_sh_level_index + 6].reshape(
            (int(num_samples / self.num_latlon), self.num_latlon)
        )
        solld = data[:, self.max_sh_level_index + 7].reshape(
            (int(num_samples / self.num_latlon), self.num_latlon)
        )

        ds = xr.Dataset(
            data_vars={
                "ptend_t": (("sample", "ncol", "lev"), ptend_t),
                "ptend_q0001": (("sample", "ncol", "lev"), ptend_q0001),
                "netsw": (("sample", "ncol"), netsw),
                "flwds": (("sample", "ncol"), flwds),
                "precsc": (("sample", "ncol"), precsc),
                "precc": (("sample", "ncol"), precc),
                "sols": (("sample", "ncol"), sols),
                "soll": (("sample", "ncol"), soll),
                "solsd": (("sample", "ncol"), solsd),
                "solld": (("sample", "ncol"), solld),
            },
            coords={
                "sample": range(ptend_t.shape[0]),
                "ncol": range(self.num_latlon),
                "lev": range(ptend_t.shape[2]),
            },
            attrs={
                "description": "Model predictions from YUS MLP model",
            },
        )

        return ds

    def undo_output_scaling(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Undoes the output scaling applied during normalisation step.
        Args:
            ds (xr.Dataset): The scaled dataset.

        Uses:
            self.cfg.dataset.output_scale_file_path (str): Path to the output scale file.
            self.cfg.dataset.v1_targets (list): List of target variable names.

        Returns:
            xr.Dataset: The unscaled dataset.

        """
        # Get output scaling factors
        out_scale = xr.open_dataset(self.cfg.dataset.output_scale_file_path)

        # Select only the variables that are in the targets
        out_scale = out_scale[list(self.cfg.dataset.v1_targets)]

        assert (
            self.max_sh_level_index - self.min_sh_level_index == self.max_t_level_index
        ), "The number of levels for specific humidity and temperature do not match. "

        #  Trim the top levels from out_scale to match the model output levels
        out_scale = out_scale.isel(lev=slice(0, self.max_t_level_index))

        # Build a rename mapping to match the variable names in the dataset
        prefix = "cam_out_"
        self.rename_dict = {
            var: var[len(prefix) :].lower()
            for var in out_scale.data_vars
            if var.startswith(prefix)
        }

        out_scale = out_scale.rename(self.rename_dict)

        # Apply the scaling factors
        unscaled_data = ds / out_scale

        return unscaled_data

    def _vertical_weighting(
        self, ds: xr.Dataset, testset: torch.utils.data.Dataset
    ) -> xr.Dataset:
        """
        Apply vertical weighting to the dataset.

        Args:
            ds (xr.Dataset): The input dataset.
            testset: The test dataset containing normalization stats and input data.

        Uses:
            self.cfg.dataset.path_to_grid_info (str): Path to the grid information file.
            self.num_latlon (int): Number of latitude-longitude points.
            self.max_t_level_index (int): Maximum index for temperature levels.
            self.grid_info: Grid information dataset.

        Returns:
            xr.Dataset: The vertically weighted dataset.
        """

        self.grav = 9.80616  # acceleration of gravity ~ m/s^2

        state_ps = testset.normalised_input_ds["state_ps"]

        # Un normlaise surface pressure
        state_ps = (
            state_ps * testset.normalisation_stats["range"]["state_ps"]
            + testset.normalisation_stats["mean"]["state_ps"]
        )

        state_ps = np.reshape(state_ps, (-1, self.num_latlon)).to_numpy()

        # Fixed pressure portion
        pressure_grid_p1 = np.array(self.grid_info["P0"] * self.grid_info["hyai"])[
            :, np.newaxis, np.newaxis
        ]

        # Terrain following portion
        pressure_grid_p2 = (
            self.grid_info["hybi"].values[:, np.newaxis, np.newaxis]
            * state_ps[np.newaxis, :, :]
        )

        pressure_grid = pressure_grid_p1 + pressure_grid_p2

        dp = pressure_grid[1:, :, :] - pressure_grid[:-1, :, :]
        dp = dp.transpose(1, 2, 0)

        dp_reduced = dp[:, :, : self.max_t_level_index]

        ds["ptend_t"] = ds["ptend_t"] * (dp_reduced / self.grav)
        ds["ptend_q0001"] = ds["ptend_q0001"] * (dp_reduced / self.grav)

        return ds

    def _weight_by_area(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Weight the dataset by area.

        Args:
            ds (xr.Dataset): The input dataset.

        Uses:
            self.num_latlon (int): Number of latitude-longitude points.
            self.grid_info: Grid information dataset.

        Returns:
            xr.Dataset: The area-weighted dataset.
        """
        self.grid_info["area_wgt"] = self.grid_info["area"] / self.grid_info[
            "area"
        ].mean(dim="ncol")

        area_wgt = self.grid_info["area_wgt"].isel(ncol=self.grid_info["lat"] > 0)

        weighted_ds = ds * area_wgt

        return weighted_ds

    def _unit_conversion(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Convert units of the dataset variables to standard evaluation units.

        Args:
            ds (xr.Dataset): The input dataset.

        Returns:
            xr.Dataset: The dataset with converted units.
        """
        cp = 1.00464e3  # specific heat of dry air   ~ J/kg/K
        lv = 2.501e6  # latent heat of evaporation ~ J/kg
        rho_h20 = 1.0e3  # density of fresh water     ~ kg/m^ 3

        target_energy_conv = xr.Dataset(
            data_vars={
                "ptend_t": cp,
                "ptend_q0001": lv,
                "ptend_q0002": lv,
                "ptend_q0003": lv,
                "ptend_qn": lv,
                "ptend_wind": None,
                "cam_out_NETSW": 1.0,
                "cam_out_FLWDS": 1.0,
                "cam_out_PRECSC": lv * rho_h20,
                "cam_out_PRECC": lv * rho_h20,
                "cam_out_SOLS": 1.0,
                "cam_out_SOLL": 1.0,
                "cam_out_SOLSD": 1.0,
                "cam_out_SOLLD": 1.0,
            }
        )
        target_energy_conv = target_energy_conv[list(self.cfg.dataset.v1_targets)]
        target_energy_conv = target_energy_conv.rename(self.rename_dict)
        unit_converted = ds * target_energy_conv

        return unit_converted

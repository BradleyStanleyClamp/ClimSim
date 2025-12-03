"""
Class for calculating various evaluation metrics for model predictions. Code taken from ClimSim original code.

"""

import numpy as np


class MetricsCalculator:
    def __init__(self, num_latlon):
        self.num_latlon = num_latlon

        self.metrics_dict = {
            "MAE": self.calc_MAE,
            "RMSE": self.calc_RMSE,
            "R2": self.calc_R2,
            # 'CRPS': self.calc_CRPS,
            "bias": self.calc_bias,
        }

    def calc_MAE(self, pred, target, avg_grid=True):
        """
        calculate 'globally averaged' mean absolute error
        for vertically-resolved variables, shape should be time x grid x level
        for scalars, shape should be time x grid

        returns vector of length level or 1
        """
        assert pred.shape[1] == self.num_latlon
        assert pred.shape == target.shape
        mae = np.abs(pred - target).mean(axis=0)
        if avg_grid:
            return mae.mean(axis=0)  # we decided to average globally at end
        else:
            return mae

    def calc_RMSE(self, pred, target, avg_grid=True):
        """
        calculate 'globally averaged' root mean squared error
        for vertically-resolved variables, shape should be time x grid x level
        for scalars, shape should be time x grid

        returns vector of length level or 1
        """
        assert pred.shape[1] == self.num_latlon
        assert pred.shape == target.shape
        sq_diff = (pred - target) ** 2
        rmse = np.sqrt(sq_diff.mean(axis=0))  # mean over time
        if avg_grid:
            return rmse.mean(axis=0)  # we decided to separately average globally at end
        else:
            return rmse

    def calc_R2(self, pred, target, avg_grid=True):
        """
        calculate 'globally averaged' R-squared
        for vertically-resolved variables, input shape should be time x grid x level
        for scalars, input shape should be time x grid

        returns vector of length level or 1
        """
        assert pred.shape[1] == self.num_latlon
        assert pred.shape == target.shape
        sq_diff = (pred - target) ** 2
        tss_time = (
            target - target.mean(axis=0)[np.newaxis, ...]
        ) ** 2  # mean over time
        r_squared = 1 - sq_diff.sum(axis=0) / tss_time.sum(axis=0)  # sum over time
        if avg_grid:
            return r_squared.mean(
                axis=0
            )  # we decided to separately average globally at end
        else:
            return r_squared

    def calc_bias(self, pred, target, avg_grid=True):
        """
        calculate bias
        for vertically-resolved variables, input shape should be time x grid x level
        for scalars, input shape should be time x grid
        returns vector of length level or 1
        """
        assert pred.shape[1] == self.num_latlon
        assert pred.shape == target.shape
        bias = pred.mean(axis=0) - target.mean(axis=0)
        if avg_grid:
            return bias.mean(axis=0)  # we decided to separately average globally at end
        else:
            return bias

    def calc_CRPS(self, samplepreds, target, avg_grid=True):
        """
        calculate 'globally averaged' continuous ranked probability score
        for vertically-resolved variables, input shape should be time x grid x level x num_crps_samples
        for scalars, input shape should be time x grid x num_crps_samples
        returns vector of length level or 1
        """
        assert samplepreds.shape[1] == self.num_latlon
        assert len(samplepreds.shape) == len(target.shape) + 1
        assert len(samplepreds.shape) == 3 or len(samplepreds.shape) == 4
        num_crps = samplepreds.shape[-1]
        mae = np.mean(
            np.abs(samplepreds - target[..., np.newaxis]), axis=(0, -1)
        )  # mean over time and crps samples
        samplepreds = np.sort(samplepreds, axis=-1)
        diff = samplepreds[..., 1:] - samplepreds[..., :-1]
        count = np.arange(1, num_crps) * np.arange(num_crps - 1, 0, -1)
        if len(samplepreds.shape) == 4:
            spread = (
                (diff * count[np.newaxis, np.newaxis, np.newaxis, :])
                .sum(axis=-1)
                .mean(axis=0)
            )  # sum over crps samples and mean over time
        elif len(samplepreds.shape) == 3:
            spread = (
                (diff * count[np.newaxis, np.newaxis, :]).sum(axis=-1).mean(axis=0)
            )  # sum over crps samples and mean over time
        crps = mae - spread / (num_crps * (num_crps - 1))
        # count was not multiplied by two so no need to divide by two
        if avg_grid:
            return crps.mean(axis=0)  # we decided to separately average globally at end
        else:
            return crps

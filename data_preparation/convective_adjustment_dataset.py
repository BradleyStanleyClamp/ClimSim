"""
Dataset utilising the radiative transfer column model to perform convective adjustment.
Adapted from Brian Rose's Climate Modeling class https://www.atmos.albany.edu/facstaff/brose/classes/ENV480_Spring2014/styled-5/code-2/index.html
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from data_preparation.radiative_transfer_column_model import ColumnModel
import plotting
import logging


class ConvectiveAdjustmentDataset(Dataset):
    """
    Dataset capturing the simple model of radiative-convective equilibrium. Inputs are:
    - temperature profile,
    - longwave absorbed profile,
    - shortwave absorbed profile
    Outputs are the adjusted temperature profile.

    The dataset is controlled by three factors (of variation): abs_coeff, albedo, Q
    - abs_coeff (float): absorption coefficient
    - albedo (float): surface albedo
    - Q (float): global mean incoming solar radiation (i think)

    This dataset is used to monitor the ability of a model to generalise to the specific task of composition. As such there are the following dataset types:
    - train:
    - test, in_domain
    - val, composition
    - test, composition
    - test, extrapolation
    """

    def __init__(
        self,
        dataset_mode: str,
        dataset_type: str,
        dataset_cfg: dict,
        dataset_testing_type: str = None,
        normalisation_stats: dict = None,
        model: str = None,
    ):
        """
        Initilise the dataset.

        Args:
            dataset_mode (str): mode of the dataset ('train', 'val', 'test')
            dataset_type (str): type of dataset to build ('train', 'composition', 'ood')
            dataset_cfg (dict): configuration dictionary containing:
                - factor_ranges (dict): dictionary containing the ranges for each factor
                - num_samples_per_factor_group (int): number of samples to collect per factor group
                - num_levels (int): number of atmospheric levels in the column
                - ood_percent (float): percentage to extend the target range for ood samples
            normalisation_stats (dict): dictionary containing the mean and std for input and target normalization

        """
        super().__init__()
        self.dataset_mode = dataset_mode
        self.dataset_type = dataset_type
        self.factor_ranges = dataset_cfg["factor_ranges"]
        self.num_samples_per_factor_group = dataset_cfg["num_samples_per_factor_group"][dataset_testing_type]
        self.num_levels = dataset_cfg["num_levels"]
        self.ood_percent = dataset_cfg["ood_percent"]
        self.model = model
        logging.info(f"Building Convective Adjustment Dataset: mode={dataset_mode}, type={dataset_type}, model={model}")


        # seed
        np.random.seed(0)

        # Get dataset based on type
        self.input, self.target, self.params = self.build_convective_adjustment_dataset(
            dataset_type,
            self.factor_ranges,
            self.num_samples_per_factor_group,
            self.num_levels,
            self.ood_percent,
        )

        # Filter samples based on mode
        self.input, self.target, self.params = self.filter_dataset_by_mode(
            self.input, self.target, self.params, dataset_mode, dataset_type
        )

        # Normalize inputs and targets
        self.input, self.target, self.normalisation_stats = self.normalize_data(
            self.input, self.target, dataset_mode, normalisation_stats
        )

        # Reshape based on model input
        self.input, self.target = self.reshape_data_for_model(
            self.input, self.target, model
        )

        # Convert to torch tensors
        self.input = torch.tensor(self.input, dtype=torch.float32)
        self.target = torch.tensor(self.target, dtype=torch.float32)

    def __len__(self):
        """
        Return the length of the dataset.
        """
        return self.input.shape[0]

    def __getitem__(self, idx):
        """
        Get the item at index idx.

        Args:
            idx (int): index of the item to get

        Returns:
            tuple: (input, target) where input is a tensor of shape (num_levels, 3) and target is a tensor of shape (num_levels,)
        """
        return self.input[idx], self.target[idx]

    def filter_dataset_by_mode(
        self,
        data_X: np.ndarray,
        data_Y: np.ndarray,
        param_per_sample: list,
        dataset_mode: str,
        dataset_type: str,
    ) -> tuple[np.ndarray, np.ndarray, list]:
        """
        Filters the dataset based on the mode (train, val, test).
        V1:
        - 70% train, 30% test for in_domain
        - 50% val, 50% test for composition
        - all ood samples for test

        Args:
            data_X (np.ndarray): array of shape (num_samples, 3, num_levels+1) containing the inputs to the convective adjustment step
            data_Y (np.ndarray): array of shape (num_samples, num_levels+1) containing the outputs of the convective adjustment step
            param_per_sample (list): list of parameter dictionaries used for each sample (length num_samples)
            dataset_mode (str): mode of the dataset ('train', 'val', 'test')
        Returns:
            filtered_data_X (np.ndarray): filtered input data
            filtered_data_Y (np.ndarray): filtered target data
            filtered_param_per_sample (list): filtered parameter list
        """
        np.random.seed(0)
        if dataset_type == "in_domain":
            # Shuffle data
            indices = np.arange(len(data_X))
            np.random.shuffle(indices)
            data_X = data_X[indices]
            data_Y = data_Y[indices]
            param_per_sample = [param_per_sample[i] for i in indices]
            if dataset_mode == "train":
                split_index = int(len(data_X) * 0.7)
                return (
                    data_X[:split_index],
                    data_Y[:split_index],
                    param_per_sample[:split_index],
                )
            elif dataset_mode == "test":
                split_index = int(len(data_X) * 0.7)
                return (
                    data_X[split_index:],
                    data_Y[split_index:],
                    param_per_sample[split_index:],
                )
            else:
                raise ValueError(
                    f"Invalid dataset_mode {dataset_mode} for in_domain dataset."
                )
        elif dataset_type == "composition":
            # Shuffle data
            indices = np.arange(len(data_X))
            np.random.shuffle(indices)
            data_X = data_X[indices]
            data_Y = data_Y[indices]
            param_per_sample = [param_per_sample[i] for i in indices]
            if dataset_mode == "val":
                split_index = int(len(data_X) * 0.5)
                return (
                    data_X[:split_index],
                    data_Y[:split_index],
                    param_per_sample[:split_index],
                )
            elif dataset_mode == "test":
                split_index = int(len(data_X) * 0.5)
                return (
                    data_X[split_index:],
                    data_Y[split_index:],
                    param_per_sample[split_index:],
                )
            else:
                raise ValueError(
                    f"Invalid dataset_mode {dataset_mode} for composition dataset."
                )

        elif dataset_mode == "test" and dataset_type == "ood":
            return data_X, data_Y, param_per_sample
        else:
            raise ValueError(
                f"Invalid dataset_mode {dataset_mode} and dataset_type {dataset_type} combination."
            )

    def collect_convective_adjustment_data(
        self,
        params: dict,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Runs the radiative heating column model to collect inputs and outputs of the convective adjustment step.
        Inputs are: temperature profile, longwave absorbed profile, shortwave absorbed profile
        Outputs are: adjusted temperature profile

        Args:
            params (dict): parameters to initialize the ColumnModel.column consisting of:
                - num_levels (int): number of atmospheric levels in the column
                - abs_coeff (float): absorption coefficient
                - albedo (float): surface albedo
                - Q (float): global mean incoming solar radiation (i think)
        Returns:
            x (np.ndarray): array of shape (num_samples, 2, num_levels+1) containing the inputs to the convective adjustment step
            y (np.ndarray): array of shape (num_samples, num_levels+1) containing the outputs of the convective adjustment step
        """
        column = ColumnModel.column(params=params)
        col_t = np.concatenate(([column.Ts], column.Tatm))
        column.Longwave_Heating()
        column.Shortwave_Heating()

        col_lw_abs = np.concatenate(([column.LW_absorbed_sfc], column.LW_absorbed_atm))
        col_sw_abs = np.concatenate(([column.SW_absorbed_sfc], column.SW_absorbed_atm))
        x = np.stack((col_lw_abs, col_sw_abs), axis=0)

        column.Step_Forward()

        col_t_new = np.concatenate(([column.Ts], column.Tatm))

        y = (col_t_new - col_t) / column.params["timestep"]  # temperature tendency

        return x, y

    def build_column_model_params(
        self,
        sample_factor_values: dict,
        indx: int,
        num_levels: int,
        adj_lapse_rate: int = 6.5,
    ) -> dict:
        """
        Builds the parameters dictionary to initialize the ColumnModel.column
        Args:
            sample_factor_values (dict): dictionary containing the sampled values for each factor
            indx (int): index of the sample to use
            num_levels (int): number of atmospheric levels in the column
            adj_lapse_rate (int): lapse rate for convective adjustment (default: 6.5 K/km) (needed to be set to activate convective adjustment in ColumnModel)

        Returns:
            params (dict): parameters to initialize the ColumnModel.column consisting of:
                - num_levels (int): number of atmospheric levels in the column
                - abs_coeff (float): absorption coefficient
                - albedo (float): surface albedo
                - Q (float): global mean incoming solar radiation (i think)
        """
        return {
            "num_levels": num_levels,
            "adj_lapse_rate": adj_lapse_rate,
            "abs_coeff": sample_factor_values["abs_coeff"][indx],
            "albedo": sample_factor_values["albedo"][indx],
            "Q": sample_factor_values["Q"][indx],
        }

    def collect_data_from_sampled_factors(
        self, sample_factor_values: dict, num_samples: int, num_levels: int
    ) -> tuple[np.ndarray, np.ndarray, list]:
        """
        Given uniformly sampled factor values, collects convective adjustment data from the column model for each sample.
        Args:
            sample_factor_values (dict): dictionary containing the sampled values for each factor
            num_samples (int): number of samples to collect data for
            num_levels (int): number of atmospheric levels in the column
        Returns:
            data_X (np.ndarray): array of shape (num_samples, 3, num_levels+1) containing the inputs to the convective adjustment step
            data_Y (np.ndarray): array of shape (num_samples, num_levels+1) containing the outputs of the convective adjustment step
            param_per_sample (list): list of parameter dictionaries used for each sample (length num_samples)
        """
        data_X = []
        data_Y = []
        param_per_sample = []
        for i in range(num_samples):
            params = self.build_column_model_params(sample_factor_values, i, num_levels)
            param_per_sample.append(params)
            x, y = self.collect_convective_adjustment_data(params.copy())
            data_X.append(x)
            data_Y.append(y)
        data_X = np.array(data_X)
        data_Y = np.array(data_Y)

        return data_X, data_Y, param_per_sample

    def build_convective_adjustment_dataset(
        self,
        dataset_type: str,
        factor_ranges: dict,
        num_samples_per_factor_group: int,
        num_levels: int,
        ood_percent: float,
    ) -> tuple[np.ndarray, np.ndarray, list]:
        """
        Builds a convective adjustment dataset from sampled factor values, based on the dataset type specified.
        dataset types can be:
        - 'train': samples contain support ranges for all factors and target ranges for one factor at a time
        - 'composition': samples contains unseen combinations of factor values within the target ranges not including support ranges
        - 'ood': samples contain values outside the target ranges for all factors

        Args:
            dataset_type (str): type of dataset to build ('train', 'composition', 'ood')
            factor_ranges (dict): dictionary containing the ranges for each factor
            num_samples_per_factor_group (int): number of samples to collect per factor group
            num_levels (int): number of atmospheric levels in the column
            ood_percent (float): percentage to extend the target range for ood samples
        Returns:
            data_X (np.ndarray): array of shape (num_samples, 3, num_levels+1) containing the inputs to the convective adjustment step
            data_Y (np.ndarray): array of shape (num_samples, num_levels+1) containing the outputs of the convective adjustment step
            param_per_sample (list): list of parameter dictionaries used for each sample (length num_samples)
        """

        if dataset_type == "in_domain":
            data_X = []
            data_Y = []
            param_per_sample = []
            sample_factor_values = {}

            for target_factor in factor_ranges.keys():

                sample_factor_values[target_factor] = self.uniform_sample(
                    factor_ranges[target_factor]["range_target"],
                    size=num_samples_per_factor_group,
                )

                for support_factor in factor_ranges.keys():
                    if target_factor != support_factor:
                        sample_factor_values[support_factor] = self.uniform_sample(
                            factor_ranges[support_factor]["range_support"],
                            size=num_samples_per_factor_group,
                        )

                X, Y, params = self.collect_data_from_sampled_factors(
                    sample_factor_values, num_samples_per_factor_group, num_levels
                )

                data_X.append(X)
                data_Y.append(Y)
                param_per_sample.extend(params)
            data_X = np.concatenate(data_X, axis=0)
            data_Y = np.concatenate(data_Y, axis=0)
            return data_X, data_Y, param_per_sample

            # X, Y, params = self.collect_data_from_sampled_factors(
            #     sample_factor_values, num_samples_per_factor_group, num_levels
            # )
            # for i in range(num_samples_per_factor_group):
            #     for target_factor in factor_ranges.keys():
            #         keep = True
            #         for support_factor in factor_ranges.keys():
            #             if target_factor != support_factor:
            #                 if sample_factor_values[support_factor][i] > factor_ranges[support_factor]["range_support"][1]:
            #                     keep = False
            #         if keep:
            #             data_X.append(X[i])
            #             data_Y.append(Y[i])
            #             param_per_sample.append(params[i])
            #             break

            # data_X = np.array(data_X)
            # data_Y = np.array(data_Y)
            # return data_X, data_Y, param_per_sample

        elif dataset_type == "composition":
            sample_factor_values = {}
            for factor in factor_ranges.keys():
                composition_range = (
                    factor_ranges[factor]["range_support"][1],
                    factor_ranges[factor]["range_target"][1],
                )
                sample_factor_values[factor] = self.uniform_sample(
                    composition_range, size=num_samples_per_factor_group
                )

            return self.collect_data_from_sampled_factors(
                sample_factor_values, num_samples_per_factor_group, num_levels
            )

        elif dataset_type == "ood":
            sample_factor_values = {}
            for factor in factor_ranges.keys():
                composition_range = (
                    factor_ranges[factor]["range_target"][1],
                    factor_ranges[factor]["range_target"][1] * ood_percent,
                )
                sample_factor_values[factor] = self.uniform_sample(
                    composition_range, size=num_samples_per_factor_group
                )

            return self.collect_data_from_sampled_factors(
                sample_factor_values, num_samples_per_factor_group, num_levels
            )
        else:
            raise ValueError(f"Invalid dataset_type {dataset_type} specified.")

    def normalize_data(
        self,
        input_data: np.ndarray,
        target_data: np.ndarray,
        mode: str,
        normalisation_stats,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """
        Normalizes the input and target data using the provided normalization statistics.
        If no statistics are provided, returns the data as is.

        Args:
            input_data (np.ndarray): input data to normalize
            target_data (np.ndarray): target data to normalize
            mode (str): dataset mode ('train', 'val', 'test')
            normalisation_stats (dict): dictionary containing the mean and std for input and target normalization

        Returns:
            tuple: normalized input and target data, and normalization statistics used

        """
        if normalisation_stats is None:
            if mode != "train":
                raise ValueError(
                    "Normalization statistics must be provided for val and test modes."
                )

            normalisation_stats = {
                "input_mean": np.mean(input_data, axis=0),
                "input_std": np.std(input_data, axis=0) + 1e-8,
                "target_mean": np.mean(target_data, axis=0),
                "target_std": np.std(target_data, axis=0) + 1e-8,
            }

        input_mean = normalisation_stats.get("input_mean")
        input_std = normalisation_stats.get("input_std")
        target_mean = normalisation_stats.get("target_mean")
        target_std = normalisation_stats.get("target_std")

        input_data = (input_data - input_mean) / input_std
        target_data = (target_data - target_mean) / target_std

        return input_data, target_data, normalisation_stats

    def reshape_data_for_model(self, input_data, target_data, model=None):
        """
        Reshapes the input and target data based on the model requirements.
        For MLP model, flattens the input, the target data doesnt need reshaping.

        Args:
            input_data (np.ndarray): input data of shape (num_samples, 3, num_levels+1)
            target_data (np.ndarray): target data of shape (num_samples, num_levels+1)
            model (str): model type ('mlp' or None)
        Returns:
            reshaped_input (np.ndarray): reshaped input data
            reshaped_target (np.ndarray): reshaped target data
        """
        if model == "mlp":
            reshaped_input = input_data.reshape(input_data.shape[0], -1)
            logging.info(f"Reshaped input data to {reshaped_input.shape} for MLP model.")
            return reshaped_input, target_data
        else:
            return input_data, target_data

    def uniform_sample(self, range_tuple, size=1):
        return np.random.uniform(range_tuple[0], range_tuple[1], size=size)


if __name__ == "__main__":
    dataset_cfg = {
        "factor_ranges": {
            "abs_coeff": {
                "range_target": (1.229e-4, 2.259e-4),
                "range_support": (1.229e-4, 1.744e-4),
            },
            "albedo": {"range_target": (0.001, 0.999), "range_support": (0.001, 0.5)},
            "Q": {"range_target": (341.3, 500), "range_support": (341.3, 420.65)},
        },
        "num_samples_per_factor_group": 1000,
        "num_levels": 3,
        "ood_percent": 1.05,
    }

    train_dataset = ConvectiveAdjustmentDataset(
        dataset_mode="train",
        dataset_type="in_domain",
        dataset_cfg=dataset_cfg,
    )
    print(
        f"Train dataset size: {len(train_dataset)}, input: {train_dataset[0][0].shape}, target: {train_dataset[0][1].shape}"
    )
    normalisation_stats = train_dataset.normalisation_stats

    in_domain_test_dataset = ConvectiveAdjustmentDataset(
        dataset_mode="test",
        dataset_type="in_domain",
        dataset_cfg=dataset_cfg,
        normalisation_stats=normalisation_stats,
    )
    print(
        f"In-domain test dataset size: {len(in_domain_test_dataset)}, input: {in_domain_test_dataset[0][0].shape}, target: {in_domain_test_dataset[0][1].shape}"
    )
    composition_val_dataset = ConvectiveAdjustmentDataset(
        dataset_mode="val",
        dataset_type="composition",
        dataset_cfg=dataset_cfg,
        normalisation_stats=normalisation_stats,
    )
    print(
        f"Composition val dataset size: {len(composition_val_dataset)}, input: {composition_val_dataset[0][0].shape}, target: {composition_val_dataset[0][1].shape}"
    )
    composition_test_dataset = ConvectiveAdjustmentDataset(
        dataset_mode="test",
        dataset_type="composition",
        dataset_cfg=dataset_cfg,
        normalisation_stats=normalisation_stats,
    )
    print(
        f"Composition test dataset size: {len(composition_test_dataset)}, input: {composition_test_dataset[0][0].shape}, target: {composition_test_dataset[0][1].shape}"
    )
    ood_test_dataset = ConvectiveAdjustmentDataset(
        dataset_mode="test",
        dataset_type="ood",
        dataset_cfg=dataset_cfg,
        normalisation_stats=normalisation_stats,
    )
    print(
        f"OOD test dataset size: {len(ood_test_dataset)}, input: {ood_test_dataset[0][0].shape}, target: {ood_test_dataset[0][1].shape}"
    )

    datasets = {
        "train": train_dataset,
        # "in_domain_test": in_domain_test_dataset,
        # "composition_val": composition_val_dataset,
        # "composition_test": composition_test_dataset,
        # "ood_test": ood_test_dataset,
    }

    plotting.plot_convective_adjustment_dataset_factors_with_outputs(
        datasets=datasets, save_path="convective_adjustment_factors_outputs.png"
    )
    plotting.plot_convective_adjustment_dataset_inputs(
        datasets=datasets, save_path="convective_adjustment_inputs.png"
    )

    print("Plot saved to convective_adjustment_factors_outputs.png")

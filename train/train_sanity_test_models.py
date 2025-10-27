"""
Script that trains the models used for sanity testing process. These models are:
- Constant prediction model: always predicts the mean of the training set
- Multiple linear regression model: linear regression from inputs to outputs
"""

import warnings
import netCDF4 # Another weird import issue that is only triggered if netCDF4 imported after wandb 
with (
    warnings.catch_warnings()
):  # To catch annoying pydantic x wandb warning - looks like it should be adressed soon: https://github.com/wandb/wandb/issues/10662
    warnings.filterwarnings("ignore")
    import wandb
import logging
import string
from omegaconf import DictConfig
import hydra
from train import seed_everything
import torch
import data_preparation
import models
import yaml
import lightning as L
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from pathlib import Path
import train
import numpy as np
from evaluate.evaluate_utils import save_evaluation_results_to_json

@hydra.main(version_base=None, config_path="../config", config_name="train_general")
def main(cfg: DictConfig):
       # Seeding everything
    train.seed_everything(cfg.project.seed)
    
    torch.set_float32_matmul_precision('medium')

    # trainset, valset, testset = data_preparation.get_all_datasets(cfg.dataset, cfg.testing.dataset_testing_type)

    trainset = data_preparation.get_dataset(
        cfg.dataset, "train", 'reduced'
    )

    const_model = trainset.data_class.target_train.mean(axis=0)

    X = trainset.data_class.target_train
    bias_vector = np.ones((X.shape[0], 1))
    X = np.concatenate((X, bias_vector), axis=1)

    # mlr_weights = np.linalg.inv(X.transpose()@X)@X.transpose()@trainset.data_class.target_train

    testset = data_preparation.get_dataset(
        cfg.dataset, "test", 'full'
    )

    # Evaluate on testset 
    testset.data_class.set_pressure_grid(data_split="scoring")
    input_scoring = testset.data_class.input_scoring
    target_scoring = testset.data_class.target_scoring

    # constant prediction
    const_pred_scoring = np.repeat(const_model[np.newaxis, :], target_scoring.shape[0], axis = 0)

    # multiple linear regression
    # X_scoring = input_scoring
    # bias_vector_scoring = np.ones((X_scoring.shape[0], 1))
    # X_scoring = np.concatenate((X_scoring, bias_vector_scoring), axis=1)
    # mlr_pred_scoring = X_scoring@mlr_weights

    testset.data_class.model_names = ['const'] #, 'mlr'] # model name here
    preds = [const_pred_scoring] #, mlr_pred_scoring] # add prediction here
    testset.data_class.preds_scoring = dict(zip(testset.data_class.model_names, preds))


    testset.data_class.reweight_target(data_split="scoring")
    testset.data_class.reweight_preds(data_split="scoring")
    testset.data_class.metrics_names = ['MAE', 'RMSE', 'R2', 'bias']
    testset.data_class.create_metrics_df(data_split="scoring")

    dict_var = testset.data_class.metrics_var_scoring

    save_evaluation_results_to_json(dict_var, 'evaluation_results.json')



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    main()
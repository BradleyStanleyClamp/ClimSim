import warnings

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
from evaluate.evaluate_utils import load_evaluation_results_from_json

var_short_names = {0:'$dT/dt$',
                                1:'$dq/dt$',
                                2:'NETSW',
                                3:'FLWDS',
                                4:'PRECSC',
                                5:'PRECC',
                                6:'SOLS',
                                7:'SOLL',
                                8:'SOLSD',
                                9:'SOLLD'}


@hydra.main(version_base=None, config_path="../config", config_name="evaluate_general")
def main(cfg: DictConfig):

        # Seeding everything
    seed_everything(cfg.project.seed)

    torch.set_float32_matmul_precision("medium")

    dict_var = load_evaluation_results_from_json(cfg.saved_evaluation_results_path)

    model_names = list(dict_var.keys())
    metrics_names = list(next(iter(dict_var.values())).keys())
    letters = string.ascii_lowercase


    plot_df_byvar = {}
    for metric in metrics_names:
        plot_df_byvar[metric] = pd.DataFrame([dict_var[model][metric] for model in model_names],
                                                index=model_names)
        plot_df_byvar[metric] = plot_df_byvar[metric].rename(columns = var_short_names).transpose()


        # plot figure
    fig, axes = plt.subplots(nrows  = len(metrics_names), sharex = True)
    for i in range(len(metrics_names)):
        plot_df_byvar[metrics_names[i]].plot.bar(
            legend = False,
            ax = axes[i])
        if metrics_names[i] != 'R2':
            axes[i].set_ylabel('$W/m^2$')
        else:
            axes[i].set_ylim(0,1)

        axes[i].set_title(f'({letters[i]}) {metrics_names[i]}')
    axes[i].set_xlabel('Output variable')
    axes[i].set_xticklabels(plot_df_byvar[metrics_names[i]].index, \
        rotation=0, ha='center')

    axes[0].legend(columnspacing = .9, 
                labelspacing = .3,
                handleheight = .07,
                handlelength = 1.5,
                handletextpad = .2,
                borderpad = .2,
                ncol = 3,
                loc = 'upper right')
    fig.set_size_inches(7,8)
    fig.tight_layout()

    plt.savefig('test_plot_1.png', dpi=300)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    main()
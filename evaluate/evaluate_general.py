"""
General evaluation script, that takes a trained model and evaluates it on a given dataset using the ClimSim evlaution utlities
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


@hydra.main(version_base=None, config_path="../config", config_name="evaluate_general")
def main(cfg: DictConfig):

    # Seeding everything
    seed_everything(cfg.project.seed)

    torch.set_float32_matmul_precision("medium")

    testset = data_preparation.get_dataset(
        cfg.dataset, "test", cfg.testing.dataset_testing_type
    )

    with open(cfg.run_cfg_file_path, "r") as f:
        run_config_dict = yaml.safe_load(f)
    run_config = DictConfig(run_config_dict)

    model = models.load_model_from_checkpoint(
        checkpoint_path=cfg.checkpoint_path,
        model_name=cfg.model.name,
        model_params=run_config,
        data_params=cfg.dataset,
    )

    testset.data_class.set_pressure_grid(data_split="scoring")
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=run_config.batch_size,
        shuffle=False,
        num_workers=cfg.dataset.general_dataset_config.num_workers,
    )

    model.eval()
    model.freeze()
    outputs_list = []
    for batch in tqdm(testloader):
        input, _ = batch
        outputs = model(input)
        # Process outputs as needed
        outputs_list.append(outputs)
    
    outputs_all = torch.cat(outputs_list, dim=0)
    testset.data_class.model_names = [cfg.model.name]
    preds = [outputs_all.cpu().numpy()]
    testset.data_class.preds_scoring = dict(zip(testset.data_class.model_names, preds))


    testset.data_class.reweight_target(data_split="scoring")
    testset.data_class.reweight_preds(data_split="scoring")
    testset.data_class.metrics_names = ['MAE', 'RMSE', 'R2', 'bias']
    testset.data_class.create_metrics_df(data_split="scoring")

    letters = string.ascii_lowercase

    # create custom dictionary for plotting
    dict_var = testset.data_class.metrics_var_scoring
    plot_df_byvar = {}
    for metric in testset.data_class.metrics_names:
        plot_df_byvar[metric] = pd.DataFrame([dict_var[model][metric] for model in testset.data_class.model_names],
                                                index=testset.data_class.model_names)
        plot_df_byvar[metric] = plot_df_byvar[metric].rename(columns = testset.data_class.var_short_names).transpose()

    # plot figure
    fig, axes = plt.subplots(nrows  = len(testset.data_class.metrics_names), sharex = True)
    for i in range(len(testset.data_class.metrics_names)):
        plot_df_byvar[testset.data_class.metrics_names[i]].plot.bar(
            legend = False,
            ax = axes[i])
        if testset.data_class.metrics_names[i] != 'R2':
            axes[i].set_ylabel('$W/m^2$')
        else:
            axes[i].set_ylim(0,1)

        axes[i].set_title(f'({letters[i]}) {testset.data_class.metrics_names[i]}')
    axes[i].set_xlabel('Output variable')
    axes[i].set_xticklabels(plot_df_byvar[testset.data_class.metrics_names[i]].index, \
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

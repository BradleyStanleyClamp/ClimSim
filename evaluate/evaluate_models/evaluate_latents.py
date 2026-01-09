"""
Working script for ongoing analysis of model performance.
"""

import json
from typing import Tuple
from omegaconf import DictConfig, OmegaConf
import logging
import hydra
import os
import re
import torch
import yaml
import numpy as np
import numbers
from tqdm import tqdm
import lightning as L
import train
import data_preparation
import models
import evaluate
import plotting
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns


@hydra.main(
    version_base=None, config_path="../../config", config_name="process_and_save_data"
)
def main(cfg: DictConfig):
    train.seed_everything(cfg.project.seed)

    path_to_data = (
        "/work/scratch-pw5/bradlesc/climsim/temp/p2.1.4.9_analysis/norm_outputs_001/"
    )

    sample_size = 10000

    mam_trained_son_latents = np.load(
        os.path.join(path_to_data, "train_group_MAM", "JJA_test_latents.npy")
    )
    mam_trained_mam_latents = np.load(
        os.path.join(path_to_data, "train_group_MAM", "MAM_test_latents_sampled.npy")
    )

    logging.info(f"mam_trained_son_latents shape: {mam_trained_son_latents.shape}")
    logging.info(f"mam_trained_mam_latents shape: {mam_trained_mam_latents.shape}")

    rnd_son_indices = np.random.choice(
        mam_trained_son_latents.shape[0], size=sample_size, replace=False
    )
    # rnd_mam_indices = np.random.choice(
    #     mam_trained_mam_latents.shape[0], size=sample_size, replace=False
    # )

    mam_trained_son_latents = mam_trained_son_latents[rnd_son_indices, :, :]
    # mam_trained_mam_latents = mam_trained_mam_latents[rnd_mam_indices, :, :]

    np.save(
        os.path.join(path_to_data, "train_group_MAM", "JJA_test_latents_sampled.npy"),
        mam_trained_son_latents,
    )
    # np.save(
    #     os.path.join(path_to_data, "train_group_MAM", "MAM_test_latents_sampled.npy"),
    #     mam_trained_mam_latents,
    # )

    logging.info("Performing PCA on latent representations...")


    # PCA over all latents
    mam_trained_son_latents_over_feature = mam_trained_son_latents.reshape(
        -1, mam_trained_son_latents.shape[-1]
    )
    mam_trained_mam_latents_over_feature = mam_trained_mam_latents.reshape(
        -1, mam_trained_mam_latents.shape[-1]
    )

    latents_over_feature = np.concatenate(
        [mam_trained_son_latents_over_feature, mam_trained_mam_latents_over_feature],
        axis=0,
    )
    n_components = 4

    latents_over_feature_reduced = PCA(n_components=n_components).fit_transform(
        latents_over_feature
    )

    labels = (["mam_trained_son"] * mam_trained_son_latents_over_feature.shape[0]) + (
        ["mam_trained_mam"] * mam_trained_mam_latents_over_feature.shape[0]
    )

    df = pd.DataFrame(
        latents_over_feature_reduced, columns=[f"pc_{i+1}" for i in range(n_components)]
    )
    df["group"] = labels

    sns.set_theme(style="white")

    g = sns.pairplot(
        df,
        vars=[f"pc_{i+1}" for i in range(n_components)],
        hue="group",
        diag_kind="kde",  # or 'hist'
        plot_kws=dict(s=25, alpha=0.7, edgecolor="k", linewidth=0.2),
        diag_kws=dict(fill=True, alpha=0.5),
    )

    g.figure.suptitle("PCA of Latent Representations", y=1.02)
    g.savefig("pca_latent_representations.png", dpi=300, bbox_inches="tight")
    logging.info("PCA plot saved as pca_latent_representations.png")

    # PCA per latent level
    for i in range(mam_trained_son_latents.shape[1]):
        latents_per_level_son = mam_trained_son_latents[:, i, :]
        latents_per_level_mam = mam_trained_mam_latents[:, i, :]
        latents_per_level = np.concatenate(
            [latents_per_level_son, latents_per_level_mam], axis=0
        )
        latents_per_level_reduced = PCA(n_components=4).fit_transform(
            latents_per_level
        )
        df_level = pd.DataFrame(
            latents_per_level_reduced,
            columns=[f"pc_{j+1}" for j in range(4)],
        )
        labels = (["mam_trained_son"] * latents_per_level_son.shape[0]) + (
            ["mam_trained_mam"] * latents_per_level_mam.shape[0]
        )
        df_level["group"] = labels
        sns.set_theme(style="white")
        g_level = sns.pairplot(
            df_level,
            vars=[f"pc_{j+1}" for j in range(4)],
            hue="group",
            diag_kind="kde",  # or 'hist'
            plot_kws=dict(s=25, alpha=0.7, edgecolor="k", linewidth=0.2),
            diag_kws=dict(fill=True, alpha=0.5),
        )
        g_level.figure.suptitle(
            f"PCA of Latent Representations at Level {i}", y=1.02
        )
        g_level.savefig(
            f"pca_latent_representations_level_{i}.png", dpi=300, bbox_inches="tight"
        )

        logging.info(
            f"PCA plot for latent level {i} saved as pca_latent_representations_level_{i}.png"
        )


    # Over levels 
    mam_trained_mam_latents_over_level = mam_trained_mam_latents.transpose(0, 2, 1).reshape(-1, 45)
    mam_trained_son_latents_over_level = mam_trained_son_latents.transpose(0, 2, 1).reshape(-1, 45)

    latents_over_level = np.concatenate(
        [mam_trained_son_latents_over_level, mam_trained_mam_latents_over_level],
        axis=0,
    )
    latents_over_level_reduced = PCA(n_components=n_components).fit_transform(
        latents_over_level)
    labels = (["mam_trained_son"] * mam_trained_son_latents_over_level.shape[0]) + (
        ["mam_trained_mam"] * mam_trained_mam_latents_over_level.shape[0]
    )
    df_level = pd.DataFrame(
        latents_over_level_reduced, columns=[f"pc_{i+1}" for i in range(n_components)]
    )
    df_level["group"] = labels
    sns.set_theme(style="white")
    g_level = sns.pairplot(
        df_level,
        vars=[f"pc_{i+1}" for i in range(n_components)],
        hue="group",
        diag_kind="kde",  # or 'hist'
        plot_kws=dict(s=25, alpha=0.7, edgecolor="k", linewidth=0.2),
        diag_kws=dict(fill=True, alpha=0.5),
    )
    g_level.figure.suptitle(
        f"PCA of Latent Representations over Levels", y=1.02
    )
    g_level.savefig(
        f"pca_latent_representations_over_levels.png", dpi=300, bbox_inches="tight"
    )
    logging.info(
        f"PCA plot for latent over levels saved as pca_latent_representations_over_levels.png"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    main()

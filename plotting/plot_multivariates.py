"""
Script to plot energy distance results for different data groups.
"""

import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def plot_multivariate_results(distance_dict: dict, metric_name: str,save_path: str):
    """
    Plots energy distance results for different data groups.

    Args:
        energy_distance_dict (dict): A dictionary where keys are group indices and values are energy distance values.
        save_path (str): The path to save the plot image.
    """

    # Plot results
    fig, ax = plt.subplots(figsize=(8, 4))

    # Accept dicts keyed by ints or strings; order by integer key when possible
    keys = list(distance_dict.keys())
    try:
        keys_sorted = sorted(keys, key=lambda k: int(k))
    except Exception:
        keys_sorted = sorted(keys)

    x = list(range(len(keys_sorted)))
    # p_values = [distance_dict[k]['p_value'] for k in keys_sorted]
    y = [distance_dict[k]['value'] for k in keys_sorted]
    # y_err = [distance_dict[k]['std_err'] for k in keys_sorted]

    # ax.errorbar(x, y, yerr=y_err, marker="o", linestyle="-")
    ax.errorbar(x, y, marker="o", linestyle="-", color="#8f91a2")
    ax.set_xlabel("Training year")
    ax.set_ylabel(f"{metric_name.replace('_', ' ').title()}")
    ax.set_title(f"{metric_name.replace('_', ' ').title()} between train distribution \n and test distribution")
    ax.grid(True)
    fig.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, transparent=True)
    logging.info(f"Saved plot to {save_path}")
    plt.close(fig)


def plot_energy_distance_results_with_p_values(energy_distance_dict: dict, save_path: str, cmap: str = "viridis"):
    """
    Plots energy distance results for different data groups and colors points by p-value.

    Args:
        energy_distance_dict (dict): A dictionary where keys are group indices and values are dicts with keys:
            - 'value': energy distance value
            - 'std_err': standard error
            - 'p_value': p-value
        save_path (str): The path to save the plot image.
        cmap (str): Matplotlib colormap name (default: 'viridis')
    """
    # Prepare data, same robust key handling as above
    keys = list(energy_distance_dict.keys())
    try:
        keys_sorted = sorted(keys, key=lambda k: int(k))
    except Exception:
        keys_sorted = sorted(keys)

    x = list(range(len(keys_sorted)))
    p_values = np.array([energy_distance_dict[k]['p_value'] for k in keys_sorted])
    y = np.array([energy_distance_dict[k]['value'] for k in keys_sorted])
    y_err = np.array([energy_distance_dict[k]['std_err'] for k in keys_sorted])

    # Use discrete p-value bins with fixed colors
    # Bins: <0.001, <0.01, <0.05, >=0.05
    bins = [0.005, 0.05]
    # Choose distinct colors for each bin (from most significant to least)

    bin_colors = ["#002147", "#8f91a2", "#94b0da"]
    # digitize returns indices 0..len(bins); idx 0 => p <= bins[0], etc.
    idx = np.digitize(p_values, bins)

    # Map indices to colors
    colors = [bin_colors[i] for i in idx]

    # Plot onto explicit Axes so legend can be added
    fig, ax = plt.subplots(figsize=(8, 4))
    # Draw a grey line joining the points first so markers/errorbars overlay it
    ax.plot(x, y, color='grey', linestyle='-', linewidth=1, zorder=0)
    for xi, yi, err, c in zip(x, y, y_err, colors):
        ax.errorbar(xi, yi, yerr=err, marker="o", color=c, ecolor=c, elinewidth=1.2, capsize=3, zorder=1)

    ax.set_xlabel("Training year")
    ax.set_ylabel("Energy distance")
    ax.set_title("Energy distance between training-year distribution \n and test-year distribution (Year 8)")
    ax.grid(True)

    # Add a legend that shows the discrete p-value bins
    legend_handles = [
        Patch(color=bin_colors[0], label='$< 0.005$'),
        # Patch(color=bin_colors[1], label='$< 0.01$'),
        Patch(color=bin_colors[1], label='$< 0.05$'),
        Patch(color=bin_colors[2], label='$>= 0.05$'),
    ]
    ax.legend(handles=legend_handles, title='p-value', loc='best')

    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, transparent=True)
    logging.info(f"Saved plot to {save_path}")
    plt.close(fig)



def plot_kl_divergence_results(kl_divergence_dict: dict, save_path: str):
    """
    Plots KL divergence results for different data groups.

    Args:
        kl_divergence_dict (dict): A dictionary where keys are group indices and values are KL divergence values.
        save_path (str): The path to save the plot image.
    """

    # Plot results
    fig, ax = plt.subplots(figsize=(8, 4))

    # Accept dicts keyed by ints or strings; order by integer key when possible
    keys = list(kl_divergence_dict.keys())
    try:
        keys_sorted = sorted(keys, key=lambda k: int(k))
    except Exception:
        keys_sorted = sorted(keys)

    x = list(range(len(keys_sorted)))
    # p_values = [kl_divergence_dict[k]['p_value'] for k in keys_sorted]
    y = [kl_divergence_dict[k]['value'] for k in keys_sorted]
    # y_err = [kl_divergence_dict[k]['std_err'] for k in keys_sorted]

    ax.errorbar(x, y, marker="o", linestyle="-")
    ax.set_xlabel("Data group index")
    ax.set_ylabel("KL Divergence")
    ax.set_title("KL Divergence vs Data group index")
    ax.grid(True)
    fig.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    logging.info(f"Saved plot to {save_path}")
    plt.close(fig)
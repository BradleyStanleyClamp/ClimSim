import numpy as np
import matplotlib.pyplot as plt


def plot_convective_adjustment_dataset_factors_with_outputs(
    datasets: dict, save_path: str = None
):
    """
    Plots the distributions of convective adjustment factors of variation with different outputs.

    Parameters:
    - datasets (dict): Dict of pytorch datasets
    - save_path (str, optional): Path to save the plot. If None, the plot is shown instead.
    """

    num_factors = 3

    factor_ranges = datasets[list(datasets.keys())[0]].factor_ranges

    fig, axes = plt.subplots(num_factors, 1, figsize=(8, 12), sharex=False)
    colors = ["blue", "orange", "green", "red", "purple"]
    for ax, factor_name in zip(axes, factor_ranges.keys()):
        ax2 = ax.twinx()
        for idx, (dataset_name, dataset) in enumerate(datasets.items()):
            params = dataset.params
            data_Y = dataset.target.numpy()
            # factor_ranges = dataset.factor_ranges
            color = colors[idx]

            # Left axis: histograms
            ax.hist(
                [param[factor_name] for param in params],
                bins=30,
                alpha=0.5,
                label=dataset_name,
                color=color,
            )
            ax.set_ylabel("Count")
            ax.set_title(factor_name)

            # Right axis: comp_data_Y
            for level_idx in [0]:  # range(num_levels + 1):
                ax2.plot(
                    [param[factor_name] for param in params],
                    data_Y[:, level_idx],
                    "o",
                    label=f"{dataset_name} L{level_idx}",
                    markersize=4,
                    color=color,
                )

                break

            # ax2.set_ylabel("comp_data_Y")

            # Combine legends from both axes
            handles1, labels1 = ax.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(handles1 + handles2, labels1 + labels2, loc="best")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()


def plot_convective_adjustment_dataset_inputs(datasets: dict, save_path: str = None):
    """
    Plots the distributions of convective adjustment dataset inputs.

    """
    num_levels = datasets[list(datasets.keys())[0]].num_levels
    colors = ["blue", "orange", "green", "red", "purple"]

    fig, axes = plt.subplots(2, num_levels + 1, figsize=(20, 12), sharex=False)
    for wt in [0, 1]:
        for lvl in range(num_levels + 1):
            ax = axes[wt, lvl]
            for idx, (dataset_name, dataset) in enumerate(datasets.items()):
                color = colors[idx]
                data_X = dataset.input.numpy()
                ax.hist(
                    data_X[:, wt, lvl],
                    bins=30,
                    alpha=0.5,
                    label=dataset_name,
                    color=color,
                )

            ax.set_title(f"wave type {wt} level {lvl}")
            ax.set_ylabel("Count")
            ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()

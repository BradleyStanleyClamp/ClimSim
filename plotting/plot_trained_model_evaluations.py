import numpy as np
import matplotlib.pyplot as plt
from itertools import chain


def plot_trained_model_evaluations(
    models_dict, figsize_per_row=(6, 3), cmap_name="tab10"
):
    """
    Plot grouped bar charts for metrics across an arbitrary number of models.

    Parameters
    ----------
    models_dict : dict
        Mapping model_name -> results, where results is mapping:
            metric_name -> { var_name: array_like, ... }
    figsize_per_row : tuple
        Width, height per subplot row (will be multiplied by number of metrics).
    cmap_name : str
        Matplotlib colormap name to choose distinct colors for models.
    """
    # Collect metric names (union across models)
    metric_names = sorted(
        set(chain.from_iterable(m.keys() for m in models_dict.values()))
    )

    # Prepare figure
    nrows = len(metric_names)
    figsize = (figsize_per_row[0], figsize_per_row[1] * nrows)
    fig, axes = plt.subplots(nrows=nrows, figsize=figsize, sharex=False)
    axes = np.atleast_1d(axes)  # handle single-row case

    model_names = list(models_dict.keys())
    n_models = len(model_names)

    # colors
    cmap = plt.get_cmap(cmap_name)
    colors = [cmap(i % cmap.N) for i in range(n_models)]

    for ax, metric in zip(axes, metric_names):
        # build ordered list of vars: start from first model encountered, then add any missing
        first_model = model_names[0]
        vars_order = []
        # gather union but try to preserve order from first model
        if metric in models_dict[first_model]:
            vars_order.extend(models_dict[first_model][metric].keys())
        # add any variables present in other models
        for m in model_names:
            if metric in models_dict[m]:
                for v in models_dict[m][metric].keys():
                    if v not in vars_order:
                        vars_order.append(v)
        if not vars_order:
            # nothing to plot for this metric
            ax.text(
                0.5, 0.5, f'No data for metric "{metric}"', ha="center", va="center"
            )
            ax.set_axis_off()
            continue

        x = np.arange(len(vars_order))

        # width per bar group scaled by number of models
        total_group_width = 0.8
        single_width = total_group_width / n_models
        offsets = (np.arange(n_models) - (n_models - 1) / 2) * single_width

        # For each model, compute means for each var (nan if missing)
        for i, model in enumerate(model_names):
            values = []
            for v in vars_order:
                try:
                    val = models_dict[model][metric][v]
                    # attempt to compute mean; if scalar-like, cast to float
                    mean_val = np.asarray(val).mean().item()
                except Exception:
                    mean_val = np.nan
                values.append(mean_val)
            ax.bar(
                x + offsets[i],
                values,
                single_width,
                label=model,
                color=colors[i],
                edgecolor="k",
                linewidth=0.3,
            )

        # axis labels / formatting
        if metric.lower() == "r2":
            ax.set_ylim(0, 1)
            ax.set_ylabel("R2")
        else:
            ax.set_ylabel("$W/m^2$")

        ax.set_title(f"{metric}")
        ax.set_xticks(x)
        ax.set_xticklabels(vars_order, rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", linestyle=":", alpha=0.5)
    axes[-1].set_xlabel("Output variables")

    plt.tight_layout()
    plt.savefig("trained_model_evaluations.png")
    plt.close(fig)

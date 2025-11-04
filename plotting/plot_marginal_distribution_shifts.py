
import matplotlib.pyplot as plt
import numpy as np


# 'state_t' :: air temperature :: 60 :: K 
# 'state_q0001' :: specific humidity :: 60 :: kg/kg
# 'state_ps' :: surface pressure :: 1 :: Pa
# 'pbuf_SOLIN' :: solar insolation :: 1 :: W/m^2
# 'pbuf_LHFLX' :: surface latent heat flux :: 1 :: W/m^2
# 'pbuf_SHFLX' :: surface sensible heat flux :: 1 :: W/m^2


in_var_names = {0:'Air temperature',
                1:'Specific humidity',
                2:'Surface pressure',
                3:'Solar insolation',
                4:'Surface latent heat flux',
                5:'Surface sensible heat flux'}


def plot_dict_lines(data, save_path: str):
    """
    Plots lines connecting corresponding elements across dictionary keys.

    Example input:
        data = {1: [5, 3, 7],
                2: [8, 1, 6],
                3: [2, 4, 9]}

    This will plot 3 lines:
        - y = [5, 8, 2]
        - y = [3, 1, 4]
        - y = [7, 6, 9]
    """

    # Sort keys to ensure consistent x order
    x_vals = sorted(data.keys())

    # Convert to list of lists for easier manipulation
    y_lists = [data[k] for k in x_vals]

    # Transpose the list so that we can iterate over each "line"
    for y_line in zip(*y_lists):
        plt.plot(x_vals, y_line, marker='o')

    plt.xlabel("Keys")
    plt.ylabel("Values")
    plt.title("Lines connecting corresponding values across keys")
    plt.grid(True)
    plt.savefig(save_path, dpi=200)

def plot_standard_feature_marginals(data, save_path: str):
    """
    Plots the standard 124 features marginal distributions accross three plots, the first showing air temperature at 60 levels,
    the second showing specific humidity at 60 levels, and the third showing the remainder of variables.
    
    Each line corresponds to a list index across keys (same as the previous function).
    The first two subplots have a color gradient across lines.

    Args:
        data: dict where keys are x-values and values are lists of length 124
        save_path: path to save the resulting plot
    """
    # data = data['marginal_distances'] # For now not dealing with se
    assert all(len(v['marginal_distances']) == 124 for v in data.values()), "All value lists must be of length 124."

    # Ensure consistent x-axis order
    x_vals = sorted(data.keys())
    y_lists = [data[k]['marginal_distances'] for k in x_vals]
    se_lists = [data[k]['marginal_distribution_se'] for k in x_vals]
    
    # Transpose so that each element in `y_lines` is one "line" (across keys)
    y_lines = list(zip(*y_lists))
    se_lines = list(zip(*se_lists))

    y_lines_np = np.array(y_lines)
    max_val = np.max(y_lines_np)
    min_val = np.min(y_lines_np)

    # Define the segment ranges
    segments = [(0, 60), (60, 120), (120, 124)]

    fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True, constrained_layout=False)
    plt.subplots_adjust(right=0.85)  # room for colorbars only

    for i, (start, end) in enumerate(segments):
        ax = axes[i]
        subset = y_lines[start:end]
        se_subset = se_lines[start:end]

        if len(subset) == 0:
            continue

        # Create color gradient for the first two plots
        if i < 2:
            cmap = plt.cm.viridis  # You can use any colormap (e.g., 'plasma', 'coolwarm', 'turbo')
            colors = cmap(np.linspace(0, 1, len(subset)))
            # Normalizer for mapping line index -> colorbar ticks
            norm = plt.Normalize(vmin=0, vmax=(len(subset) - 1) if len(subset) > 1 else 1)
        else:
            cmap = plt.cm.get_cmap('tab20')  # good for up to 20 distinct colors
            colors = cmap(np.linspace(0, 1, len(subset)))

        for j, (y_line, se_line, color) in enumerate(zip(subset, se_subset, colors)):
            if i < 2:
                ax.errorbar(x_vals, y_line, yerr=se_line, color=color, alpha=0.9)
            else:
                ax.errorbar(x_vals, y_line, yerr=se_line, color=color, alpha=0.9, label=str(in_var_names[i+j]))

        # --- Colorbars for the first two, outside on the right ---
        if i < 2:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            # align vertically with each subplot
            bbox = ax.get_position()
            cbar_ax = fig.add_axes([0.87, bbox.y0, 0.02, bbox.height])
            cb = fig.colorbar(sm, cax=cbar_ax)
            # choose a few ticks and label them with the corresponding global feature indices
            if len(subset) > 1:
                tick_idxs = np.linspace(0, len(subset) - 1, min(6, len(subset))).astype(int)
                cb.set_ticks(tick_idxs)
                cb.set_ticklabels([str(start + int(t)) for t in tick_idxs])
            cb.set_label("Feature index pressure level")
            ax.set_title(f'{in_var_names[i]}')

        # --- Legend INSIDE the bottom-right corner of the last plot ---
        if i == 2:
            ax.legend(
                loc='upper right',
                fontsize='small',
                frameon=False
            )
            ax.set_title('Scalar input features')

        ax.set_ylabel("Values")
        ax.grid(True)
        ax.set_ylim(min_val, max_val)
        fig.suptitle("Marginal Distribution Distances for Standard Features", fontsize=16, y=0.95)
    

    axes[-1].set_xlabel("Training year")
    # plt.tight_layout()
    plt.savefig(save_path, dpi=200, transparent=True)
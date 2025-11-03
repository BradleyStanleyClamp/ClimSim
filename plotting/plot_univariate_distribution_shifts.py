
import matplotlib.pyplot as plt
import numpy as np

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

def plot_dict_segments(data, save_path: str):
    """
    Plots three subplots (stacked vertically) showing different index ranges of list values in `data`.
    
    - Plot 1: items [0:60]
    - Plot 2: items [60:120]
    - Plot 3: items [120:124]
    
    Each line corresponds to a list index across keys (same as the previous function).
    The first two subplots have a color gradient across lines.
    """

    # Ensure consistent x-axis order
    x_vals = sorted(data.keys())
    y_lists = [data[k] for k in x_vals]
    
    # Transpose so that each element in `y_lines` is one "line" (across keys)
    y_lines = list(zip(*y_lists))

    # Define the segment ranges
    segments = [(0, 60), (60, 120), (120, 124)]
    titles = ['Items 0–60', 'Items 60–120', 'Items 120–124']

    fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True)

    for i, (start, end) in enumerate(segments):
        ax = axes[i]
        subset = y_lines[start:end]

        if len(subset) == 0:
            continue

        # Create color gradient for the first two plots
        if i < 2:
            cmap = plt.cm.viridis  # You can use any colormap (e.g., 'plasma', 'coolwarm', 'turbo')
            colors = cmap(np.linspace(0, 1, len(subset)))
        else:
            colors = ['gray'] * len(subset)

        for y_line, color in zip(subset, colors):
            ax.plot(x_vals, y_line, color=color, alpha=0.9)

        ax.set_title(titles[i])
        ax.set_ylabel("Values")
        ax.grid(True)

    axes[-1].set_xlabel("Keys")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
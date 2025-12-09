import matplotlib.pyplot as plt
import json
from matplotlib.patches import Patch


def plot_marginal_composition_analysis(data):
    """
    data: list of dicts with keys:
        "overlap_percentage", "coverage",
        "train_min", "train_max", "test_min", "test_max"
    """

    num_groups = len(data)
    fig, axes = plt.subplots(num_groups, 1, figsize=(8, 4 * num_groups))

    for i, (group_idx, data_group) in enumerate(data.items()):
        ax = axes[i]

        # X positions
        x = list(range(len(data_group)))

        # Heights of bars
        heights = [(d["overlap_percentage"]) for d in data_group]

        # Colors based on coverage
        colors = ["green" if d["coverage"] else "red" for d in data_group]

        # Create bar plot

        ax.bar(x, heights, color=colors)

        # Labeling
        ax.set_yscale("log")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Overlap Percentage")
        ax.set_title(f"{group_idx}")

        # Optional: show values above bars
        # for i, v in enumerate(heights):
        #     ax.text(i, v + 0.5, f"{v:.1f}", ha="center")

        legend_patches = [
            Patch(color="green", label="Coverage = True"),
            Patch(color="red", label="Coverage = False"),
        ]
        ax.legend(handles=legend_patches, title="Coverage Status")

    plt.tight_layout()
    plt.savefig("marginal_composition_analysis.png")
    plt.close()


if __name__ == "__main__":
    # path_to_data = "/home/users/bradlesc/projects/ClimSim/logs/p2.1.3/7/testing/quick_test/2025-12-08-16-49-17/levels_45/composition_metrics_energy_distance.json"
    path_to_data = '/home/users/bradlesc/projects/ClimSim/logs/p2.1.3/7/testing/dress_rehearsal/2025-12-08-16-51-33/levels_45/composition_metrics_energy_distance.json'
    with open(path_to_data, "r") as f:
        data = json.load(f)

    plot_marginal_composition_analysis(data)

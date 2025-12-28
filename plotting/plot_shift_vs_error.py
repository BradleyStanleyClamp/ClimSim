import matplotlib.pyplot as plt

def plot_distribution_shift_vs_error(
    distribution_shift_results,
    models_results,
    error_key="mean_test_loss",
    min_key="min_test_loss",
    max_key="max_test_loss",
    connect_points=True,
):
    """
    Plot distribution shift (x) vs model error (y) for multiple models,
    with min/max error bars and consistent colors per model.
    """

    seasons = list(distribution_shift_results.keys())
    x_vals = [distribution_shift_results[s]["value"] for s in seasons]

    plt.figure(figsize=(7, 5))
    ax = plt.gca()

    for model_name, model_groups in models_results.items():
        xs, ys = [], []
        yerr_lower, yerr_upper = [], []

        for season, x in zip(seasons, x_vals):
            if season not in model_groups:
                continue

            mean = model_groups[season][error_key]
            min_val = model_groups[season][min_key]
            max_val = model_groups[season][max_key]

            xs.append(x)
            ys.append(mean)
            yerr_lower.append(mean - min_val)
            yerr_upper.append(max_val - mean)

        # Draw error bars + markers
        err_container = ax.errorbar(
            xs,
            ys,
            yerr=[yerr_lower, yerr_upper],
            fmt="o",
            capsize=4,
            label=f"{model_name} (min–max)"
        )

        # Extract the color used by errorbar
        color = err_container.lines[0].get_color()

        # Draw connecting line with the SAME color
        if connect_points and len(xs) > 1:
            xs_sorted, ys_sorted = zip(*sorted(zip(xs, ys)))
            ax.plot(xs_sorted, ys_sorted, color=color)

    # Labels and title
    ax.set_xlabel("Distribution Shift")
    ax.set_ylabel(error_key.replace("_", " ").title())
    ax.set_title("Distribution Shift vs Model Error")

    # Season labels below x-axis
    y_min, y_max = ax.get_ylim()
    y_offset = (y_max - y_min) * 0.12

    for season, x in zip(seasons, x_vals):
        ax.text(
            x,
            y_min - y_offset,
            season,
            ha="center",
            va="top",
            fontsize=10,
            clip_on=False
        )

    plt.subplots_adjust(bottom=0.25)

    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig("distribution_shift_vs_error.png", dpi=200)
    plt.close()

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np




def plot_multiple_marginal_distributions_on_single_plot(data_dict: dict, variable_name: str, save_path: str, groups_to_plot=None):
    """
    Plots multiple distributions of a variable on a single plot.

    Args:
        data_dict (dict): A dictionary where keys are labels and values are numpy arrays of the variable.
        variable_name (str): The name of the variable to plot.
        save_path (str): The path to save the plot image.
    """
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    for label, group in data_dict.items():
        if groups_to_plot is not None and label not in groups_to_plot:
            continue
        data = group[variable_name]
        sns.kdeplot(data, ax=ax, fill=True, label=label)

    plt.title(f'Distribution of {variable_name}')
    plt.xlabel(variable_name)
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(save_path, transparent=True)
    plt.close()
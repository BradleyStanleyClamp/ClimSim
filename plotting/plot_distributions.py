import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_single_distribution_from_numpy(ax: plt.Axes, data: np.ndarray, variable_name: str, save_path: str):
    """
    Plots the distribution of a single variable from a numpy array.

    Args:
        data (np.ndarray): The data array containing the variable.
        variable_name (str): The name of the variable to plot.
        save_path (str): The path to save the plot image.
    """
    plot = sns.kdeplot(data, ax=ax, fill=True)

    return plot


def plot_multiple_distributions_on_single_plot(data_dict: dict, variable_name: str, save_path: str):
    """
    Plots multiple distributions of a variable on a single plot.

    Args:
        data_dict (dict): A dictionary where keys are labels and values are numpy arrays of the variable.
        variable_name (str): The name of the variable to plot.
        save_path (str): The path to save the plot image.
    """
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    for label, data in data_dict.items():
        sns.kdeplot(data[variable_name], ax=ax, fill=True, label=label)

    plt.title(f'Distribution of {variable_name}')
    plt.xlabel(variable_name)
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(save_path)
    plt.close()
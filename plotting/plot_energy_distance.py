"""
Script to plot energy distance results for different data groups.
"""

import matplotlib.pyplot as plt
import logging
from pathlib import Path
import numpy as np

def plot_energy_distance_results(energy_distance_dict: dict, save_path: str):
    """
    Plots energy distance results for different data groups.

    Args:
        energy_distance_dict (dict): A dictionary where keys are group indices and values are energy distance values.
        save_path (str): The path to save the plot image.
    """
      

    # Plot results
    plt.figure(figsize=(8, 4))
    x = list(range(len(energy_distance_dict)))
    plt.plot(x, [energy_distance_dict[i]['value'] for i in x], marker="o", linestyle="-")
    plt.xlabel("Data group index")
    plt.ylabel("Energy distance")
    plt.title("Energy distance vs Data group index")
    plt.grid(True)
    plt.tight_layout()

    plot_path = Path.cwd() / "energy_distance.png"
    plt.savefig(plot_path, dpi=200)
    logging.info(f"Saved plot to {plot_path}")
    plt.close()
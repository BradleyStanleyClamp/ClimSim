import matplotlib.pyplot as plt
import seaborn as sns

def init_plotting_settings():
    """
    Initializes global plotting settings for matplotlib and seaborn to ensure consistent styling across all plots.
    This includes setting LaTeX fonts, general font sizes, line widths, and grid appearance.
    """
    # Set up matplotlib to use LaTeX fonts
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Computer Modern Roman'
    sns.set_style("whitegrid")
    transparent_plot = True

    # General font and line settings
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['lines.markersize'] = 8

    # Customize grid appearance
    plt.rcParams['grid.color'] = '#CCCCCC'  # Light gray
    plt.rcParams['grid.linestyle'] = '--'   # Dashed lines
    plt.rcParams['grid.linewidth'] = 0.5    # Thin lines
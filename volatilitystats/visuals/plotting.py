import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Optional

def plot_volatility_series(
    vol_dict: Dict[str, pd.Series],
    title: str = "Volatility Comparison",
    figsize: tuple = (12, 5),
    ylabel: str = "Volatility",
    savepath: Optional[str] = None
):
    """
    Plot multiple volatility series for comparison.

    Parameters
    ----------
    vol_dict : dict
        Dictionary mapping label to volatility series.
    title : str
        Plot title.
    figsize : tuple
        Size of the figure.
    ylabel : str
        Y-axis label.
    savepath : str, optional
        If provided, save the figure to this path.
    """
    plt.figure(figsize=figsize)
    for label, series in vol_dict.items():
        series = series.dropna()
        plt.plot(series.index, series.values, label=label)

    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Date")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath)
    else:
        plt.show()

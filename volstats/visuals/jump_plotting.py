import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional

def plot_price_with_jumps(
    df: pd.DataFrame,
    price_column: str,
    jump_series: pd.Series,
    title: str = "Price with Detected Jumps",
    figsize: tuple = (12, 5),
    savepath: Optional[str] = None
):
    """
    Plot price series with jump points marked.

    Parameters
    ----------
    df : pd.DataFrame
        Original price dataframe.
    price_column : str
        Column name containing price.
    jump_series : pd.Series
        Boolean series (indexed by datetime) where True indicates a jump.
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    savepath : str, optional
        If given, save the figure to this path.
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    price = df[price_column].dropna()
    jumps = jump_series[jump_series].index

    plt.figure(figsize=figsize)
    plt.plot(price.index, price.values, label="Price", color="black")
    plt.scatter(jumps, price.loc[jumps], color="red", label="Jumps", zorder=5)

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath)
    else:
        plt.show()

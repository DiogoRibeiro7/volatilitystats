import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Literal
from volatilitystats.utils.confidence import compute_confidence_bands

def plot_forecast_with_confidence_band(
    forecast: pd.Series,
    lower: Optional[pd.Series] = None,
    upper: Optional[pd.Series] = None,
    realized: Optional[pd.Series] = None,
    stderr: Optional[pd.Series] = None,
    ci_method: Literal["normal", "bootstrap"] = "normal",
    z: float = 1.96,
    title: str = "Forecast with Confidence Interval",
    ylabel: str = "Volatility",
    figsize: tuple = (12, 5),
    savepath: Optional[str] = None,
    compute_if_missing: bool = True
):
    """
    Plot forecasted volatility with confidence intervals and optional realized volatility.

    Parameters
    ----------
    forecast : pd.Series
        Point forecast series.
    lower : pd.Series, optional
        Lower bound of confidence interval (ignored if stderr provided).
    upper : pd.Series, optional
        Upper bound of confidence interval (ignored if stderr provided).
    realized : pd.Series, optional
        Realized volatility to compare.
    stderr : pd.Series, optional
        Standard error estimates of forecast.
    ci_method : {"normal", "bootstrap"}
        Method for computing CI if stderr is provided.
    z : float
        Critical value (default 1.96 for 95% CI).
    title : str
        Plot title.
    ylabel : str
        Y-axis label.
    figsize : tuple
        Figure size.
    savepath : str, optional
        File path to save the plot.
    compute_if_missing : bool
        If True, compute confidence bands from stderr if lower/upper not given.
    """
    if stderr is not None and compute_if_missing:
        lower, upper = compute_confidence_bands(forecast, stderr, method=ci_method, z=z)

    plt.figure(figsize=figsize)
    plt.plot(forecast.index, forecast.values, label="Forecast", color="blue")

    if lower is not None and upper is not None:
        plt.fill_between(lower.index, lower.values, upper.values, color="blue", alpha=0.2, label="Confidence Band")

    if realized is not None:
        realized = realized.dropna()
        plt.plot(realized.index, realized.values, label="Realized", color="orange")

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

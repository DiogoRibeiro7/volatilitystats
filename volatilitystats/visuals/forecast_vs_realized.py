import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional
import numpy as np

def plot_forecast_vs_realized(
    forecast: pd.Series,
    realized: pd.Series,
    title: str = "Forecast vs Realized Volatility",
    ylabel: str = "Volatility",
    figsize: tuple = (12, 6),
    savepath: Optional[str] = None,
    show_metrics: bool = True,
    show_residuals: bool = False,
    rolling_corr_window: Optional[int] = None
):
    """
    Plot model-forecasted vs realized volatility over time with optional diagnostics.

    Parameters
    ----------
    forecast : pd.Series
        Forecasted volatility series (e.g., from GARCH).
    realized : pd.Series
        Realized volatility (e.g., Yang-Zhang, TSRV).
    title : str
        Plot title.
    ylabel : str
        Y-axis label.
    figsize : tuple
        Size of the figure.
    savepath : str, optional
        Path to save figure (optional).
    show_metrics : bool
        If True, display MAE and RMSE.
    show_residuals : bool
        If True, plot forecast - realized residuals.
    rolling_corr_window : int, optional
        If set, plot rolling correlation over this window size.
    """
    forecast = forecast.dropna()
    realized = realized.dropna()
    common_index = forecast.index.intersection(realized.index)
    f = forecast[common_index]
    r = realized[common_index]

    residuals = f - r
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals ** 2))

    fig, ax1 = plt.subplots(figsize=figsize)
    ax1.plot(common_index, f, label="Forecasted", color="blue")
    ax1.plot(common_index, r, label="Realized", color="orange")
    ax1.set_title(title)
    ax1.set_ylabel(ylabel)
    ax1.set_xlabel("Date")
    ax1.legend()
    ax1.grid(True)

    if show_metrics:
        ax1.text(0.01, 0.95, f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}",
                 transform=ax1.transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))

    if show_residuals:
        ax2 = ax1.twinx()
        ax2.plot(common_index, residuals, label="Residuals", color="gray", alpha=0.4)
        ax2.set_ylabel("Residuals")
        ax2.legend(loc="upper right")

    if rolling_corr_window:
        rolling_corr = f.rolling(rolling_corr_window).corr(r)
        fig, ax3 = plt.subplots(figsize=(figsize[0], 2))
        ax3.plot(common_index, rolling_corr, label=f"Rolling Corr ({rolling_corr_window})", color="green")
        ax3.set_title("Rolling Correlation")
        ax3.set_ylabel("Correlation")
        ax3.grid(True)
        ax3.set_xlabel("Date")
        ax3.legend()
        plt.tight_layout()

    plt.tight_layout()
    if savepath:
        fig.savefig(savepath)
    else:
        plt.show()
        
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.diagnostic import acorr_ljungbox
from typing import Optional

def plot_residual_diagnostics(
    residuals: pd.Series,
    lags: int = 20,
    title_prefix: str = "",
    savepath: Optional[str] = None
):
    """
    Diagnostic plots for model residuals: histogram, ACF, and Ljung-Box test.

    Parameters
    ----------
    residuals : pd.Series
        Residual series (model - realized).
    lags : int
        Number of lags for ACF and Ljung-Box test.
    title_prefix : str
        Optional title prefix.
    savepath : str, optional
        Path to save figure.
    """
    fig, axs = plt.subplots(1, 3, figsize=(18, 4))

    # Histogram
    sns.histplot(residuals, bins=30, kde=True, ax=axs[0], color="gray")
    axs[0].set_title(f"{title_prefix}Residual Histogram")
    axs[0].set_xlabel("Residual")

    # ACF plot
    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(residuals, ax=axs[1], lags=lags, title=f"{title_prefix}Residual ACF")

    # Ljung-Box test
    lb_test = acorr_ljungbox(residuals, lags=[lags], return_df=True)
    p_value = lb_test["lb_pvalue"].values[0]
    axs[2].axis("off")
    axs[2].text(0.5, 0.5, f"Ljung-Box p-value (lag {lags}):\n{p_value:.4f}",
                ha='center', va='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    axs[2].set_title(f"{title_prefix}Ljung-Box Test")

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath)
    else:
        plt.show()

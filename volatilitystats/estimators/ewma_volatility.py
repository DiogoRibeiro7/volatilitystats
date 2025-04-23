import numpy as np
import pandas as pd

def ewma_volatility(
    returns: pd.Series,
    lambda_: float = 0.94,
    annualization_factor: int = 252
) -> pd.Series:
    """
    Computes exponentially weighted moving volatility (RiskMetrics style).

    Parameters
    ----------
    returns : pd.Series
        Series of log returns.
    lambda_ : float
        Decay factor, e.g., 0.94 (default) for daily data.
    annualization_factor : int
        Number of trading periods in a year (e.g., 252).

    Returns
    -------
    pd.Series
        EWMA annualized volatility.
    """
    returns = returns.fillna(0)
    squared_returns = returns**2
    ewma_var = squared_returns.ewm(alpha=1 - lambda_, adjust=False).mean()
    annualized_vol = np.sqrt(ewma_var) * np.sqrt(annualization_factor)

    return annualized_vol.rename("EWMAVolatility")

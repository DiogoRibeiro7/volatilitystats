import numpy as np
import pandas as pd

def parkinson_volatility(
    df: pd.DataFrame,
    window: int = 20,
    annualization_factor: int = 252
) -> pd.Series:
    """
    Computes the Parkinson volatility estimator using high and low prices.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'High' and 'Low' columns.
    window : int
        Size of the rolling window for volatility computation.
    annualization_factor : int
        Number of trading periods in a year (e.g., 252 for daily data).

    Returns
    -------
    pd.Series
        Annualized Parkinson volatility.
    """
    if 'High' not in df.columns or 'Low' not in df.columns:
        raise ValueError("Input DataFrame must contain 'High' and 'Low' columns.")

    hl_log_sq = np.log(df['High'] / df['Low']) ** 2
    factor = 1 / (4 * np.log(2))
    rolling_var = hl_log_sq.rolling(window).mean() * factor
    annualized_vol = np.sqrt(rolling_var) * np.sqrt(annualization_factor)

    return annualized_vol.rename("ParkinsonVolatility")

import numpy as np
import pandas as pd

def standard_volatility(
    df: pd.DataFrame,
    window: int = 20,
    annualization_factor: int = 252
) -> pd.Series:
    """
    Computes the standard (close-to-close) volatility using log returns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a 'Close' column containing price data.
    window : int
        Size of the rolling window for volatility computation.
    annualization_factor : int
        Number of trading periods in a year (e.g., 252 for daily data).

    Returns
    -------
    pd.Series
        Annualized rolling standard deviation of log returns.
    """
    if 'Close' not in df.columns:
        raise ValueError("Input DataFrame must contain a 'Close' column.")

    log_returns = np.log(df['Close'] / df['Close'].shift(1))
    rolling_std = log_returns.rolling(window).std()
    annualized_vol = rolling_std * np.sqrt(annualization_factor)

    return annualized_vol.rename("StandardVolatility")

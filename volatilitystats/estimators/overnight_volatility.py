import numpy as np
import pandas as pd

def overnight_volatility(
    df: pd.DataFrame,
    window: int = 20,
    annualization_factor: int = 252
) -> pd.Series:
    """
    Computes overnight (close-to-open) volatility.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'Close' and 'Open' columns.
    window : int
        Size of the rolling window for volatility computation.
    annualization_factor : int
        Number of trading periods in a year (e.g., 252 for daily data).

    Returns
    -------
    pd.Series
        Annualized overnight volatility.
    """
    if 'Close' not in df.columns or 'Open' not in df.columns:
        raise ValueError("Input DataFrame must contain 'Close' and 'Open' columns.")

    overnight_returns = np.log(df['Open'] / df['Close'].shift(1))
    rolling_std = overnight_returns.rolling(window).std()
    annualized_vol = rolling_std * np.sqrt(annualization_factor)

    return annualized_vol.rename("OvernightVolatility")

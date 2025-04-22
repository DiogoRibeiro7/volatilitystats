import numpy as np
import pandas as pd

def garman_klass_volatility(
    df: pd.DataFrame,
    window: int = 20,
    annualization_factor: int = 252
) -> pd.Series:
    """
    Computes the Garman-Klass volatility estimator using OHLC data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'Open', 'High', 'Low', and 'Close' columns.
    window : int
        Size of the rolling window for volatility computation.
    annualization_factor : int
        Number of trading periods in a year (e.g., 252 for daily data).

    Returns
    -------
    pd.Series
        Annualized Garman-Klass volatility.
    """
    required_cols = {'Open', 'High', 'Low', 'Close'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Input DataFrame must contain columns: {required_cols}")

    log_hl = np.log(df['High'] / df['Low'])
    log_co = np.log(df['Close'] / df['Open'])

    gk_var = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
    rolling_gk = gk_var.rolling(window).mean()
    annualized_vol = np.sqrt(rolling_gk) * np.sqrt(annualization_factor)

    return annualized_vol.rename("GarmanKlassVolatility")

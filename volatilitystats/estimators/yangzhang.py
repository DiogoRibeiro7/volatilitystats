import numpy as np
import pandas as pd

def yang_zhang_volatility(
    df: pd.DataFrame,
    window: int = 20,
    annualization_factor: int = 252
) -> pd.Series:
    """
    Computes the Yang-Zhang volatility estimator using open, high, low, and close prices.

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
        Annualized Yang-Zhang volatility.
    """
    required_cols = {'Open', 'High', 'Low', 'Close'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Input DataFrame must contain columns: {required_cols}")

    log_oc = np.log(df['Close'] / df['Open'])
    log_oo = np.log(df['Open'] / df['Open'].shift(1))
    log_co = np.log(df['Close'] / df['Open'].shift(1))

    ro = log_oo.rolling(window).var()
    rc = log_oc.rolling(window).var()
    rs = 0.5 * (np.log(df['High'] / df['Low']) ** 2) - (2 * np.log(2) - 1) * rc

    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    yz_var = ro * k + rc * (1 - k) + rs.rolling(window).mean()
    annualized_vol = np.sqrt(yz_var) * np.sqrt(annualization_factor)

    return annualized_vol.rename("YangZhangVolatility")

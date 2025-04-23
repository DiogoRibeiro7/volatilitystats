import numpy as np
import pandas as pd

def rogers_satchell_volatility(
    df: pd.DataFrame,
    window: int = 20,
    annualization_factor: int = 252
) -> pd.Series:
    """
    Computes the Rogers-Satchell volatility estimator using OHLC data.

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
        Annualized Rogers-Satchell volatility.
    """
    required_cols = {'Open', 'High', 'Low', 'Close'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Input DataFrame must contain columns: {required_cols}")

    ho = np.log(df['High'] / df['Open'])
    hc = np.log(df['High'] / df['Close'])
    lo = np.log(df['Low'] / df['Open'])
    lc = np.log(df['Low'] / df['Close'])

    rs_range = ho * hc + lo * lc
    rolling_rs = rs_range.rolling(window).mean()
    annualized_vol = np.sqrt(rolling_rs) * np.sqrt(annualization_factor)

    return annualized_vol.rename("RogersSatchellVolatility")

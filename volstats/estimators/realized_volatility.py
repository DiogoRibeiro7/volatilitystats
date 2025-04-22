import numpy as np
import pandas as pd

def realized_volatility(
    df: pd.DataFrame,
    returns_col: str = "returns",
    window: int = 20,
    annualization_factor: int = 252
) -> pd.Series:
    """
    Computes realized volatility from intraday squared returns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a datetime index and intraday squared returns
        grouped by day (one row per return).
    returns_col : str
        Column name with intraday returns (not squared).
    window : int
        Rolling window size for daily aggregation.
    annualization_factor : int
        Number of trading periods in a year (e.g., 252).

    Returns
    -------
    pd.Series
        Daily annualized realized volatility.
    """
    if returns_col not in df.columns:
        raise ValueError(f"Column '{returns_col}' not found in input DataFrame.")

    df = df.copy()
    df['date'] = df.index.date
    df['squared'] = df[returns_col] ** 2

    daily_var = df.groupby('date')['squared'].sum()
    daily_vol = np.sqrt(daily_var)
    annualized_vol = daily_vol.rolling(window).mean() * np.sqrt(annualization_factor)

    return annualized_vol.rename("RealizedVolatility")

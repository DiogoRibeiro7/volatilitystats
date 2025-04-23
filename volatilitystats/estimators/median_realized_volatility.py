import numpy as np
import pandas as pd
from typing import Union

def median_realized_volatility(
    returns: Union[np.ndarray, pd.Series]
) -> float:
    """
    Median Realized Volatility estimator.

    Parameters
    ----------
    returns : array-like
        High-frequency log returns within a day.

    Returns
    -------
    float
        Median-based realized variance estimate.

    Notes
    -----
    More robust to extreme outliers than squared-based estimators.
    """
    returns = np.asarray(returns)
    med_rv = np.median(np.abs(returns))
    return med_rv * np.sqrt(np.pi / 2)

def median_rv_series(
    df: pd.DataFrame,
    price_column: str,
    time_column: str,
    freq: str = "1D"
) -> pd.Series:
    """
    Compute median-based realized volatility for each time group.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing timestamps and price data.
    price_column : str
        Column name for price.
    time_column : str
        Column name for timestamp.
    freq : str, optional
        Resampling frequency, e.g., '1D' for daily, by default "1D".

    Returns
    -------
    pd.Series
        Time series of median realized volatility.
    """
    df = df.copy()
    df[time_column] = pd.to_datetime(df[time_column])
    df.set_index(time_column, inplace=True)
    df.sort_index(inplace=True)

    log_price = np.log(df[price_column])
    log_returns = log_price.diff().dropna()

    result = log_returns.groupby(pd.Grouper(freq=freq)).apply(
        lambda x: median_realized_volatility(x.values)
    )
    result.name = "Median Realized Volatility"
    return result

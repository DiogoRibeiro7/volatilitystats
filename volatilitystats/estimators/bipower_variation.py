import numpy as np
import pandas as pd
from typing import Union

def bipower_variation(
    returns: Union[np.ndarray, pd.Series]
) -> float:
    """
    Bipower Variation estimator of integrated volatility (robust to jumps).

    Parameters
    ----------
    returns : array-like
        High-frequency log returns within a day.

    Returns
    -------
    float
        Bipower variation estimate.

    References
    ----------
    Barndorff-Nielsen and Shephard (2004), "Power and bipower variation with stochastic volatility and jumps"
    """
    returns = np.asarray(returns)
    abs_r = np.abs(returns)

    if len(abs_r) < 2:
        return 0.0

    prod = abs_r[1:] * abs_r[:-1]
    mu1 = np.sqrt(2 / np.pi)
    return mu1**-2 * np.sum(prod)

def bipower_variation_series(
    df: pd.DataFrame,
    price_column: str,
    time_column: str,
    freq: str = "1D"
) -> pd.Series:
    """
    Compute bipower variation by time group.

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
        Time series of bipower variation.
    """
    df = df.copy()
    df[time_column] = pd.to_datetime(df[time_column])
    df.set_index(time_column, inplace=True)
    df.sort_index(inplace=True)

    log_price = np.log(df[price_column])
    log_returns = log_price.diff().dropna()

    result = log_returns.groupby(pd.Grouper(freq=freq)).apply(
        lambda x: bipower_variation(x.values)
    )
    result.name = "Bipower Variation"
    return result

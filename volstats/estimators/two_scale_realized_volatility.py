import numpy as np
import pandas as pd
from typing import Union

def two_scale_realized_volatility(
    returns: Union[np.ndarray, pd.Series],
    K: int = 2
) -> float:
    """
    Two-Scale Realized Volatility (TSRV) estimator.

    Parameters
    ----------
    returns : array-like
        High-frequency log returns within a day.
    K : int, optional
        Number of sub-samples (e.g., K=2 means two sub-grids), by default 2.

    Returns
    -------
    float
        Two-Scale Realized Variance estimate.

    References
    ----------
    Zhang, Mykland, Aït-Sahalia (2005). "A Tale of Two Time Scales." Journal of the American Statistical Association.
    """
    returns = np.asarray(returns)
    n = len(returns)

    if n < K:
        raise ValueError("Number of returns must be greater than number of sub-grids (K).")

    rv = np.sum(returns ** 2)
    subsampled_rvs = [
        np.sum(returns[k::K] ** 2) * K for k in range(K)
    ]

    rv_bar = np.mean(subsampled_rvs)
    tsrv = rv_bar - (rv / n)
    return max(tsrv, 0.0)

def tsrv_series(
    df: pd.DataFrame,
    price_column: str,
    time_column: str,
    freq: str = "1D",
    K: int = 2
) -> pd.Series:
    """
    Compute TSRV for each group defined by resampling frequency.

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
    K : int
        Number of sub-grids.

    Returns
    -------
    pd.Series
        Time series of TSRV values.
    """
    df = df.copy()
    df[time_column] = pd.to_datetime(df[time_column])
    df.set_index(time_column, inplace=True)
    df.sort_index(inplace=True)

    log_price = np.log(df[price_column])
    log_returns = log_price.diff().dropna()

    tsrv_by_period = log_returns.groupby(pd.Grouper(freq=freq)).apply(lambda x: two_scale_realized_volatility(x.values, K=K))
    tsrv_by_period.name = f"TSRV({K})"
    return tsrv_by_period

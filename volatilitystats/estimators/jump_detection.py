import numpy as np
import pandas as pd
from typing import Union

def barndorff_nielsen_shephard_jump_test(
    returns: Union[np.ndarray, pd.Series],
    threshold: float = 4.0
) -> bool:
    """
    Jump detection based on the Barndorff-Nielsen & Shephard test.

    Compares realized variance (RV) and bipower variation (BV).
    A statistically significant difference suggests the presence of jumps.

    Parameters
    ----------
    returns : array-like
        High-frequency log returns.
    threshold : float
        Z-score threshold above which a jump is detected.

    Returns
    -------
    bool
        True if a jump is detected, False otherwise.

    References
    ----------
    Barndorff-Nielsen and Shephard (2004)
    "Power and Bipower Variation with Stochastic Volatility and Jumps"
    """
    returns = np.asarray(returns)
    n = len(returns)
    if n < 3:
        return False

    rv = np.sum(returns ** 2)
    bv = np.sum(np.abs(returns[1:]) * np.abs(returns[:-1])) * (np.pi / 2)

    diff = rv - bv
    var = (np.pi / 2 + np.pi - 5) * (np.sum(np.abs(returns) ** 4)) / n
    if var <= 0:
        return False

    z_score = diff / np.sqrt(var / n)
    return np.abs(z_score) > threshold

def detect_jumps_series(
    df: pd.DataFrame,
    price_column: str,
    time_column: str,
    freq: str = "1D",
    threshold: float = 4.0
) -> pd.Series:
    """
    Detect jumps in each period using the BNS test.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with timestamps and price.
    price_column : str
        Name of price column.
    time_column : str
        Name of timestamp column.
    freq : str
        Resampling frequency.
    threshold : float
        Z-score threshold.

    Returns
    -------
    pd.Series
        Boolean time series indicating jump presence.
    """
    df = df.copy()
    df[time_column] = pd.to_datetime(df[time_column])
    df.set_index(time_column, inplace=True)
    df.sort_index(inplace=True)

    log_price = np.log(df[price_column])
    log_returns = log_price.diff().dropna()

    result = log_returns.groupby(pd.Grouper(freq=freq)).apply(
        lambda x: barndorff_nielsen_shephard_jump_test(x.values, threshold=threshold)
    )
    result.name = "Jump Detected"
    return result

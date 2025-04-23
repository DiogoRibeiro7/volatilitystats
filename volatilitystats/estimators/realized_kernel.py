import numpy as np
import pandas as pd
from typing import Union

def realized_kernel(
    returns: Union[np.ndarray, pd.Series],
    kernel: str = "bartlett",
    bandwidth: int = None
) -> float:
    """
    Realized Kernel estimator for high-frequency volatility.

    Parameters
    ----------
    returns : array-like
        High-frequency log returns.
    kernel : str, optional
        Kernel type: 'bartlett', 'parzen', or 'uniform'. Default is 'bartlett'.
    bandwidth : int, optional
        Bandwidth (number of lags). If None, uses default sqrt(n).

    Returns
    -------
    float
        Realized kernel estimate.

    References
    ----------
    Barndorff-Nielsen, Hansen, Lunde, and Shephard (2008)
    "Designing Realized Kernels to Measure the Ex-Post Variation of Equity Prices in the Presence of Noise"
    """
    returns = np.asarray(returns)
    n = len(returns)
    if bandwidth is None:
        bandwidth = int(np.sqrt(n))

    gamma_0 = np.sum(returns ** 2)
    rk = gamma_0

    for h in range(1, bandwidth + 1):
        gamma_h = np.sum(returns[h:] * returns[:-h])
        weight = kernel_weight(h, bandwidth, kernel)
        rk += 2 * weight * gamma_h

    return max(rk, 0.0)

def kernel_weight(h: int, bandwidth: int, kernel: str) -> float:
    """Kernel weights for realized kernel estimator."""
    x = h / (bandwidth + 1)
    if kernel == "bartlett":
        return 1 - x
    elif kernel == "parzen":
        return 1 - 6 * x**2 + 6 * x**3 if x <= 0.5 else 2 * (1 - x)**3
    elif kernel == "uniform":
        return 1.0
    else:
        raise ValueError(f"Unsupported kernel type: {kernel}")

def realized_kernel_series(
    df: pd.DataFrame,
    price_column: str,
    time_column: str,
    freq: str = "1D",
    kernel: str = "bartlett",
    bandwidth: int = None
) -> pd.Series:
    """
    Compute realized kernel estimator for each time group.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with timestamps and price data.
    price_column : str
        Name of the price column.
    time_column : str
        Name of the timestamp column.
    freq : str
        Frequency to group by (e.g., '1D').
    kernel : str
        Kernel function type.
    bandwidth : int, optional
        Bandwidth parameter.

    Returns
    -------
    pd.Series
        Time series of realized kernel volatility.
    """
    df = df.copy()
    df[time_column] = pd.to_datetime(df[time_column])
    df.set_index(time_column, inplace=True)
    df.sort_index(inplace=True)

    log_price = np.log(df[price_column])
    log_returns = log_price.diff().dropna()

    result = log_returns.groupby(pd.Grouper(freq=freq)).apply(
        lambda x: realized_kernel(x.values, kernel=kernel, bandwidth=bandwidth)
    )
    result.name = f"Realized Kernel ({kernel})"
    return result

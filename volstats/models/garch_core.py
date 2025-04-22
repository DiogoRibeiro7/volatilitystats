import numpy as np
import pandas as pd
from typing import Sequence

def garch(
    returns: pd.Series,
    omega: float,
    alpha: Sequence[float],
    beta: Sequence[float],
    initial_vol: float | None = None
) -> pd.Series:
    """
    General GARCH(p, q) conditional volatility estimator.

    Parameters
    ----------
    returns : pd.Series
        Series of log returns.
    omega : float
        Constant term in the GARCH model.
    alpha : Sequence[float]
        ARCH coefficients (lags of squared residuals).
    beta : Sequence[float]
        GARCH coefficients (lags of conditional variance).
    initial_vol : float, optional
        Initial volatility. If None, uses variance of returns.

    Returns
    -------
    pd.Series
        Conditional volatility time series.
    """
    q = len(alpha)
    p = len(beta)
    max_lag = max(p, q)
    n = len(returns)

    eps = returns.fillna(0).values
    sigma2 = np.zeros(n)

    if initial_vol is None:
        sigma2[:max_lag] = np.var(eps)
    else:
        sigma2[:max_lag] = initial_vol**2

    for t in range(max_lag, n):
        arch_term = sum(alpha[i] * eps[t - i - 1] ** 2 for i in range(q))
        garch_term = sum(beta[j] * sigma2[t - j - 1] for j in range(p))
        sigma2[t] = omega + arch_term + garch_term

    return pd.Series(np.sqrt(sigma2), index=returns.index, name=f"GARCH({p},{q}) Volatility")

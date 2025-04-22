import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Sequence

def harch_log_likelihood(params: Sequence[float], returns: pd.Series, lags: Sequence[int]) -> float:
    """
    Log-likelihood function for the HARCH model.

    Parameters
    ----------
    params : list of float
        Model parameters: [omega, alpha_1, ..., alpha_k]
    returns : pd.Series
        Log returns.
    lags : list of int
        Lag window sizes (e.g., [1, 5, 22])

    Returns
    -------
    float
        Negative log-likelihood.
    """
    omega = params[0]
    alpha = params[1:]

    y = returns.fillna(0).values
    n = len(y)
    sigma2 = np.full(n, np.var(y))

    for t in range(max(lags), n):
        sigma2[t] = omega + sum(
            alpha[i] * np.mean(y[t - lags[i]:t] ** 2)
            for i in range(len(lags))
        )
        if sigma2[t] <= 0:
            return np.inf

    log_lik = -0.5 * (np.log(2 * np.pi) + np.log(sigma2) + y**2 / sigma2)
    total_ll = -np.sum(log_lik)

    if not np.isfinite(total_ll):
        return np.inf

    return total_ll

def estimate_harch_params(returns: pd.Series, lags: Sequence[int]) -> dict:
    """
    Estimate HARCH model parameters via MLE.

    Parameters
    ----------
    returns : pd.Series
        Log returns.
    lags : list of int
        Lag window sizes (e.g., [1, 5, 22])

    Returns
    -------
    dict
        Dictionary of estimated parameters and conditional volatility.
    """
    k = len(lags)
    initial_guess = [1e-6] + [0.05] * k
    bounds = [(1e-6, None)] + [(1e-6, 1)] * k

    result = minimize(
        harch_log_likelihood,
        initial_guess,
        args=(returns, lags),
        bounds=bounds,
        method="L-BFGS-B"
    )

    omega = result.x[0]
    alpha = result.x[1:]

    y = returns.fillna(0).values
    n = len(y)
    sigma2 = np.full(n, np.var(y))

    for t in range(max(lags), n):
        sigma2[t] = omega + sum(
            alpha[i] * np.mean(y[t - lags[i]:t] ** 2)
            for i in range(len(lags))
        )

    volatility = pd.Series(np.sqrt(sigma2), index=returns.index, name="HARCH Volatility")

    return {
        "omega": omega,
        "alpha": alpha,
        "volatility": volatility
    }

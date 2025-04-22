import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Sequence
from volstats.utils.confidence import compute_confidence_bands

def harch_log_likelihood(params: Sequence[float], returns: pd.Series, lags: Sequence[int]) -> float:
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
    return np.inf if not np.isfinite(total_ll) else total_ll

def estimate_harch_params(
    returns: pd.Series,
    lags: Sequence[int],
    with_confidence: bool = False,
    stderr_fraction: float = 0.1
) -> dict:
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

    output = {
        "omega": omega,
        "alpha": alpha,
        "volatility": volatility
    }

    if with_confidence:
        stderr = pd.Series(stderr_fraction * volatility, index=volatility.index)
        lower, upper = compute_confidence_bands(volatility, stderr)
        output["stderr"] = stderr
        output["lower"] = lower
        output["upper"] = upper

    return output

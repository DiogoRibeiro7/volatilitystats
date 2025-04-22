import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Sequence
from volstats.utils.confidence import compute_confidence_bands

def egarch_log_likelihood(params: Sequence[float], returns: pd.Series, p: int, q: int) -> float:
    """
    Negative log-likelihood for EGARCH(p, q) under normal errors.
    """
    omega = params[0]
    alpha = params[1 : 1 + q]
    gamma = params[1 + q : 1 + 2 * q]
    beta = params[1 + 2 * q : 1 + 2 * q + p]

    eps = returns.fillna(0).values
    n = len(eps)
    log_sigma2 = np.zeros(n)
    log_sigma2[:max(p, q)] = np.log(np.var(eps))

    for t in range(max(p, q), n):
        z_terms = [eps[t - i - 1] / np.exp(0.5 * log_sigma2[t - i - 1]) for i in range(q)]
        arch = sum(alpha[i] * (np.abs(z_terms[i]) - np.sqrt(2 / np.pi)) + gamma[i] * z_terms[i] for i in range(q))
        garch = sum(beta[j] * log_sigma2[t - j - 1] for j in range(p))
        log_sigma2[t] = omega + arch + garch

    log_lik = -0.5 * (np.log(2 * np.pi) + log_sigma2 + eps**2 / np.exp(log_sigma2))
    log_lik_sum = -np.sum(log_lik)
    return np.inf if not np.isfinite(log_lik_sum) else log_lik_sum

def estimate_egarch_params(
    returns: pd.Series,
    p: int = 1,
    q: int = 1,
    with_confidence: bool = False,
    stderr_fraction: float = 0.1
) -> dict:
    """
    Estimate EGARCH(p, q) parameters via MLE.

    Parameters
    ----------
    returns : pd.Series
        Log returns.
    p : int
        Order of lagged log-volatility terms.
    q : int
        Order of lagged innovation terms.
    with_confidence : bool
        If True, compute confidence bands.
    stderr_fraction : float
        Multiplier to simulate stderr when not estimated directly.

    Returns
    -------
    dict
        Model parameters, volatility series, and optional confidence intervals.
    """
    k = 1 + 2 * q + p
    initial_guess = [0.0] + [0.05] * q + [0.0] * q + [0.9 / p] * p
    bounds = [(-10, 10)] + [(1e-6, 1.0)] * q + [(-1, 1)] * q + [(1e-6, 1.0)] * p

    result = minimize(
        egarch_log_likelihood,
        initial_guess,
        args=(returns, p, q),
        bounds=bounds,
        method="L-BFGS-B"
    )

    omega = result.x[0]
    alpha = result.x[1 : 1 + q]
    gamma = result.x[1 + q : 1 + 2 * q]
    beta = result.x[1 + 2 * q : 1 + 2 * q + p]

    eps = returns.fillna(0).values
    n = len(eps)
    log_sigma2 = np.zeros(n)
    log_sigma2[:max(p, q)] = np.log(np.var(eps))

    for t in range(max(p, q), n):
        z_terms = [eps[t - i - 1] / np.exp(0.5 * log_sigma2[t - i - 1]) for i in range(q)]
        arch = sum(alpha[i] * (np.abs(z_terms[i]) - np.sqrt(2 / np.pi)) + gamma[i] * z_terms[i] for i in range(q))
        garch = sum(beta[j] * log_sigma2[t - j - 1] for j in range(p))
        log_sigma2[t] = omega + arch + garch

    volatility = pd.Series(np.exp(0.5 * log_sigma2), index=returns.index, name=f"EGARCH({p},{q}) Volatility")

    output = {
        "omega": omega,
        "alpha": alpha,
        "gamma": gamma,
        "beta": beta,
        "volatility": volatility
    }

    if with_confidence:
        stderr = pd.Series(stderr_fraction * volatility, index=volatility.index)
        lower, upper = compute_confidence_bands(volatility, stderr)
        output["stderr"] = stderr
        output["lower"] = lower
        output["upper"] = upper

    return output

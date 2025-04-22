import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Sequence

def egarch_log_likelihood(params: Sequence[float], returns: pd.Series, p: int, q: int) -> float:
    """
    Negative log-likelihood for EGARCH(p, q) under normal errors.

    Parameters
    ----------
    params : Sequence[float]
        Model parameters: [omega, alpha_1..q, gamma_1..q, beta_1..p].
    returns : pd.Series
        Log returns.
    p : int
        Number of lagged log-volatility terms.
    q : int
        Number of lagged innovation terms.

    Returns
    -------
    float
        Negative log-likelihood value.
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

    if not np.isfinite(log_lik_sum):
        return np.inf

    return log_lik_sum

def estimate_egarch_params(
    returns: pd.Series,
    p: int = 1,
    q: int = 1
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

    Returns
    -------
    dict
        Estimated parameters and volatility series.
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

    return {
        "omega": omega,
        "alpha": alpha,
        "gamma": gamma,
        "beta": beta,
        "volatility": volatility
    }

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Sequence

def gjr_garch_log_likelihood(
    params: Sequence[float],
    returns: pd.Series,
    p: int,
    q: int
) -> float:
    """
    Negative log-likelihood for GJR-GARCH(p, q) model.

    Parameters
    ----------
    params : Sequence[float]
        Model parameters: [omega, alpha_1..q, gamma_1..q, beta_1..p].
    returns : pd.Series
        Log returns.
    p : int
        Order of GARCH terms.
    q : int
        Order of ARCH and asymmetry terms.

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
    sigma2 = np.zeros(n)
    sigma2[:max(p, q)] = np.var(eps)

    for t in range(max(p, q), n):
        arch_term = sum(alpha[i] * eps[t - i - 1] ** 2 for i in range(q))
        asym_term = sum(gamma[i] * eps[t - i - 1] ** 2 * (eps[t - i - 1] < 0) for i in range(q))
        garch_term = sum(beta[j] * sigma2[t - j - 1] for j in range(p))
        sigma2[t] = omega + arch_term + asym_term + garch_term
        if sigma2[t] <= 0:
            return np.inf

    log_lik = -0.5 * (np.log(2 * np.pi) + np.log(sigma2) + eps**2 / sigma2)
    return -np.sum(log_lik)

def estimate_gjr_garch_params(
    returns: pd.Series,
    p: int = 1,
    q: int = 1
) -> dict:
    """
    Estimate GJR-GARCH(p, q) parameters via MLE.

    Parameters
    ----------
    returns : pd.Series
        Log returns.
    p : int
        Number of GARCH terms.
    q : int
        Number of ARCH/asymmetry terms.

    Returns
    -------
    dict
        Estimated parameters and volatility series.
    """
    k = 1 + 2 * q + p
    initial_guess = [1e-6] + [0.05] * q + [0.05] * q + [0.9 / p] * p
    bounds = [(1e-6, 1.0)] + [(1e-6, 1.0)] * q + [(0.0, 1.0)] * q + [(1e-6, 1.0)] * p

    result = minimize(
        gjr_garch_log_likelihood,
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
    sigma2 = np.zeros(n)
    sigma2[:max(p, q)] = np.var(eps)

    for t in range(max(p, q), n):
        arch_term = sum(alpha[i] * eps[t - i - 1] ** 2 for i in range(q))
        asym_term = sum(gamma[i] * eps[t - i - 1] ** 2 * (eps[t - i - 1] < 0) for i in range(q))
        garch_term = sum(beta[j] * sigma2[t - j - 1] for j in range(p))
        sigma2[t] = omega + arch_term + asym_term + garch_term

    volatility = pd.Series(np.sqrt(sigma2), index=returns.index, name=f"GJR-GARCH({p},{q}) Volatility")

    return {
        "omega": omega,
        "alpha": alpha,
        "gamma": gamma,
        "beta": beta,
        "volatility": volatility
    }

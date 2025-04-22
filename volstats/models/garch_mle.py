import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Sequence, Union
from volstats.models import garch

def garch_log_likelihood(
    params: Sequence[float],
    returns: pd.Series,
    p: int,
    q: int
) -> float:
    """
    Negative log-likelihood for GARCH(p, q) under Gaussian errors.

    Parameters
    ----------
    params : Sequence[float]
        Model parameters: [omega, alpha_1, ..., alpha_q, beta_1, ..., beta_p].
    returns : pd.Series
        Log returns.
    p : int
        Number of GARCH terms (lags of conditional variance).
    q : int
        Number of ARCH terms (lags of squared residuals).

    Returns
    -------
    float
        Negative log-likelihood value.
    """
    omega = params[0]
    alpha = params[1 : 1 + q]
    beta = params[1 + q : 1 + q + p]

    eps = returns.fillna(0).values
    n = len(eps)
    sigma2 = np.zeros(n)
    sigma2[:max(p, q)] = np.var(eps.astype(float))

    for t in range(max(p, q), n):
        arch_term = sum(alpha[i] * eps[t - i - 1] ** 2 for i in range(q))
        garch_term = sum(beta[j] * sigma2[t - j - 1] for j in range(p))
        sigma2[t] = omega + arch_term + garch_term
        if sigma2[t] <= 0:
            return np.inf

    log_lik = -0.5 * (np.log(2 * np.pi) + np.log(sigma2) + eps**2 / sigma2)
    return -np.sum(log_lik)

def estimate_garch_params(
    returns: pd.Series,
    p: int = 1,
    q: int = 1,
    bounds: Union[Sequence[tuple], None] = None
) -> dict:
    """
    Estimate GARCH(p, q) parameters via maximum likelihood.

    Parameters
    ----------
    returns : pd.Series
        Log returns.
    p : int
        Number of GARCH terms.
    q : int
        Number of ARCH terms.
    bounds : Sequence[tuple], optional
        Bounds for optimization. If None, uses default positive bounds.

    Returns
    -------
    dict
        Dictionary with omega, alpha list, beta list, and volatility series.
    """
    k = 1 + q + p
    if bounds is None:
        bounds = [(1e-6, 1.0)] * k

    initial_guess = [1e-6] + [0.05] * q + [0.9 / p] * p

    result = minimize(
        garch_log_likelihood,
        initial_guess,
        args=(returns, p, q),
        bounds=bounds,
        method="L-BFGS-B"
    )

    omega = result.x[0]
    alpha = result.x[1 : 1 + q]
    beta = result.x[1 + q : 1 + q + p]
    vol = garch(returns, omega, alpha, beta)

    return {
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "volatility": vol
    }

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Sequence
from volatilitystats.utils.confidence import compute_confidence_bands

def gjr_garch_log_likelihood(
    params: Sequence[float],
    returns: pd.Series,
    p: int,
    q: int
) -> float:
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
    q: int = 1,
    with_confidence: bool = False,
    stderr_fraction: float = 0.1
) -> dict:
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

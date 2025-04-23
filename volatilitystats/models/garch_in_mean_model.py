import numpy as np
import pandas as pd
from scipy.optimize import minimize
from volatilitystats.utils.confidence import compute_confidence_bands

def garch_in_mean_log_likelihood(params: np.ndarray, returns: pd.Series, p: int, q: int) -> float:
    mu = params[0]
    lmbda = params[1]
    omega = params[2]
    alpha = params[3 : 3 + q]
    beta = params[3 + q : 3 + q + p]

    y = returns.fillna(0).values
    n = len(y)
    eps = np.zeros(n)
    sigma2 = np.zeros(n)
    sigma2[:max(p, q)] = np.var(y)

    for t in range(max(p, q), n):
        eps[t - 1] = y[t - 1] - mu - lmbda * np.sqrt(sigma2[t - 1])
        arch = sum(alpha[i] * eps[t - i - 1] ** 2 for i in range(q))
        garch = sum(beta[j] * sigma2[t - j - 1] for j in range(p))
        sigma2[t] = omega + arch + garch
        if sigma2[t] <= 0:
            return np.inf

    eps[-1] = y[-1] - mu - lmbda * np.sqrt(sigma2[-1])
    log_lik = -0.5 * (np.log(2 * np.pi) + np.log(sigma2) + eps**2 / sigma2)
    total_ll = -np.sum(log_lik)
    return np.inf if not np.isfinite(total_ll) else total_ll

def estimate_garch_in_mean_params(
    returns: pd.Series,
    p: int = 1,
    q: int = 1,
    with_confidence: bool = False,
    stderr_fraction: float = 0.1
) -> dict:
    k = 3 + q + p
    initial_guess = [0.0, 0.0, 1e-6] + [0.05] * q + [0.9 / p] * p
    bounds = [(-10, 10), (-5, 5), (1e-6, 10)] + [(1e-6, 1)] * q + [(1e-6, 1)] * p

    result = minimize(
        garch_in_mean_log_likelihood,
        initial_guess,
        args=(returns, p, q),
        method="L-BFGS-B",
        bounds=bounds
    )

    mu = result.x[0]
    lmbda = result.x[1]
    omega = result.x[2]
    alpha = result.x[3 : 3 + q]
    beta = result.x[3 + q : 3 + q + p]

    y = returns.fillna(0).values
    n = len(y)
    eps = np.zeros(n)
    sigma2 = np.zeros(n)
    sigma2[:max(p, q)] = np.var(y)

    for t in range(max(p, q), n):
        eps[t - 1] = y[t - 1] - mu - lmbda * np.sqrt(sigma2[t - 1])
        arch = sum(alpha[i] * eps[t - i - 1] ** 2 for i in range(q))
        garch = sum(beta[j] * sigma2[t - j - 1] for j in range(p))
        sigma2[t] = omega + arch + garch

    eps[-1] = y[-1] - mu - lmbda * np.sqrt(sigma2[-1])
    volatility = pd.Series(np.sqrt(sigma2), index=returns.index, name=f"GARCH-in-Mean({p},{q}) Volatility")
    mean_component = mu + lmbda * volatility

    output = {
        "mu": mu,
        "lambda": lmbda,
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "volatility": volatility,
        "conditional_mean": pd.Series(mean_component, index=returns.index, name="Conditional Mean")
    }

    if with_confidence:
        stderr = pd.Series(stderr_fraction * volatility, index=volatility.index)
        lower, upper = compute_confidence_bands(volatility, stderr)
        output["stderr"] = stderr
        output["lower"] = lower
        output["upper"] = upper

    return output

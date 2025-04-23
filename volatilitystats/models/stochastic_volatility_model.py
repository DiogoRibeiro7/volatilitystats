import numpy as np
import pandas as pd
from scipy.optimize import minimize
from volatilitystats.utils.confidence import compute_confidence_bands

def sv_log_likelihood(params, returns):
    """
    Approximate log-likelihood for a basic stochastic volatility model.
    """
    mu, phi, sigma_eta = params
    y = returns.fillna(0).values
    n = len(y)

    h = np.zeros(n)
    h[0] = mu / (1 - phi)

    for t in range(1, n):
        h[t] = mu + phi * h[t - 1]

    log_lik = -0.5 * (np.log(2 * np.pi) + h + y**2 / np.exp(h))
    return -np.sum(log_lik)

def estimate_sv_params(
    returns: pd.Series,
    with_confidence: bool = False,
    stderr_fraction: float = 0.1
) -> dict:
    """
    Estimate stochastic volatility parameters via approximate MLE.
    """
    initial_guess = [0.0, 0.95, 0.2]
    bounds = [(-10, 10), (0.01, 0.999), (1e-4, 5.0)]

    result = minimize(
        sv_log_likelihood,
        initial_guess,
        args=(returns,),
        bounds=bounds,
        method="L-BFGS-B"
    )

    mu, phi, sigma_eta = result.x
    y = returns.fillna(0).values
    n = len(y)
    h = np.zeros(n)
    h[0] = mu / (1 - phi)

    for t in range(1, n):
        h[t] = mu + phi * h[t - 1]

    volatility = pd.Series(np.exp(0.5 * h), index=returns.index, name="SV Volatility")

    output = {
        "mu": mu,
        "phi": phi,
        "sigma_eta": sigma_eta,
        "volatility": volatility
    }

    if with_confidence:
        stderr = pd.Series(stderr_fraction * volatility, index=volatility.index)
        lower, upper = compute_confidence_bands(volatility, stderr)
        output["stderr"] = stderr
        output["lower"] = lower
        output["upper"] = upper

    return output

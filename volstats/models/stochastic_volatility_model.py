import numpy as np
import pandas as pd
from scipy.optimize import minimize

def sv_log_likelihood(params, returns):
    """
    Approximate log-likelihood for a basic stochastic volatility model:
    
        r_t = sigma_t * epsilon_t,
        log(sigma_t^2) = mu + phi * log(sigma_{t-1}^2) + eta_t

    where epsilon_t ~ N(0,1) and eta_t ~ N(0, sigma_eta^2)
    
    Parameters
    ----------
    params : list
        Model parameters: [mu, phi, sigma_eta]
    returns : pd.Series
        Log returns time series

    Returns
    -------
    float
        Negative log-likelihood
    """
    mu, phi, sigma_eta = params
    y = returns.fillna(0).values
    n = len(y)

    h = np.zeros(n)
    h[0] = mu / (1 - phi)  # unconditional mean

    for t in range(1, n):
        h[t] = mu + phi * h[t - 1]  # mean process (eta is integrated out)

    log_lik = -0.5 * (np.log(2 * np.pi) + h + y**2 / np.exp(h))
    return -np.sum(log_lik)

def estimate_sv_params(returns: pd.Series) -> dict:
    """
    Estimate stochastic volatility parameters via approximate MLE.

    Parameters
    ----------
    returns : pd.Series
        Log returns time series

    Returns
    -------
    dict
        Estimated parameters and volatility series
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

    return {
        "mu": mu,
        "phi": phi,
        "sigma_eta": sigma_eta,
        "volatility": volatility
    }

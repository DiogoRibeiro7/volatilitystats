import numpy as np
import pandas as pd
from scipy.optimize import minimize

def component_garch_log_likelihood(params, returns):
    """
    Negative log-likelihood for component GARCH(1,1) model.
    
    Parameters
    ----------
    params : list
        Model parameters: [omega, alpha, beta, tau, phi]
    returns : pd.Series
        Log returns.

    Returns
    -------
    float
        Negative log-likelihood value.
    """
    omega, alpha, beta, tau, phi = params
    eps = returns.fillna(0).values
    n = len(eps)

    sigma2 = np.zeros(n)
    q = np.zeros(n)  # long-term (permanent) component

    q[0] = tau / (1 - phi)
    sigma2[0] = q[0]

    for t in range(1, n):
        q[t] = tau + phi * q[t - 1]
        sigma2[t] = omega + alpha * (eps[t - 1] ** 2 - q[t - 1]) + beta * sigma2[t - 1] + q[t - 1]
        if sigma2[t] <= 0:
            return np.inf

    log_lik = -0.5 * (np.log(2 * np.pi) + np.log(sigma2) + eps**2 / sigma2)
    return -np.sum(log_lik)

def estimate_component_garch_params(returns: pd.Series) -> dict:
    """
    Estimate Component GARCH(1,1) parameters via MLE.

    Parameters
    ----------
    returns : pd.Series
        Log returns.

    Returns
    -------
    dict
        Estimated parameters and volatility series.
    """
    initial_guess = [1e-6, 0.05, 0.85, 1e-6, 0.95]
    bounds = [(1e-6, 1), (1e-6, 1), (1e-6, 1), (1e-6, 1), (1e-6, 0.999)]

    result = minimize(
        component_garch_log_likelihood,
        initial_guess,
        args=(returns,),
        bounds=bounds,
        method="L-BFGS-B"
    )

    omega, alpha, beta, tau, phi = result.x

    eps = returns.fillna(0).values
    n = len(eps)
    sigma2 = np.zeros(n)
    q = np.zeros(n)

    q[0] = tau / (1 - phi)
    sigma2[0] = q[0]

    for t in range(1, n):
        q[t] = tau + phi * q[t - 1]
        sigma2[t] = omega + alpha * (eps[t - 1] ** 2 - q[t - 1]) + beta * sigma2[t - 1] + q[t - 1]

    volatility = pd.Series(np.sqrt(sigma2), index=returns.index, name="ComponentGARCH(1,1) Volatility")

    return {
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "tau": tau,
        "phi": phi,
        "volatility": volatility
    }

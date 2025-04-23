import numpy as np
from typing import Sequence

def forecast_garch(
    omega: float,
    alpha: Sequence[float],
    beta: Sequence[float],
    last_eps: Sequence[float],
    last_sigma2: Sequence[float],
    steps: int = 10
) -> np.ndarray:
    """
    Forecast future GARCH(p, q) volatility using recursive updates.

    Parameters
    ----------
    omega : float
        GARCH model omega parameter.
    alpha : Sequence[float]
        ARCH coefficients.
    beta : Sequence[float]
        GARCH coefficients.
    last_eps : Sequence[float]
        Most recent residuals (squared), length q.
    last_sigma2 : Sequence[float]
        Most recent conditional variances, length p.
    steps : int
        Number of steps ahead to forecast.

    Returns
    -------
    np.ndarray
        Forecasted volatility values (standard deviation).
    """
    q = len(alpha)
    p = len(beta)

    sigma2_forecast = []
    for h in range(steps):
        arch = sum(alpha[i] * (last_eps[-(i+1)] if i < len(last_eps) else 0.0) for i in range(q))
        garch = sum(beta[j] * (last_sigma2[-(j+1)] if j < len(last_sigma2) else 0.0) for j in range(p))
        next_sigma2 = omega + arch + garch
        sigma2_forecast.append(next_sigma2)
        # Append the new value to use for next forecast step
        last_eps.append(0.0)  # Future residuals are expected to be 0
        last_sigma2.append(next_sigma2)

    return np.sqrt(np.array(sigma2_forecast))

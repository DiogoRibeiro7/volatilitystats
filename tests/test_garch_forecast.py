import numpy as np
import pytest
from volstats.models.garch_forecast import forecast_garch

def test_forecast_garch_basic():
    omega = 0.1
    alpha = [0.1]
    beta = [0.8]
    last_eps = [1.0]
    last_sigma2 = [1.0]
    steps = 5

    forecast = forecast_garch(omega, alpha, beta, last_eps.copy(), last_sigma2.copy(), steps)
    assert len(forecast) == steps
    assert np.all(forecast > 0)
    assert isinstance(forecast, np.ndarray)

def test_forecast_garch_multiple_lags():
    omega = 0.2
    alpha = [0.05, 0.02]
    beta = [0.85, 0.05]
    last_eps = [1.0, 1.0]
    last_sigma2 = [1.0, 1.0]
    forecast = forecast_garch(omega, alpha, beta, last_eps.copy(), last_sigma2.copy(), steps=10)
    assert forecast.shape == (10,)
    assert np.all(forecast > 0)

def test_forecast_garch_stability_increasing():
    omega = 0.05
    alpha = [0.0]
    beta = [0.95]
    last_eps = [0.0]
    last_sigma2 = [0.5]
    forecast = forecast_garch(omega, alpha, beta, last_eps.copy(), last_sigma2.copy(), steps=10)
    assert np.all(np.diff(forecast) >= -1e-5)  # allow for small numerical variation

def test_forecast_garch_zero_arch():
    forecast = forecast_garch(omega=0.1, alpha=[], beta=[0.9], last_eps=[], last_sigma2=[1.0], steps=3)
    assert len(forecast) == 3
    assert np.all(forecast > 0)

def test_forecast_garch_zero_garch():
    forecast = forecast_garch(omega=0.1, alpha=[0.5], beta=[], last_eps=[1.0], last_sigma2=[], steps=3)
    assert len(forecast) == 3
    assert np.all(forecast > 0)

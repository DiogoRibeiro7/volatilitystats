import numpy as np
import pandas as pd
import pytest
from volatilitystats.models.garch_in_mean_model import estimate_garch_in_mean_params, garch_in_mean_log_likelihood

@pytest.fixture
def returns():
    return pd.Series(np.random.normal(0, 1, 300))

def test_garch_in_mean_11_estimation(returns):
    result = estimate_garch_in_mean_params(returns, p=1, q=1)
    assert "volatility" in result
    assert len(result["volatility"]) == len(returns)
    assert np.all(result["volatility"] > 0)
    assert np.isfinite(result["conditional_mean"]).all()

def test_garch_in_mean_21_estimation(returns):
    result = estimate_garch_in_mean_params(returns, p=2, q=1)
    assert "volatility" in result
    assert np.all(result["volatility"] > 0)
    assert len(result["alpha"]) == 1
    assert len(result["beta"]) == 2

def test_garch_in_mean_log_likelihood_nan_protection(returns):
    returns_nan = pd.Series([np.nan] * 300)
    params = [0.0, 0.0, 1e-6, 0.05, 0.9]
    result = garch_in_mean_log_likelihood(params, returns_nan, p=1, q=1)
    assert np.isinf(result) or np.isfinite(result)

def test_garch_in_mean_log_likelihood_empty():
    returns = pd.Series([], dtype=float)
    params = [0.0, 0.0, 1e-6, 0.05, 0.9]
    result = garch_in_mean_log_likelihood(params, returns, p=1, q=1)
    assert result == 0.0 or np.isinf(result)

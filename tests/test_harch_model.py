import numpy as np
import pandas as pd
import pytest
from volatilitystats.models.harch_model import estimate_harch_params, harch_log_likelihood

@pytest.fixture
def returns():
    return pd.Series(np.random.normal(0, 1, 300))

def test_harch_basic_estimation(returns):
    lags = [1, 5, 22]
    result = estimate_harch_params(returns, lags)
    assert "volatility" in result
    assert np.all(result["volatility"] > 0)
    assert len(result["alpha"]) == len(lags)

def test_harch_log_likelihood_nan_returns():
    returns_nan = pd.Series([np.nan] * 300)
    params = [1e-6, 0.05, 0.05, 0.05]
    result = harch_log_likelihood(params, returns_nan, lags=[1, 5, 22])
    assert np.isinf(result) or np.isfinite(result)

def test_harch_log_likelihood_empty():
    returns = pd.Series([], dtype=float)
    params = [1e-6, 0.05, 0.05, 0.05]
    result = harch_log_likelihood(params, returns, lags=[1, 5, 22])
    assert result == 0.0 or np.isinf(result)

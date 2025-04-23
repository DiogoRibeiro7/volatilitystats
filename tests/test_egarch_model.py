import numpy as np
import pandas as pd
import pytest
from volatilitystats.models.egarch_model import estimate_egarch_params, egarch_log_likelihood

def test_estimate_egarch_params_valid():
    returns = pd.Series(np.random.normal(0, 1, 200))
    result = estimate_egarch_params(returns, p=1, q=1)
    assert "volatility" in result
    assert len(result["volatility"]) == len(returns)
    assert np.all(np.isfinite(result["volatility"]))

def test_egarch_log_likelihood_nan_safe():
    returns = pd.Series([np.nan] * 100)
    params = [0.0, 0.1, 0.0, 0.8]  # EGARCH(1,1)
    result = egarch_log_likelihood(params, returns, p=1, q=1)
    assert np.isinf(result), "Should return inf when returns are all NaN"

def test_egarch_log_likelihood_empty():
    returns = pd.Series([], dtype=float)
    params = [0.0, 0.1, 0.0, 0.8]
    result = egarch_log_likelihood(params, returns, p=1, q=1)
    assert result == 0.0

def test_egarch_log_likelihood_invalid_sigma_handling():
    returns = pd.Series([10.0] * 100)
    params = [10.0, -10.0, 0.0, 0.0]  # clearly unstable
    result = egarch_log_likelihood(params, returns, p=1, q=1)
    assert np.isinf(result), "Function should return inf when instability occurs"

def test_estimate_egarch_params_shapes():
    returns = pd.Series(np.random.normal(0, 1, 150))
    result = estimate_egarch_params(returns, p=2, q=2)
    assert len(result["alpha"]) == 2
    assert len(result["gamma"]) == 2
    assert len(result["beta"]) == 2

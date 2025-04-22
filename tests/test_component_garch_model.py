import pytest
import pandas as pd
import numpy as np
from volstats.models.component_garch_model import component_garch_log_likelihood

def test_component_garch_log_likelihood_valid_input():
    # Test with valid parameters and returns
    params = [1e-6, 0.05, 0.85, 1e-6, 0.95]
    returns = pd.Series(np.random.normal(0, 1, 100))
    result = component_garch_log_likelihood(params, returns)
    assert np.isfinite(result), "Log-likelihood should be finite for valid inputs"

def test_component_garch_log_likelihood_negative_sigma2():
    # Test with parameters that could lead to negative sigma2
    params = [1e-6, 1.0, 0.0, 0.0, 0.0]  # alpha too high
    returns = pd.Series([100.0] + [0.0] * 99)
    result = component_garch_log_likelihood(params, returns)
    assert np.isinf(result), "Log-likelihood should return infinity for invalid sigma2"


def test_component_garch_log_likelihood_empty_returns():
    # Test with empty returns
    params = [1e-6, 0.05, 0.85, 1e-6, 0.95]
    returns = pd.Series([])
    result = component_garch_log_likelihood(params, returns)
    assert result == 0, "Log-likelihood should be 0 for empty returns"

def test_component_garch_log_likelihood_invalid_params_length():
    # Test with invalid parameter length
    params = [1e-6, 0.05, 0.85]  # Missing tau and phi
    returns = pd.Series(np.random.normal(0, 1, 100))
    with pytest.raises(ValueError):
        component_garch_log_likelihood(params, returns)

def test_component_garch_log_likelihood_nan_returns():
    # Test with NaN values in returns
    params = [1e-6, 0.05, 0.85, 1e-6, 0.95]
    returns = pd.Series([np.nan] * 100)
    result = component_garch_log_likelihood(params, returns)
    assert np.isfinite(result), "Log-likelihood should handle NaN values in returns"
    
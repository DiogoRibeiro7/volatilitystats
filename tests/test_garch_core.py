import numpy as np
import pandas as pd
import pytest
from volatilitystats.models.garch_core import garch

@pytest.fixture
def returns():
    return pd.Series(np.random.normal(0, 1, 200))

def test_garch_11(returns):
    vol = garch(returns, omega=0.1, alpha=[0.05], beta=[0.9])
    assert len(vol) == len(returns)
    assert vol.notna().all()
    assert (vol > 0).all()

def test_garch_21(returns):
    vol = garch(returns, omega=0.1, alpha=[0.05, 0.02], beta=[0.9])
    assert len(vol) == len(returns)
    assert vol.notna().all()

def test_garch_12(returns):
    vol = garch(returns, omega=0.1, alpha=[0.05], beta=[0.7, 0.1])
    assert len(vol) == len(returns)
    assert vol.notna().all()

def test_garch_10(returns):
    vol = garch(returns, omega=0.1, alpha=[0.05], beta=[])
    assert len(vol) == len(returns)
    assert vol.notna().all()

def test_garch_01(returns):
    vol = garch(returns, omega=0.1, alpha=[], beta=[0.9])
    assert len(vol) == len(returns)
    assert vol.notna().all()

def test_garch_output_behavior_changes(returns):
    base = garch(returns, omega=0.1, alpha=[0.1], beta=[0.8])
    altered = garch(returns, omega=0.5, alpha=[0.2], beta=[0.7])
    assert not np.allclose(base, altered), "Changing params should change output"

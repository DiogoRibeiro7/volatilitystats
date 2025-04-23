import pandas as pd
from typing import Literal
from volatilitystats.utils.confidence import compute_confidence_bands as compute_confidence_bands

def plot_forecast_with_confidence_band(forecast: pd.Series, lower: pd.Series | None = None, upper: pd.Series | None = None, realized: pd.Series | None = None, stderr: pd.Series | None = None, ci_method: Literal['normal', 'bootstrap'] = 'normal', z: float = 1.96, title: str = 'Forecast with Confidence Interval', ylabel: str = 'Volatility', figsize: tuple = (12, 5), savepath: str | None = None, compute_if_missing: bool = True): ...

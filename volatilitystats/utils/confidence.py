import pandas as pd
import numpy as np
from typing import Tuple, Literal

def compute_confidence_bands(
    forecast: pd.Series,
    stderr: pd.Series,
    method: Literal["normal", "bootstrap"] = "normal",
    z: float = 1.96
) -> Tuple[pd.Series, pd.Series]:
    """
    Compute confidence bands around forecasted values.

    Parameters
    ----------
    forecast : pd.Series
        Forecast series.
    stderr : pd.Series
        Standard error or bootstrapped deviation estimate.
    method : {"normal", "bootstrap"}
        Method for computing intervals.
    z : float
        Critical value for normal-based intervals.

    Returns
    -------
    Tuple[pd.Series, pd.Series]
        Lower and upper confidence bounds.
    """
    if method == "normal":
        upper = forecast + z * stderr
        lower = forecast - z * stderr
    elif method == "bootstrap":
        q = stderr.quantile(1 - (1 - 0.95) / 2)
        upper = forecast + q
        lower = forecast - q
    else:
        raise ValueError("Unsupported method for confidence band.")

    return lower.clip(lower=0), upper

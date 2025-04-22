import pandas as pd
from typing import Sequence
from volstats.models import garch as garch
from volstats.utils.confidence import compute_confidence_bands as compute_confidence_bands

def garch_log_likelihood(params: Sequence[float], returns: pd.Series, p: int, q: int) -> float: ...
def estimate_garch_params(returns: pd.Series, p: int = 1, q: int = 1, bounds: Sequence[tuple] | None = None, with_confidence: bool = False, stderr_fraction: float = 0.1) -> dict: ...

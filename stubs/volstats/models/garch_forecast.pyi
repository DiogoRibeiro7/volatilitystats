import numpy as np
from typing import Sequence

def forecast_garch(omega: float, alpha: Sequence[float], beta: Sequence[float], last_eps: Sequence[float], last_sigma2: Sequence[float], steps: int = 10) -> np.ndarray: ...

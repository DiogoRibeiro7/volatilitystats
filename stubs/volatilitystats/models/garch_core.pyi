import pandas as pd
from typing import Sequence

def garch(returns: pd.Series, omega: float, alpha: Sequence[float], beta: Sequence[float], initial_vol: float | None = None) -> pd.Series: ...

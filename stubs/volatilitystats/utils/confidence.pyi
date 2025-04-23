import pandas as pd
from typing import Literal

def compute_confidence_bands(forecast: pd.Series, stderr: pd.Series, method: Literal['normal', 'bootstrap'] = 'normal', z: float = 1.96) -> tuple[pd.Series, pd.Series]: ...

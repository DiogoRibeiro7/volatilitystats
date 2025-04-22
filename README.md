# volstats

**Volatility models and estimators in pure Python**

`volstats` is a Python package for computing and comparing volatility estimators, including both classical methods (e.g., Parkinson, Yang-Zhang) and model-based approaches (e.g., GARCH). It is designed to be lightweight, dependency-free (just NumPy, SciPy, Pandas), and easy to extend.

---

## 📦 Features

- ✅ Standard close-to-close volatility (log returns)
- ✅ High–low based volatility (Parkinson)
- ✅ Open–high–low–close estimator (Yang-Zhang)
- ✅ GARCH(p, q) model-based volatility
- ✅ MLE estimation of GARCH(1,1) parameters
- ✅ Forecast future volatility with recursive GARCH predictions

---

## 📁 Package Structure

```plaintext
volstats/
│
├── __init__.py
│
├── estimators/
│   ├── __init__.py
│   ├── standard.py       # Close-to-close volatility
│   ├── parkinson.py
│   └── yangzhang.py
│
├── models/
│   ├── __init__.py
│   ├── garch_core.py     # GARCH(p,q)
│   ├── garch_mle.py      # MLE for GARCH(1,1)
│   └── garch_forecast.py # Forecasting
│
├── utils/
│   └── likelihood.py     # Common likelihood helpers (optional for reuse)
```


---

## 🚀 Quickstart

```python
import yfinance as yf
import numpy as np
from volstats.models.garch_mle import estimate_garch_params
from volstats.estimators.yangzhang import yang_zhang_volatility

df = yf.download("AAPL", start="2022-01-01", end="2023-01-01")
log_returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()

# Estimate GARCH(1,1) parameters and volatility
garch_result = estimate_garch_params(log_returns)
garch_vol = garch_result["volatility"]

# Compare with Yang-Zhang
yz_vol = yang_zhang_volatility(df)

# Plot or analyze
```

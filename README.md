# volstats

**Volatility models and estimators in pure Python**

`volstats` is a Python package for computing and comparing volatility estimators, including both classical methods (e.g., Parkinson, Yang-Zhang) and model-based approaches (e.g., GARCH, EGARCH, HARCH). It is designed to be lightweight, dependency-free (just NumPy, SciPy, Pandas), and easy to extend.

---

## 📦 Features

### 🧮 Statistical Estimators
- ✅ Standard close-to-close volatility (log returns)
- ✅ Parkinson estimator (high–low range)
- ✅ Yang-Zhang estimator (OHLC)
- ✅ Rogers-Satchell and Garman-Klass
- ✅ EWMA (Exponentially Weighted Moving Average)
- ✅ Realized Volatility: Two-Scale, Median, Bipower

### 📈 Volatility Models
- ✅ GARCH(p, q), EGARCH(p, q), GJR-GARCH(p, q)
- ✅ GARCH-in-Mean (GARCH-M)
- ✅ Component GARCH
- ✅ HARCH
- ✅ Stochastic Volatility (simulation-based)

### ⚙️ Utilities and Tooling
- ✅ Forecasting with GARCH
- ✅ MLE parameter estimation for all models
- ✅ Realized volatility estimators with resampled time grouping
- ✅ Clean modular structure (estimators/models/tests/docs)
- ✅ Full support for `poetry`, `pytest`, and `invoke` tasks

---

## 📁 Package Structure

```plaintext
volstats/
│
├── estimators/
│   ├── standard.py            # Close-to-close volatility
│   ├── parkinson.py
│   ├── yangzhang.py
│   ├── rogers_satchell.py
│   ├── garman_klass.py
│   ├── ewma_volatility.py
│   ├── two_scale_realized_volatility.py
│   ├── median_realized_volatility.py
│   └── bipower_variation.py
│
├── models/
│   ├── garch_core.py          # GARCH(p, q) volatility
│   ├── garch_forecast.py      # GARCH forecasting
│   ├── garch_mle.py           # GARCH parameter estimation
│   ├── egarch_model.py
│   ├── gjr_garch_model.py
│   ├── garch_in_mean_model.py
│   ├── component_garch_model.py
│   ├── harch_model.py
│   └── stochastic_volatility_model.py
│
├── tests/                    # Unit tests (pytest)
├── docs/                     # Sphinx-based documentation
└── roadmap.md                # Project milestones and goals
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


## 📚 Documentation
Full usage examples and API reference coming soon at [📘 Read the Docs](https://volstats.readthedocs.io) (planned).

## 🤝 Contributing
Open an issue, suggest improvements, or help us add new estimators and models.

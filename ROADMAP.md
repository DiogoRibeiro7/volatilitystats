# volstats: Roadmap

## Overview
`volstats` is a Python library for estimating and modeling financial market volatility. It combines robust statistical estimators with parametric models to support both academic research and real-time volatility analysis.

This roadmap outlines planned features, models, improvements, and organizational steps to evolve `volstats` into a comprehensive volatility analysis package.

---

## ✅ Completed

### Estimators

- Standard Close-to-Close Volatility
- Parkinson Volatility
- Yang-Zhang Volatility
- Rogers-Satchell Volatility
- Garman-Klass Volatility
- EWMA Volatility
- Two-Scale Realized Volatility
- Median Realized Volatility
- Bipower Variation

### Models

- GARCH (p, q)
- GARCH-in-Mean (p, q)
- EGARCH (p, q)
- GJR-GARCH (p, q)
- Component GARCH
- HARCH
- Stochastic Volatility

### Tooling

- Modularized under `estimators/` and `models/`
- `pyproject.toml` with Poetry
- Invoke tasks for build, publish, testing, docs, and git
- `tests/` with full pytest coverage
- Realized volatility estimators support time grouping via pandas

---

## 🚧 In Progress

- 📘 Sphinx-based documentation with usage examples
- 🔍 Interactive plots for volatility vs returns (e.g., using Plotly)
- 🧪 Benchmarking volatility estimators on real datasets (e.g., TAQ, Yahoo, Crypto)

---

## 🧠 Planned

### Estimators

- Realized Kernel Estimator
- Jump Detection (based on difference RV vs BV)
- Minimum Realized Volatility
- Realized Semivariance (Upside / Downside)

### Models

- Heston Model (stochastic volatility with closed-form option pricing)
- GARCH-MIDAS
- Multivariate GARCH (DCC-GARCH, BEKK)

### Utilities

- CSV → volatility series loader
- JSON schema for exporting model parameters
- CLI tool for estimator selection and visualization
- REST API endpoint for real-time volatility analysis (FastAPI-based)

---

## 💡 Suggestions Welcome

Please open issues or discussions for anything you’d like to see prioritized!

---

## 📅 Release Milestones

- **v0.1.0**: Core estimators + univariate models ✅
- **v0.2.0**: Realized measures, component models, documentation 🚧
- **v0.3.0**: Interactive CLI, plotting, REST API, backtesting tools 🧠

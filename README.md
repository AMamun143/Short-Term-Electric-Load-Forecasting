# Short-Term Electric Load Forecasting

Author: Abdullah Mamun

This project implements a hybrid soft-computing workflow for short-term electric load forecasting (STLF).  
It compares multiple baselines and a neuro-fuzzy hybrid with PSO-style optimization.

## What is included

- Reproducible pipeline (`bash run.sh`)
- Public data download script (`scripts/download_data.py`)
- Baselines:
  - Persistence
  - Linear Regression
  - MLP
- Hybrid models:
  - ANFIS-like neuro-fuzzy model
  - ANFIS-PSO model
- Evaluation metrics:
  - MAE, RMSE, MAPE
  - Interval coverage (PICP) and interval width (MPIW)
- Final artifacts:
  - `results/metrics.csv`
  - `results/predictions.csv`
  - `figs/forecast_comparison.png`
  - `figs/anfis_pso_interval.png`
  - `paper/project_paper.md`

## Reproduce

```bash
bash run.sh
```

## Project paper

See `paper/project_paper.md` for motivation, literature discussion, novelty, utility, data rationale, and citations.

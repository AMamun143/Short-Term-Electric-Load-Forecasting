# Project Paper: Short-Term Electric Load Forecasting

Course: Spring 2026 Applied Soft Computing  
Author: Abdullah Mamun

## 1) Motivation

Short-term load forecasting (hour-ahead to day-ahead) is important for daily power grid operation.  
If load is under-forecasted, operators can face reliability risk. If load is over-forecasted, utilities may schedule too much reserve and increase cost.  
Because modern demand patterns are nonlinear and affected by seasonality, simple linear methods often miss local patterns.

This project studies a hybrid soft-computing setup that combines:

- interpretable fuzzy rule structure (ANFIS-like),
- data-driven fitting from historical load,
- and PSO-style global search for premise tuning.

## 2) Related Work and Baselines

The instructor feedback highlighted that novelty must be evaluated against realistic baselines, not only exact-method matches.  
Based on that, this project includes **five** models:

1. **Persistence baseline** (forecast = previous hour),
2. **Linear Regression baseline**,
3. **MLP baseline**,
4. **ANFIS-like neuro-fuzzy model** (gradient-based premise fitting),
5. **ANFIS-PSO hybrid** (PSO for premise optimization + least-squares consequents).

This baseline set is important because:

- Persistence is often strong in short horizons,
- linear models are transparent and fast,
- MLP captures nonlinear structure,
- ANFIS and ANFIS-PSO test whether hybrid soft-computing gives practical gains.

## 3) Novelty and Utility

### Novelty

The novelty is not only “using ANFIS,” but **how** it is evaluated:

- a direct comparison to multiple baselines on the same split,
- explicit uncertainty intervals (residual-quantile intervals),
- and reproducible one-command execution.

### Utility

- Better short-term predictions reduce operational cost and reserve mismatch.
- Neuro-fuzzy structure keeps some interpretability.
- Interval forecasts give uncertainty awareness, useful for decision support.

## 4) Data Source and Appropriateness

Dataset: AEP hourly load series (publicly mirrored from the PJM/energy benchmark ecosystem).  
Download is scripted in `scripts/download_data.py`.

Why appropriate:

- Hourly resolution matches STLF use case,
- long temporal span supports seasonal and weekly patterns,
- no manual data collection step is needed for reproducibility.

Features include:

- lag terms (`1, 2, 3, 24, 48, 168`),
- rolling means (`24h`, `168h`),
- calendar/cycle features (`hour`, `day-of-week`, `month`, sine/cosine hour).

Splits are chronological (70/15/15) to avoid leakage.

## 5) Methodology

### Point forecasting metrics

- MAE
- RMSE
- MAPE

### Interval metrics

- PICP (prediction interval coverage probability)
- MPIW (mean prediction interval width)

Intervals are built from validation residual quantiles (5% and 95%) and then applied on test forecasts.

## 6) Results Summary

The script `scripts/run_experiment.py` writes full metrics to `results/metrics.csv`.  
Two visual outputs are generated:

- `figs/forecast_comparison.png`
- `figs/anfis_pso_interval.png`

In this implementation, the ANFIS-PSO model provides competitive forecast quality and stable uncertainty behavior relative to the baselines.

## 7) Discussion

### Strengths

- Multiple baselines are included (as requested in feedback).
- Pipeline is fully reproducible from scratch.
- Both point and interval performance are reported.

### Limitations

- Only one load region is currently used.
- Weather variables are not yet integrated in this version.
- ANFIS implementation is compact and can be extended with richer rule structures.

### Future work

- Add weather and holiday effects,
- multi-region transfer experiments,
- compare with transformer-based time-series models for stronger deep baseline.

## 8) Reproducibility Checklist

- Code: yes (`src/`, `scripts/`)
- Data retrieval script: yes (`scripts/download_data.py`)
- Single command run: yes (`bash run.sh`)
- Output artifacts: yes (`results/`, `figs/`)

## 9) Citations and Acknowledgements

1. J.-S. R. Jang, “ANFIS: Adaptive-network-based fuzzy inference system,” IEEE TSMC, 1993.  
2. T. Hong et al., “Probabilistic energy forecasting: GEFCom2012 and beyond,” IJF, 2016.  
3. W. Kong et al., “Short-term residential load forecasting based on LSTM,” IEEE TSG, 2019.  
4. M. Raza and A. Khosravi, “AI-based load forecasting review,” Renewable & Sustainable Energy Reviews, 2015.  
5. Public hourly load benchmark mirrors based on PJM/energy forecasting datasets.  

Acknowledgement: This work uses open data and open-source Python tools for non-commercial academic purposes.

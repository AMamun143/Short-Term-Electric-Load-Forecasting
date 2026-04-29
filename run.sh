#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python scripts/download_data.py
python scripts/run_experiment.py
python scripts/make_presentation.py

echo "Done."
echo "Results: results/metrics.csv"
echo "Figures: figs/forecast_comparison.png and figs/anfis_pso_interval.png"
echo "Slides: /home/abdullah/Downloads/Short_Term_Load_Forecasting_Presentation.pptx"

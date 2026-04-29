from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.maximum(np.abs(y_true), 1e-6)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def interval_metrics(y_true: np.ndarray, y_low: np.ndarray, y_high: np.ndarray) -> tuple[float, float]:
    coverage = float(np.mean((y_true >= y_low) & (y_true <= y_high)) * 100.0)
    width = float(np.mean(y_high - y_low))
    return coverage, width

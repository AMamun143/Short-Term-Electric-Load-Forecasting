from __future__ import annotations

import pandas as pd


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.sort_values("Datetime").reset_index(drop=True)
    df = df.rename(columns={"AEP_MW": "load_mw"})
    return df


def make_features(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    out = df.copy()
    out["hour"] = out["Datetime"].dt.hour
    out["dow"] = out["Datetime"].dt.dayofweek
    out["month"] = out["Datetime"].dt.month
    out["is_weekend"] = (out["dow"] >= 5).astype(int)

    out["sin_hour"] = pd.Series(
        (out["hour"] * 2 * 3.141592653589793 / 24.0).apply(__import__("math").sin)
    )
    out["cos_hour"] = pd.Series(
        (out["hour"] * 2 * 3.141592653589793 / 24.0).apply(__import__("math").cos)
    )

    for lag in [1, 2, 3, 24, 48, 168]:
        out[f"lag_{lag}"] = out["load_mw"].shift(lag)

    out["rolling_24"] = out["load_mw"].shift(1).rolling(24).mean()
    out["rolling_168"] = out["load_mw"].shift(1).rolling(168).mean()

    out["target"] = out["load_mw"].shift(-horizon)
    out = out.dropna().reset_index(drop=True)
    return out


def time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    return train, val, test

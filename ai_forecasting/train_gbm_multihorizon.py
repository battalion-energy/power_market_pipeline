#!/usr/bin/env python3
"""
Train a multi-horizon Gradient Boosting baseline for ERCOT RT prices.

The goal is to produce higher-fidelity 48h forecasts that capture ramps/spikes
for demo purposes while staying lightweight enough to retrain quickly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import joblib

import polars as pl

DATA_FILE = (
    "/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/"
    "master_enhanced_with_net_load_reserves_2019_2025.parquet"
)

OUTPUT_DIR = Path("models")
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "gbm_multihorizon.joblib"
FEATURES_PATH = OUTPUT_DIR / "gbm_feature_columns.json"
METADATA_PATH = OUTPUT_DIR / "gbm_multihorizon_metadata.json"

FORECAST_HORIZON = 48
MAX_LAG = 168

TRAIN_START = pd.Timestamp("2022-01-01")
TRAIN_END = pd.Timestamp("2023-06-30")
VAL_END = pd.Timestamp("2024-06-30")

GB_PARAMS = dict(
    learning_rate=0.08,
    max_depth=8,
    max_leaf_nodes=128,
    min_samples_leaf=64,
    l2_regularization=0.2,
    max_iter=400,
    random_state=42,
)

# -----------------------------------------------------------------------------
# Feature engineering helpers
# -----------------------------------------------------------------------------

BASE_FEATURES = [
    "price_mean",
    "price_da",
    "load_forecast_mean",
    "load_forecast_trend_24h",
    "load_forecast_spread_pct",
    "ordc_online_reserves_min",
    "ordc_scarcity_indicator_max",
    "ordc_critical_indicator_max",
    "REGUP",
    "REGDN",
    "RRS",
    "NSPIN",
    "temp_avg",
    "temp_max_hourly",
    "temp_min_hourly",
    "cloud_cover_pct",
]

PRICE_LAGS = [1, 2, 3, 6, 12, 18, 24, 30, 36, 48, 72, 96, 120, 144, 168]
DA_LAGS = [1, 2, 3, 6, 12, 18, 24, 36, 48]
LOAD_LAGS = [1, 6, 12, 24]
SPREAD_LAGS = [1, 24]


def load_dataframe(path: str) -> pd.DataFrame:
    columns = list({*BASE_FEATURES, "timestamp"})
    lf = pl.read_parquet(path, columns=columns)
    agg = lf.group_by("timestamp").mean().sort("timestamp")
    df = agg.to_pandas()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.drop_duplicates("timestamp").set_index("timestamp").sort_index()

    # Replace impossible values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Interpolate exogenous with small gaps
    for col in BASE_FEATURES:
        if col not in df.columns:
            continue
        series = df[col]
        df[col] = (
            series.interpolate(method="time", limit_direction="both")
            .ffill()
            .bfill()
        )

    df["da_rt_spread"] = df["price_da"] - df["price_mean"]
    return df


def add_time_features(df: pd.DataFrame) -> None:
    idx = df.index
    df["hour"] = idx.hour
    df["dow"] = idx.dayofweek
    df["month"] = idx.month
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)


def add_lag_features(df: pd.DataFrame) -> List[str]:
    feature_cols: List[str] = []

    for lag in PRICE_LAGS:
        col = f"price_mean_lag_{lag}"
        df[col] = df["price_mean"].shift(lag)
        feature_cols.append(col)

    for lag in DA_LAGS:
        col = f"price_da_lag_{lag}"
        df[col] = df["price_da"].shift(lag)
        feature_cols.append(col)

    for lag in LOAD_LAGS:
        col = f"load_forecast_mean_lag_{lag}"
        df[col] = df["load_forecast_mean"].shift(lag)
        feature_cols.append(col)

    for lag in SPREAD_LAGS:
        col = f"spread_lag_{lag}"
        df[col] = df["da_rt_spread"].shift(lag)
        feature_cols.append(col)

    df["price_mean_rolling_24"] = df["price_mean"].rolling(24).mean()
    df["price_mean_rolling_168"] = df["price_mean"].rolling(168).mean()
    feature_cols.extend(["price_mean_rolling_24", "price_mean_rolling_168"])

    return feature_cols


def add_future_targets(df: pd.DataFrame, horizon: int) -> List[str]:
    target_cols = []
    for h in range(1, horizon + 1):
        col = f"target_rt_h{h}"
        df[col] = df["price_mean"].shift(-h)
        target_cols.append(col)
    return target_cols


def build_dataset() -> tuple[pd.DataFrame, List[str], List[str]]:
    df = load_dataframe(DATA_FILE)
    if TRAIN_START is not None:
        df = df[df.index >= TRAIN_START]
    add_time_features(df)
    lag_cols = add_lag_features(df)
    target_cols = add_future_targets(df, FORECAST_HORIZON)

    feature_cols = (
        BASE_FEATURES
        + ["da_rt_spread", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]
        + lag_cols
    )

    df = df.dropna(subset=feature_cols + target_cols)
    return df, feature_cols, target_cols


def make_supervised(
    df: pd.DataFrame, feature_cols: List[str], target_cols: List[str]
) -> tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]:
    max_shift = max(PRICE_LAGS + DA_LAGS + LOAD_LAGS + SPREAD_LAGS)
    records = []
    targets = []
    timestamps = []
    for end_idx in range(max_shift, len(df)):
        ts = df.index[end_idx]
        row = df.iloc[end_idx][feature_cols]
        target = df.iloc[end_idx][target_cols].values
        if np.isnan(row).any() or np.isnan(target).any():
            continue
        records.append(row.to_numpy(dtype=np.float32))
        targets.append(target.flatten())
        timestamps.append(ts)
    X = np.vstack(records)
    y = np.vstack(targets)
    return X, y, timestamps


@dataclass
class Split:
    name: str
    X: np.ndarray
    y: np.ndarray
    timestamps: List[pd.Timestamp]


def split_by_time(
    X: np.ndarray,
    y: np.ndarray,
    timestamps: List[pd.Timestamp],
    train_end: pd.Timestamp,
    val_end: pd.Timestamp,
) -> Dict[str, Split]:
    ts = np.array(timestamps)
    train_idx = ts <= train_end
    val_idx = (ts > train_end) & (ts <= val_end)
    test_idx = ts > val_end

    splits = {
        "train": Split("train", X[train_idx], y[train_idx], ts[train_idx].tolist()),
        "val": Split("val", X[val_idx], y[val_idx], ts[val_idx].tolist()),
        "test": Split("test", X[test_idx], y[test_idx], ts[test_idx].tolist()),
    }
    for name, split in splits.items():
        print(f"{name.title()} rows: {len(split.X):,}")
    return splits


def build_pipeline() -> Pipeline:
    scaler = StandardScaler()
    gb = HistGradientBoostingRegressor(**GB_PARAMS)
    multi = MultiOutputRegressor(gb, n_jobs=-1)
    pipe = Pipeline([
        ("scaler", scaler),
        ("gbm", multi),
    ])
    return pipe


def evaluate(model: Pipeline, split: Split) -> Dict[str, float]:
    preds = model.predict(split.X)
    mae_per_h = np.mean(np.abs(preds - split.y), axis=0)
    metrics = {
        "mae_mean": float(mae_per_h.mean()),
        "mae_1h": float(mae_per_h[0]),
        "mae_24h": float(mae_per_h[23]),
        "mae_48h": float(mae_per_h[-1]),
        "mae_per_horizon": mae_per_h.tolist(),
    }
    return metrics


def main() -> None:
    df, feature_cols, target_cols = build_dataset()
    print(f"Dataset rows after cleaning: {len(df):,}")

    X, y, timestamps = make_supervised(df, feature_cols, target_cols)
    print(f"Supervised samples: {len(X):,} with {X.shape[1]} features")

    splits = split_by_time(X, y, timestamps, TRAIN_END, VAL_END)

    model = build_pipeline()
    model.fit(splits["train"].X, splits["train"].y)
    print("✓ GBM training complete")

    metrics = {
        name: evaluate(model, split) for name, split in splits.items()
    }

    joblib.dump(model, MODEL_PATH)
    FEATURES_PATH.write_text(json.dumps(feature_cols, indent=2))

    metadata = {
        "created": datetime.utcnow().isoformat(),
        "data_file": DATA_FILE,
        "forecast_horizon": FORECAST_HORIZON,
        "train_rows": len(splits["train"].X),
        "val_rows": len(splits["val"].X),
        "test_rows": len(splits["test"].X),
        "feature_count": len(feature_cols),
        "gb_params": GB_PARAMS,
        "metrics": metrics,
        "train_end": TRAIN_END.isoformat(),
        "val_end": VAL_END.isoformat(),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))

    print(f"Artifacts:")
    print(f"  • Model:    {MODEL_PATH}")
    print(f"  • Features: {FEATURES_PATH}")
    print(f"  • Metadata: {METADATA_PATH}")
    for name, stats in metrics.items():
        print(
            f"{name.upper()} mean MAE {stats['mae_mean']:.2f} "
            f"(1h {stats['mae_1h']:.2f} / 24h {stats['mae_24h']:.2f} / 48h {stats['mae_48h']:.2f})"
        )


if __name__ == "__main__":
    main()

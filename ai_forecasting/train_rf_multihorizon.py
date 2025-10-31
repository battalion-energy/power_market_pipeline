#!/usr/bin/env python3
"""
Train a multi-horizon Random Forest forecaster for ERCOT prices.

This script learns a 48-hour ahead forecast of system-wide real-time prices using
lagged prices, load forecasts, and ORDC/reserve indicators. It produces:
  • models/random_forest_multihorizon.joblib          (trained regressor)
  • models/random_forest_feature_columns.txt          (ordered feature list)
  • models/random_forest_metadata.json                (metrics + config snapshot)

The resulting model is fast to run and provides per-hour MAE statistics that can be
used to pick the best demo vintages or to build a backup forecaster for the dashboard.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import polars as pl
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DATA_FILE = (
    "/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/"
    "master_enhanced_with_net_load_reserves_2019_2025.parquet"
)

OUTPUT_DIR = Path("models")
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "random_forest_multihorizon.joblib"
FEATURES_PATH = OUTPUT_DIR / "random_forest_feature_columns.txt"
METADATA_PATH = OUTPUT_DIR / "random_forest_metadata.json"

FORECAST_HORIZON = 48

TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15  # remainder is test

RF_PARAMS = dict(
    n_estimators=320,
    max_depth=18,
    min_samples_split=6,
    min_samples_leaf=3,
    max_features="sqrt",
    n_jobs=-1,
    random_state=42,
)

BASE_FEATURES = [
    "price_da",
    "price_mean",
    "load_forecast_mean",
    "load_forecast_trend_24h",
    "load_forecast_spread_pct",
    "ordc_online_reserves_min",
    "ordc_scarcity_indicator_max",
    "net_load_MW",
    "net_load_ramp_1h",
    "net_load_ramp_3h",
    "wind_generation_MW",
    "solar_generation_MW",
    "reserve_margin_pct",
    "renewable_penetration_pct",
]

IMPORTANT_EXOGENOUS = [
    col
    for col in BASE_FEATURES
    if col not in {"price_da", "price_mean"}  # lagged separately
]

PRICE_LAGS = [1, 2, 3, 6, 12, 24, 48, 72, 168]
LOAD_LAGS = [1, 6, 24]


@dataclass
class SplitResult:
    name: str
    X: np.ndarray
    y: np.ndarray
    timestamps: Sequence[pd.Timestamp]


# -----------------------------------------------------------------------------
# Data loading / feature engineering
# -----------------------------------------------------------------------------

def load_hourly_dataframe(path: str) -> pd.DataFrame:
    print(f"Loading data from {path} ...")
    columns = list({*BASE_FEATURES, "timestamp"})
    pl_df = pl.read_parquet(path, columns=columns)
    hourly = pl_df.group_by("timestamp").mean().sort("timestamp")
    df = hourly.to_pandas()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.drop_duplicates("timestamp").set_index("timestamp").sort_index()
    print(f"  Loaded {len(df):,} rows spanning {df.index.min()} → {df.index.max()}")

    # Ensure float dtype
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def fill_exogenous(df: pd.DataFrame, columns: Iterable[str]) -> None:
    """Fill missing values in-place for the specified columns."""
    for col in columns:
        if col not in df.columns:
            continue
        series = df[col]
        df[col] = (
            series.interpolate(method="time", limit_direction="both")
            .ffill()
            .bfill()
            .fillna(series.median())
        )


def add_time_features(df: pd.DataFrame) -> None:
    idx = df.index
    df["hour"] = idx.hour
    df["dow"] = idx.dayofweek
    df["month"] = idx.month
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7.0)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)


def add_lagged_features(df: pd.DataFrame) -> List[str]:
    feature_cols: List[str] = []

    for lag in PRICE_LAGS:
        col = f"price_mean_lag_{lag}"
        df[col] = df["price_mean"].shift(lag)
        feature_cols.append(col)

        da_col = f"price_da_lag_{lag}"
        df[da_col] = df["price_da"].shift(lag)
        feature_cols.append(da_col)

    for lag in LOAD_LAGS:
        col = f"load_forecast_mean_lag_{lag}"
        df[col] = df["load_forecast_mean"].shift(lag)
        feature_cols.append(col)

    df["load_forecast_delta_24"] = df["load_forecast_mean"] - df["load_forecast_mean_lag_24"]
    feature_cols.append("load_forecast_delta_24")

    return feature_cols


def add_future_targets(df: pd.DataFrame, horizon: int) -> List[str]:
    target_cols = []
    for h in range(1, horizon + 1):
        col = f"target_h{h}"
        df[col] = df["price_mean"].shift(-h)
        target_cols.append(col)
    return target_cols


# -----------------------------------------------------------------------------
# Dataset assembly
# -----------------------------------------------------------------------------

def build_dataset(horizon: int) -> tuple[pd.DataFrame, List[str], List[str]]:
    df = load_hourly_dataframe(DATA_FILE)

    fill_exogenous(df, IMPORTANT_EXOGENOUS)
    fill_exogenous(df, ["price_da"])

    add_time_features(df)
    lag_feature_cols = add_lagged_features(df)
    target_cols = add_future_targets(df, horizon)

    feature_cols = (
        ["price_mean", "price_da"]
        + IMPORTANT_EXOGENOUS
        + ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]
        + lag_feature_cols
    )

    required_cols = feature_cols + target_cols
    df_clean = df.dropna(subset=required_cols).copy()
    print(f"  After dropping NaNs: {len(df_clean):,} usable training rows")

    return df_clean, feature_cols, target_cols


def split_time_series(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_cols: Sequence[str],
    train_frac: float,
    val_frac: float,
) -> Dict[str, SplitResult]:
    n_total = len(df)
    train_end = int(n_total * train_frac)
    val_end = int(n_total * (train_frac + val_frac))

    splits = {
        "train": SplitResult(
            "train",
            df[feature_cols].iloc[:train_end].to_numpy(dtype=np.float32),
            df[target_cols].iloc[:train_end].to_numpy(dtype=np.float32),
            df.index[:train_end],
        ),
        "val": SplitResult(
            "val",
            df[feature_cols].iloc[train_end:val_end].to_numpy(dtype=np.float32),
            df[target_cols].iloc[train_end:val_end].to_numpy(dtype=np.float32),
            df.index[train_end:val_end],
        ),
        "test": SplitResult(
            "test",
            df[feature_cols].iloc[val_end:].to_numpy(dtype=np.float32),
            df[target_cols].iloc[val_end:].to_numpy(dtype=np.float32),
            df.index[val_end:],
        ),
    }

    for name, split in splits.items():
        if len(split.X) == 0:
            raise ValueError(f"{name} split is empty — adjust fractions.")
        print(
            f"  {name.title():<5} → {len(split.X):,} rows "
            f"({split.timestamps[0]} → {split.timestamps[-1]})"
        )

    return splits


# -----------------------------------------------------------------------------
# Training / evaluation
# -----------------------------------------------------------------------------

def train_random_forest(train_split: SplitResult) -> RandomForestRegressor:
    print("\n" + "=" * 80)
    print("Training multi-output Random Forest")
    print("=" * 80)
    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(train_split.X, train_split.y)
    print("  ✓ Model fitted")
    return model


def evaluate_split(
    model: RandomForestRegressor,
    split: SplitResult,
    label: str,
) -> Dict[str, float]:
    preds = model.predict(split.X)
    mae_per_horizon = np.mean(np.abs(preds - split.y), axis=0)
    rmse_per_horizon = np.sqrt(np.mean((preds - split.y) ** 2, axis=0))

    hours = np.arange(1, len(mae_per_horizon) + 1)
    summary = {
        "mae_mean": float(mae_per_horizon.mean()),
        "mae_1h": float(mae_per_horizon[0]),
        "mae_24h": float(mae_per_horizon[23]),
        "mae_48h": float(mae_per_horizon[-1]),
        "rmse_mean": float(rmse_per_horizon.mean()),
        "mae_per_horizon": mae_per_horizon.tolist(),
        "rmse_per_horizon": rmse_per_horizon.tolist(),
    }

    print(f"\n[{label.upper()}] mean MAE: {summary['mae_mean']:.2f} $/MWh")
    print(
        "    Horizon MAE (1/24/48h): "
        f"{summary['mae_1h']:.2f} / {summary['mae_24h']:.2f} / {summary['mae_48h']:.2f}"
    )

    best_idx = np.argsort(mae_per_horizon)[:5]
    print(
        "    Best horizons:",
        ", ".join(f"{hours[i]}→{mae_per_horizon[i]:.2f}" for i in best_idx),
    )

    return summary


def sample_forecast(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_cols: Sequence[str],
    model: RandomForestRegressor,
    origin_time: pd.Timestamp,
) -> Dict[str, List[float]]:
    """Generate a single 48-hour forecast for inspection."""
    if origin_time not in df.index:
        raise ValueError(f"Origin {origin_time} not in dataset index")

    row = df.loc[origin_time, feature_cols].to_numpy(dtype=np.float32).reshape(1, -1)
    preds = model.predict(row)[0]
    row_actual = df.loc[origin_time, target_cols].to_numpy(dtype=np.float32)
    hours = list(range(1, FORECAST_HORIZON + 1))
    return {
        "hours": hours,
        "pred": preds.tolist(),
        "actual": row_actual.tolist(),
        "origin": origin_time.isoformat(),
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    df, feature_cols, target_cols = build_dataset(FORECAST_HORIZON)
    splits = split_time_series(df, feature_cols, target_cols, TRAIN_FRACTION, VAL_FRACTION)

    model = train_random_forest(splits["train"])

    metrics = {
        split_name: evaluate_split(model, split, split_name)
        for split_name, split in splits.items()
    }

    # Pick last timestamp from validation set for inspection
    sample_origin = splits["test"].timestamps[len(splits["test"].timestamps) // 2]
    sample = sample_forecast(df, feature_cols, target_cols, model, sample_origin)

    joblib.dump(model, MODEL_PATH)
    FEATURES_PATH.write_text("\n".join(feature_cols))
    metadata = {
        "created": datetime.utcnow().isoformat(),
        "data_file": DATA_FILE,
        "forecast_horizon": FORECAST_HORIZON,
        "train_rows": len(splits["train"].X),
        "val_rows": len(splits["val"].X),
        "test_rows": len(splits["test"].X),
        "feature_count": len(feature_cols),
        "rf_params": RF_PARAMS,
        "metrics": metrics,
        "sample_forecast": sample,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))

    print("\n" + "=" * 80)
    print("Artifacts saved:")
    print(f"  • Model:          {MODEL_PATH}")
    print(f"  • Feature list:   {FEATURES_PATH}")
    print(f"  • Metadata JSON:  {METADATA_PATH}")
    print("=" * 80)
    print("Sample forecast origin:", sample["origin"])
    print("First 5 preds:", [round(v, 2) for v in sample["pred"][:5]])
    print("First 5 actuals:", [round(v, 2) for v in sample["actual"][:5]])


if __name__ == "__main__":
    main()

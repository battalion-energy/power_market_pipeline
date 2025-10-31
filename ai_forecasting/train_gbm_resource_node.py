#!/usr/bin/env python3
"""
Train a multi-horizon GBM forecaster for a specific ERCOT resource node price.

Targets the real-time resource-node LMP for GAMBIT_BESS1 while leveraging the
system-wide feature matrix (net load, ORDC, load forecasts, etc.).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

import polars as pl

DATA_FILE = (
    "/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/"
    "master_enhanced_with_net_load_reserves_2019_2025.parquet"
)
DISPATCH_DIR = Path(
    "/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/"
    "bess_analysis/hourly/dispatch"
)

OUTPUT_DIR = Path("models")
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "gbm_multihorizon_gambit.joblib"
FEATURES_PATH = OUTPUT_DIR / "gbm_gambit_feature_columns.json"
METADATA_PATH = OUTPUT_DIR / "gbm_multihorizon_gambit_metadata.json"

NODE_NAME = "GAMBIT_BESS1"
TARGET_COL = "price_node_rt"
TARGET_DA_COL = "price_node_da"
FORECAST_HORIZON = 48

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

BASE_FEATURES = [
    "price_mean",
    "price_da",
    "load_forecast_mean",
    "load_forecast_trend_24h",
    "load_forecast_spread_pct",
    "ordc_online_reserves_min",
    "ordc_scarcity_indicator_max",
    "ordc_critical_indicator_max",
    "actual_system_load_MW",
    "net_load_MW",
    "net_load_ramp_1h",
    "net_load_ramp_3h",
    "reserve_margin_pct",
    "wind_generation_MW",
    "solar_generation_MW",
    "renewable_penetration_pct",
    "net_load_roll_24h_mean",
    "net_load_roll_24h_std",
    "net_load_roll_24h_max",
    "net_load_roll_24h_min",
    "houston_pct_of_net_load",
    "north_pct_of_net_load",
    "high_renewable_flag",
    "large_ramp_flag",
    "low_net_load_flag",
]
]

EXOGENOUS = [
    col
    for col in BASE_FEATURES
    if col not in {"price_mean", "price_da"}
]

PRICE_LAGS = [1, 2, 3, 6, 12, 18, 24, 30, 36, 48, 72, 96, 120, 144, 168]
LOAD_LAGS = [1, 6, 12, 24]


@dataclass
class Split:
    name: str
    X: np.ndarray
    y: np.ndarray
    timestamps: Sequence[pd.Timestamp]


def load_master_features(path: str) -> pd.DataFrame:
    columns = list({*BASE_FEATURES, "timestamp"})
    df = pl.read_parquet(path, columns=columns).group_by("timestamp").mean().sort("timestamp")
    pdf = df.to_pandas()
    pdf["timestamp"] = pd.to_datetime(pdf["timestamp"])
    pdf = pdf.drop_duplicates("timestamp").set_index("timestamp").sort_index()
    for col in pdf.columns:
        pdf[col] = pd.to_numeric(pdf[col], errors="coerce")
    return pdf


def load_dispatch_prices(node: str) -> pd.DataFrame:
    files = sorted(DISPATCH_DIR.glob(f"{node}_*_dispatch.parquet"))
    if not files:
        raise FileNotFoundError(f"No dispatch files found for {node}")
    frames = []
    for fp in files:
        df = pl.read_parquet(fp).with_columns(
            pl.col("hour_start_local")
            .dt.replace_time_zone(None)
            .alias("timestamp")
        )
        rt_col = (
            (pl.col("rt_price_avg") if "rt_price_avg" in df.columns else pl.lit(None))
            .cast(pl.Float64)
            .alias(TARGET_COL)
        )
        da_col = (
            (pl.col("da_price_hour") if "da_price_hour" in df.columns else pl.lit(None))
            .cast(pl.Float64)
            .alias(TARGET_DA_COL)
        )
        frames.append(
            df.select(
                "timestamp",
                rt_col,
                da_col,
            )
        )
    merged = pl.concat(frames)
    pdf = merged.to_pandas()
    pdf["timestamp"] = pd.to_datetime(pdf["timestamp"])
    pdf.drop_duplicates("timestamp", inplace=True)
    pdf.set_index("timestamp", inplace=True)
    pdf.sort_index(inplace=True)
    return pdf


def fill_exogenous(df: pd.DataFrame, columns: Iterable[str]) -> None:
    for col in columns:
        if col not in df.columns:
            continue
        series = df[col]
        df[col] = (
            series.interpolate(method="time", limit_direction="both")
            .ffill()
            .bfill()
        )


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
        col_node = f"{TARGET_COL}_lag_{lag}"
        df[col_node] = df[TARGET_COL].shift(lag)
        feature_cols.append(col_node)

        col_mean = f"price_mean_lag_{lag}"
        df[col_mean] = df["price_mean"].shift(lag)
        feature_cols.append(col_mean)

        col_da = f"price_da_lag_{lag}"
        df[col_da] = df["price_da"].shift(lag)
        feature_cols.append(col_da)

    for lag in LOAD_LAGS:
        col = f"load_forecast_mean_lag_{lag}"
        df[col] = df["load_forecast_mean"].shift(lag)
        feature_cols.append(col)

    df["price_diff_node_sys"] = df[TARGET_COL] - df["price_mean"]
    df["price_diff_da"] = df[TARGET_DA_COL] - df["price_da"]
    df["price_node_rolling_24"] = df[TARGET_COL].rolling(24).mean()
    df["price_node_rolling_168"] = df[TARGET_COL].rolling(168).mean()

    feature_cols.extend(
        [
            "price_diff_node_sys",
            "price_diff_da",
            "price_node_rolling_24",
            "price_node_rolling_168",
        ]
    )

    return feature_cols


def add_future_targets(df: pd.DataFrame, horizon: int) -> List[str]:
    cols = []
    for h in range(1, horizon + 1):
        col = f"target_node_h{h}"
        df[col] = df[TARGET_COL].shift(-h)
        cols.append(col)
    return cols


def build_dataset() -> tuple[pd.DataFrame, List[str], List[str]]:
    master = load_master_features(DATA_FILE)
    dispatch = load_dispatch_prices(NODE_NAME)

    merged = master.join(dispatch, how="inner")
    merged.sort_index(inplace=True)

    fill_exogenous(merged, EXOGENOUS + [TARGET_COL, TARGET_DA_COL, "price_mean", "price_da"])
    add_time_features(merged)
    lag_cols = add_lag_features(merged)
    target_cols = add_future_targets(merged, FORECAST_HORIZON)

    feature_cols = (
        BASE_FEATURES
        + [TARGET_COL, TARGET_DA_COL]
        + ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]
        + lag_cols
    )

    required = feature_cols + target_cols
    clean = merged.dropna(subset=required).copy()
    return clean, feature_cols, target_cols


def make_supervised(
    df: pd.DataFrame, feature_cols: List[str], target_cols: List[str]
) -> tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]:
    records = df[feature_cols].to_numpy(dtype=np.float32)
    targets = df[target_cols].to_numpy(dtype=np.float32)
    timestamps = df.index.to_list()
    return records, targets, timestamps


def split_by_time(
    X: np.ndarray,
    y: np.ndarray,
    timestamps: List[pd.Timestamp],
    train_end: pd.Timestamp,
    val_end: pd.Timestamp,
) -> Dict[str, Split]:
    ts = np.array(timestamps)
    train_mask = ts <= train_end
    val_mask = (ts > train_end) & (ts <= val_end)
    test_mask = ts > val_end

    splits = {
        "train": Split("train", X[train_mask], y[train_mask], ts[train_mask]),
        "val": Split("val", X[val_mask], y[val_mask], ts[val_mask]),
        "test": Split("test", X[test_mask], y[test_mask], ts[test_mask]),
    }
    for name, split in splits.items():
        print(f"{name.title()} rows: {len(split.X):,}")
    return splits


def build_pipeline() -> Pipeline:
    scaler = StandardScaler()
    gbm = HistGradientBoostingRegressor(**GB_PARAMS)
    multi = MultiOutputRegressor(gbm, n_jobs=-1)
    return Pipeline([("scaler", scaler), ("gbm", multi)])


def evaluate(model: Pipeline, split: Split) -> Dict[str, float]:
    preds = model.predict(split.X)
    mae_per_h = np.mean(np.abs(preds - split.y), axis=0)
    return {
        "mae_mean": float(mae_per_h.mean()),
        "mae_1h": float(mae_per_h[0]),
        "mae_24h": float(mae_per_h[23]),
        "mae_48h": float(mae_per_h[-1]),
        "mae_per_horizon": mae_per_h.tolist(),
    }


def main() -> None:
    print("=" * 80)
    print(f"Training GBM for {NODE_NAME} resource-node price")
    print("=" * 80)
    print(f"Started: {datetime.now()}")

    df, feature_cols, target_cols = build_dataset()
    print(f"Dataset rows: {len(df):,} covering {df.index.min()} → {df.index.max()}")

    X, y, timestamps = make_supervised(df, feature_cols, target_cols)
    splits = split_by_time(X, y, timestamps, TRAIN_END, VAL_END)

    model = build_pipeline()
    model.fit(splits["train"].X, splits["train"].y)
    print("✓ Training complete")

    metrics = {name: evaluate(model, split) for name, split in splits.items()}

    joblib.dump(model, MODEL_PATH)
    FEATURES_PATH.write_text(json.dumps(feature_cols, indent=2))
    metadata = {
        "created": datetime.utcnow().isoformat(),
        "node": NODE_NAME,
        "data_file": DATA_FILE,
        "dispatch_dir": str(DISPATCH_DIR),
        "forecast_horizon": FORECAST_HORIZON,
        "train_rows": len(splits["train"].X),
        "val_rows": len(splits["val"].X),
        "test_rows": len(splits["test"].X),
        "feature_count": len(feature_cols),
        "gb_params": GB_PARAMS,
        "metrics": metrics,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))

    print("\nArtifacts:")
    print(f"  • Model:    {MODEL_PATH}")
    print(f"  • Features: {FEATURES_PATH}")
    print(f"  • Metadata: {METADATA_PATH}")
    for name, stats in metrics.items():
        print(
            f"{name.upper()} mean MAE {stats['mae_mean']:.2f} "
            f"(1h {stats['mae_1h']:.2f} / 24h {stats['mae_24h']:.2f} / 48h {stats['mae_48h']:.2f})"
        )

    print(f"\nCompleted: {datetime.now()}")


if __name__ == "__main__":
    main()

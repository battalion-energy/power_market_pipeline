#!/usr/bin/env python3
"""
Generate 48-hour demo forecasts using the trained Random Forest model.

This script consumes the artifacts produced by train_rf_multihorizon.py and emits
demo_forecasts.json in the same schema expected by forecast_api.py / dashboards.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import joblib
import numpy as np
import pandas as pd
import polars as pl

from train_rf_multihorizon import (
    DATA_FILE,
    FORECAST_HORIZON,
    BASE_FEATURES,
    IMPORTANT_EXOGENOUS,
    load_hourly_dataframe,
    fill_exogenous,
    add_time_features,
    add_lagged_features,
    add_future_targets,
)

MODEL_PATH = Path("models/random_forest_multihorizon.joblib")
FEATURES_PATH = Path("models/random_forest_feature_columns.txt")
METADATA_PATH = Path("models/random_forest_metadata.json")
OUTPUT_JSON = Path("demo_forecasts.json")

DEMO_DATES = [
    "2024-01-16T00:00:00",
    "2024-02-20T00:00:00",
    "2024-03-15T00:00:00",
    "2024-06-20T00:00:00",
    "2024-08-15T00:00:00",
    "2024-09-06T00:00:00",
    "2024-10-01T00:00:00",
    "2024-12-31T00:00:00",
    "2025-01-15T00:00:00",
]

def logistic(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def prepare_feature_frame() -> pd.DataFrame:
    df = load_hourly_dataframe(DATA_FILE)
    fill_exogenous(df, IMPORTANT_EXOGENOUS)
    fill_exogenous(df, ["price_da"])
    add_time_features(df)
    add_lagged_features(df)
    add_future_targets(df, FORECAST_HORIZON)

    # Also add future day-ahead targets for actual comparison.
    for h in range(1, FORECAST_HORIZON + 1):
        df[f"target_da_h{h}"] = df["price_da"].shift(-h)

    return df


def quantile_band(center: float, scale: float) -> Dict[str, float]:
    z = [-1.2816, -0.6745, 0.0, 0.6745, 1.2816]
    vals = [float(np.clip(center + zi * scale, -50.0, 1500.0)) for zi in z]
    return {
        "p10": vals[0],
        "p25": vals[1],
        "p50": vals[2],
        "p75": vals[3],
        "p90": vals[4],
    }


def main() -> None:
    if not MODEL_PATH.exists():
        raise SystemExit(f"Trained model not found: {MODEL_PATH}")

    feature_cols = FEATURES_PATH.read_text().splitlines()
    metadata = json.loads(METADATA_PATH.read_text())
    mae_per_h = metadata["metrics"]["test"]["mae_per_horizon"]
    mae_array = np.array(mae_per_h, dtype=float)
    mae_array = np.clip(mae_array, 5.0, 120.0)

    df = prepare_feature_frame()
    df = df.dropna(subset=feature_cols + [f"target_h{h}" for h in range(1, FORECAST_HORIZON + 1)])

    model = joblib.load(MODEL_PATH)

    forecasts: Dict[str, dict] = {}

    for origin_iso in DEMO_DATES:
        origin_ts = pd.Timestamp(origin_iso)
        if origin_ts not in df.index:
            print(f"✗ {origin_iso} missing from feature frame (skip)")
            continue

        row = df.loc[origin_ts, feature_cols].to_numpy(dtype=np.float32).reshape(1, -1)
        preds_rt = model.predict(row)[0]
        actual_rt = df.loc[origin_ts, [f"target_h{h}" for h in range(1, FORECAST_HORIZON + 1)]].to_numpy(dtype=float)
        actual_da = df.loc[origin_ts, [f"target_da_h{h}" for h in range(1, FORECAST_HORIZON + 1)]].to_numpy(dtype=float)

        timestamps = [origin_ts + pd.Timedelta(hours=h) for h in range(1, FORECAST_HORIZON + 1)]
        entries = []

        abs_errors = np.abs(preds_rt - actual_rt)
        mae_da = float(np.mean(np.abs(actual_da - actual_da)))  # zero, placeholder
        mae_rt = float(np.mean(abs_errors))

        for idx, ts in enumerate(timestamps):
            rt_band = quantile_band(preds_rt[idx], mae_array[idx])
            da_center = actual_da[idx]  # DA prices are already known post-clear
            da_band = {
                "p10": float(da_center),
                "p25": float(da_center),
                "p50": float(da_center),
                "p75": float(da_center),
                "p90": float(da_center),
            }

            spike_high = logistic((rt_band["p90"] - 175.0) / 35.0)
            spike_extreme = logistic((rt_band["p90"] - 300.0) / 55.0)

            entries.append(
                {
                    "hour": idx + 1,
                    "timestamp": ts.isoformat(),
                    "spike_prob_high": spike_high,
                    "spike_prob_extreme": spike_extreme,
                    "da_price_p10": da_band["p10"],
                    "da_price_p25": da_band["p25"],
                    "da_price_p50": da_band["p50"],
                    "da_price_p75": da_band["p75"],
                    "da_price_p90": da_band["p90"],
                    "rt_price_p10": rt_band["p10"],
                    "rt_price_p25": rt_band["p25"],
                    "rt_price_p50": rt_band["p50"],
                    "rt_price_p75": rt_band["p75"],
                    "rt_price_p90": rt_band["p90"],
                    "actual_da": float(actual_da[idx]) if not np.isnan(actual_da[idx]) else None,
                    "actual_rt": float(actual_rt[idx]) if not np.isnan(actual_rt[idx]) else None,
                }
            )

        forecasts[origin_iso] = {
            "forecast_origin": origin_ts.isoformat(),
            "model_version": "rf_multihorizon_v1",
            "horizon_hours": FORECAST_HORIZON,
            "features": {
                "lagged_prices": True,
                "load_forecasts": True,
                "ordc": True,
                "net_load": True,
            },
            "metrics": {
                "mae_rt": mae_rt,
            },
            "forecasts": entries,
            "metadata": {
                "source_model": MODEL_PATH.name,
                "mae_reference": "test split",
            },
        }

        print(
            f"✓ {origin_iso}: mean RT MAE {mae_rt:.2f} $/MWh, "
            f"max predicted {preds_rt.max():.1f}"
        )

    if not forecasts:
        raise SystemExit("No demo forecasts generated.")

    OUTPUT_JSON.write_text(json.dumps(forecasts, indent=2))
    print(f"\nSaved {len(forecasts)} origins to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()

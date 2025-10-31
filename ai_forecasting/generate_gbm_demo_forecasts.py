#!/usr/bin/env python3
"""
Generate walk-forward demo forecasts using the GBM multi-horizon model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import polars as pl

from train_gbm_multihorizon import (
    DATA_FILE,
    FORECAST_HORIZON,
    BASE_FEATURES,
    PRICE_LAGS,
    DA_LAGS,
    LOAD_LAGS,
    SPREAD_LAGS,
    add_time_features,
    add_lag_features,
    add_future_targets,
)

MODEL_PATH = Path("models/gbm_multihorizon.joblib")
FEATURES_PATH = Path("models/gbm_feature_columns.json")
METADATA_PATH = Path("models/gbm_multihorizon_metadata.json")
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


def load_feature_frame(required_columns: List[str]) -> pd.DataFrame:
    base_cols = [
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
    columns = list({"timestamp", *base_cols})
    lf = pl.read_parquet(DATA_FILE, columns=columns)
    agg = lf.group_by("timestamp").mean().sort("timestamp")
    df = agg.to_pandas()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.drop_duplicates("timestamp").set_index("timestamp").sort_index()

    df["da_rt_spread"] = df["price_da"] - df["price_mean"]

    add_time_features(df)
    add_lag_features(df)
    add_future_targets(df, FORECAST_HORIZON)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.ffill().bfill()
    # Ensure all required columns exist
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing engineered columns: {missing[:5]}...")
    return df


def quantiles(center: float, scale: float) -> Dict[str, float]:
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
        raise SystemExit("GBM model not found. Run train_gbm_multihorizon.py first.")

    feature_cols = json.loads(FEATURES_PATH.read_text())
    metadata = json.loads(METADATA_PATH.read_text())
    mae_curve = np.array(metadata["metrics"]["test"]["mae_per_horizon"], dtype=float)
    mae_curve = np.clip(mae_curve, 5.0, 120.0)

    df = load_feature_frame(feature_cols + [f"target_rt_h{h}" for h in range(1, FORECAST_HORIZON + 1)])

    model = joblib.load(MODEL_PATH)

    forecasts: Dict[str, dict] = {}
    for origin_iso in DEMO_DATES:
        origin_ts = pd.Timestamp(origin_iso)
        if origin_ts not in df.index:
            print(f"✗ {origin_iso}: missing from frame")
            continue

        row = df.loc[origin_ts, feature_cols].to_numpy(dtype=np.float32).reshape(1, -1)
        preds = model.predict(row)[0]
        actuals = df.loc[
            origin_ts, [f"target_rt_h{h}" for h in range(1, FORECAST_HORIZON + 1)]
        ].to_numpy(dtype=float)

        timestamps = [origin_ts + pd.Timedelta(hours=h) for h in range(1, FORECAST_HORIZON + 1)]
        entries = []
        for idx, ts in enumerate(timestamps):
            rt_band = quantiles(preds[idx], mae_curve[idx])
            da_value = df.loc[origin_ts, "price_da"]  # current DA (already cleared)
            entries.append(
                {
                    "hour": idx + 1,
                    "timestamp": ts.isoformat(),
                    "spike_prob_high": logistic((rt_band["p90"] - 200.0) / 50.0),
                    "spike_prob_extreme": logistic((rt_band["p90"] - 400.0) / 70.0),
                    "da_price_p10": float(da_value),
                    "da_price_p25": float(da_value),
                    "da_price_p50": float(da_value),
                    "da_price_p75": float(da_value),
                    "da_price_p90": float(da_value),
                    "rt_price_p10": rt_band["p10"],
                    "rt_price_p25": rt_band["p25"],
                    "rt_price_p50": rt_band["p50"],
                    "rt_price_p75": rt_band["p75"],
                    "rt_price_p90": rt_band["p90"],
                    "actual_da": float(da_value),
                    "actual_rt": float(actuals[idx]) if not np.isnan(actuals[idx]) else None,
                }
            )

        mae_rt = float(np.mean(np.abs(preds - actuals)))
        forecasts[origin_iso] = {
            "forecast_origin": origin_ts.isoformat(),
            "model_version": "gbm_multihorizon_v1",
            "horizon_hours": FORECAST_HORIZON,
            "features": {
                "price_lags": True,
                "load_forecasts": True,
                "ordc": True,
                "weather": True,
            },
            "metrics": {
                "mae_rt": mae_rt,
            },
            "forecasts": entries,
            "metadata": {
                "source_model": MODEL_PATH.name,
            },
        }
        print(f"✓ {origin_iso}: RT mean {preds.mean():.2f}, MAE {mae_rt:.2f}")

    if not forecasts:
        raise SystemExit("No forecasts generated.")

    OUTPUT_JSON.write_text(json.dumps(forecasts, indent=2))
    print(f"\nSaved {len(forecasts)} origins to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()

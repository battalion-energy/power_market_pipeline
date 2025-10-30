#!/usr/bin/env python3
"""
Create Walk-Forward Demo Forecasts with a Fast Linear Baseline.

This script builds lightweight per-date models that capture diurnal structure using
exogenous drivers (load forecast, ORDC reserve indicators, seasonal signals) and
recent price history. It is intentionally simple so that we can regenerate believable
curves quickly for investor demos without waiting on the heavier deep learning stack.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, Mapping, Tuple

import numpy as np
import pandas as pd
import polars as pl

# Paths
DATA_FILE = (
    "/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/"
    "master_enhanced_with_net_load_reserves_2019_2025.parquet"
)
OUTPUT_FILE = Path("demo_forecasts.json")

# Forecast configuration
DEFAULT_HISTORY_DAYS = 180
DEFAULT_HORIZON = 48

# Curated demo timestamps with interesting market conditions in 2024-2025.
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

# Columns required from the parquet dataset.
BASE_COLUMNS = [
    "timestamp",
    "price_da",
    "price_mean",
    "load_forecast_mean",
    "load_forecast_trend_24h",
    "load_forecast_spread_pct",
    "ordc_online_reserves_min",
    "ordc_scarcity_indicator_max",
]

FEATURES_DA = [
    "price_da_lag_24",
    "price_da_lag_48",
    "price_da_lag_168",
    "load_forecast_mean",
    "load_forecast_mean_lag_24",
    "load_forecast_delta_24",
    "load_forecast_trend_24h",
    "load_forecast_spread_pct",
    "ordc_online_reserves_min",
    "ordc_scarcity_indicator_max",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
]

FEATURES_RT = [
    "price_mean_lag_1",
    "price_mean_lag_24",
    "price_mean_lag_48",
    "price_mean_lag_168",
    "price_da_lag_24",
    "load_forecast_mean",
    "load_forecast_delta_24",
    "load_forecast_trend_24h",
    "load_forecast_spread_pct",
    "ordc_online_reserves_min",
    "ordc_scarcity_indicator_max",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
]

LAG_SPECS_DA: Dict[str, Tuple[str, int]] = {
    "price_da_lag_24": ("price_da", 24),
    "price_da_lag_48": ("price_da", 48),
    "price_da_lag_168": ("price_da", 168),
}

LAG_SPECS_RT: Dict[str, Tuple[str, int]] = {
    "price_mean_lag_1": ("price_mean", 1),
    "price_mean_lag_24": ("price_mean", 24),
    "price_mean_lag_48": ("price_mean", 48),
    "price_mean_lag_168": ("price_mean", 168),
    "price_da_lag_24": ("price_da", 24),
}


def load_dataset(data_file: str) -> pd.DataFrame:
    """Load parquet data, aggregate to hourly granularity, and add engineered features."""
    print(f"Loading data from {data_file} ...")
    lf = pl.read_parquet(data_file, columns=BASE_COLUMNS)
    lf = lf.group_by("timestamp").mean().sort("timestamp")
    df = lf.to_pandas()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = (
        df.drop_duplicates(subset="timestamp")
        .set_index("timestamp")
        .sort_index()
        .astype(float, errors="ignore")
    )

    exogenous = [
        "load_forecast_mean",
        "load_forecast_trend_24h",
        "load_forecast_spread_pct",
        "ordc_online_reserves_min",
        "ordc_scarcity_indicator_max",
    ]
    for col in exogenous:
        if col in df:
            series = df[col].astype(float)
            df[col] = (
                series.interpolate(method="time", limit_direction="both")
                .ffill()
                .bfill()
                .fillna(0.0)
            )

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

    for lag in [1, 24, 48, 168]:
        df[f"price_da_lag_{lag}"] = df["price_da"].shift(lag)
        df[f"price_mean_lag_{lag}"] = df["price_mean"].shift(lag)

    df["load_forecast_mean_lag_24"] = df["load_forecast_mean"].shift(24)
    df["load_forecast_delta_24"] = (
        df["load_forecast_mean"] - df["load_forecast_mean_lag_24"]
    )

    print(
        f"  Loaded {len(df):,} hourly rows from "
        f"{df.index.min().date()} to {df.index.max().date()}"
    )
    return df


def fit_linear_model(
    train_df: pd.DataFrame,
    features: Iterable[str],
    target_col: str,
) -> Mapping[str, object]:
    """Fit a simple linear model (with intercept) on the provided training data."""
    feats = list(features)
    subset = train_df.dropna(subset=[target_col] + feats)
    if subset.empty:
        raise ValueError(f"no data available to fit {target_col}")

    fill = subset[feats].median().fillna(0.0).reindex(feats)
    standardized = subset[feats].fillna(fill)

    means = standardized.mean().reindex(feats)
    stds = standardized.std().replace(0, 1.0).reindex(feats)

    X = ((standardized - means) / stds).to_numpy()
    y = subset[target_col].to_numpy()
    X = np.column_stack([X, np.ones(len(X))])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)

    preds = X @ beta
    residuals = y - preds
    resid_default = float(np.nanstd(residuals, ddof=1)) if len(residuals) else 10.0
    resid_default = resid_default if resid_default > 0 else 10.0

    resid_by_hour = (
        subset.assign(resid=residuals)
        .groupby("hour")["resid"]
        .std()
        .dropna()
        .to_dict()
    )

    return {
        "features": feats,
        "fill": fill,
        "mean": means,
        "std": stds,
        "beta": beta,
        "resid_default": resid_default,
        "resid_by_hour": resid_by_hour,
        "train_rows": int(len(subset)),
    }


def predict_sequence(
    df: pd.DataFrame,
    timestamps: Iterable[pd.Timestamp],
    model: Mapping[str, object],
    lag_spec: Mapping[str, Tuple[str, int]],
    target_series: str,
    aux_predictions: Mapping[str, Mapping[pd.Timestamp, float]] | None = None,
) -> Tuple[OrderedDict[pd.Timestamp, float], Dict[pd.Timestamp, float]]:
    """Predict sequentially while replacing lag features with prior forecasts."""
    feats = model["features"]
    fill = model["fill"]
    means = model["mean"]
    stds = model["std"]
    beta = model["beta"]
    resid_default = float(model["resid_default"])
    resid_by_hour = model["resid_by_hour"]

    predictions: OrderedDict[pd.Timestamp, float] = OrderedDict()
    std_map: Dict[pd.Timestamp, float] = {}

    for ts in timestamps:
        row = df.loc[ts, feats].copy()

        for feature_name, (series_name, lag_hours) in lag_spec.items():
            if feature_name not in row:
                continue
            ref_time = ts - timedelta(hours=lag_hours)
            value = None
            if series_name == target_series and ref_time in predictions:
                value = predictions[ref_time]
            elif (
                aux_predictions
                and series_name in aux_predictions
                and ref_time in aux_predictions[series_name]
            ):
                value = aux_predictions[series_name][ref_time]
            elif ref_time in df.index:
                value = df.at[ref_time, series_name]
            row[feature_name] = value

        row = row.fillna(fill)
        scaled = (row - means) / stds
        vec = np.append(scaled.to_numpy(), 1.0)
        pred = float(np.clip(vec @ beta, -50.0, 1500.0))

        predictions[ts] = pred
        hour = int(df.at[ts, "hour"]) if "hour" in df.columns else ts.hour
        std = resid_by_hour.get(hour, resid_default)
        std_map[ts] = float(np.clip(std, 3.0, 120.0))

    return predictions, std_map


def quantiles_from_prediction(pred: float, std: float) -> Tuple[float, float, float, float, float]:
    """Generate P10-P90 quantiles assuming symmetric residuals."""
    z_scores = [-1.2816, -0.6745, 0.0, 0.6745, 1.2816]
    return tuple(float(np.clip(pred + z * std, -50.0, 1500.0)) for z in z_scores)


def logistic(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def generate_forecast_for_origin(
    df: pd.DataFrame,
    origin: pd.Timestamp,
    history_days: int,
    horizon: int,
) -> Tuple[str, Mapping[str, object]]:
    """Train short-history models and produce a 48-hour forecast for a single origin."""
    if origin not in df.index:
        raise ValueError("origin timestamp not present in dataset")

    forecast_window = df.loc[origin: origin + timedelta(hours=horizon - 1)]
    if len(forecast_window) < horizon:
        raise ValueError("insufficient data after origin for requested horizon")

    train_cutoff = origin - timedelta(hours=1)
    train_start = train_cutoff - timedelta(days=history_days)
    train_df = df.loc[train_start:train_cutoff]
    if len(train_df) < 500:
        raise ValueError("not enough training history (need at least ~500 rows)")

    model_da = fit_linear_model(train_df, FEATURES_DA, "price_da")
    model_rt = fit_linear_model(train_df, FEATURES_RT, "price_mean")

    da_preds, da_stds = predict_sequence(
        df,
        forecast_window.index,
        model_da,
        LAG_SPECS_DA,
        target_series="price_da",
    )
    rt_preds, rt_stds = predict_sequence(
        df,
        forecast_window.index,
        model_rt,
        LAG_SPECS_RT,
        target_series="price_mean",
        aux_predictions={"price_da": da_preds},
    )

    da_values = np.array(list(da_preds.values()), dtype=float)
    rt_values = np.array(list(rt_preds.values()), dtype=float)
    actual_da = forecast_window["price_da"].to_numpy(dtype=float)
    actual_rt = forecast_window["price_mean"].to_numpy(dtype=float)

    mask_da = ~np.isnan(actual_da)
    mask_rt = ~np.isnan(actual_rt)
    mae_da = float(np.mean(np.abs(da_values[mask_da] - actual_da[mask_da]))) if mask_da.any() else None
    mae_rt = float(np.mean(np.abs(rt_values[mask_rt] - actual_rt[mask_rt]))) if mask_rt.any() else None

    entries = []
    for idx, ts in enumerate(forecast_window.index, start=1):
        da_q = quantiles_from_prediction(da_preds[ts], da_stds[ts])
        rt_q = quantiles_from_prediction(rt_preds[ts], rt_stds[ts])

        spike_high = logistic((rt_q[4] - 175.0) / 35.0)
        spike_extreme = logistic((rt_q[4] - 300.0) / 55.0)

        actual_da_val = forecast_window.at[ts, "price_da"]
        actual_rt_val = forecast_window.at[ts, "price_mean"]

        entries.append(
            {
                "hour": idx,
                "timestamp": ts.isoformat(),
                "spike_prob_high": spike_high,
                "spike_prob_extreme": spike_extreme,
                "da_price_p10": da_q[0],
                "da_price_p25": da_q[1],
                "da_price_p50": da_q[2],
                "da_price_p75": da_q[3],
                "da_price_p90": da_q[4],
                "rt_price_p10": rt_q[0],
                "rt_price_p25": rt_q[1],
                "rt_price_p50": rt_q[2],
                "rt_price_p75": rt_q[3],
                "rt_price_p90": rt_q[4],
                "actual_da": None if np.isnan(actual_da_val) else float(actual_da_val),
                "actual_rt": None if np.isnan(actual_rt_val) else float(actual_rt_val),
            }
        )

    forecast_payload = {
        "forecast_origin": origin.isoformat(),
        "model_version": "linear_baseline_v1",
        "horizon_hours": horizon,
        "training_samples_da": model_da["train_rows"],
        "training_samples_rt": model_rt["train_rows"],
        "training_window_start": train_df.index.min().isoformat(),
        "training_window_end": train_cutoff.isoformat(),
        "metrics": {
            "mae_da": mae_da,
            "mae_rt": mae_rt,
        },
        "forecasts": entries,
        "metadata": {
            "history_days": history_days,
            "features_da": FEATURES_DA,
            "features_rt": FEATURES_RT,
        },
    }

    origin_key = origin.isoformat()
    return origin_key, forecast_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate lightweight walk-forward demo forecasts."
    )
    parser.add_argument(
        "--dates",
        nargs="+",
        help="ISO8601 timestamps to forecast (default: curated demo dates).",
    )
    parser.add_argument(
        "--history-days",
        type=int,
        default=DEFAULT_HISTORY_DAYS,
        help=f"Days of history for each walk-forward fit (default: {DEFAULT_HISTORY_DAYS}).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_HORIZON,
        help=f"Forecast horizon in hours (default: {DEFAULT_HORIZON}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_dates = args.dates or DEMO_DATES

    df = load_dataset(DATA_FILE)

    demo_results: OrderedDict[str, Mapping[str, object]] = OrderedDict()
    print("=" * 80)
    print("GENERATING DEMO FORECASTS")
    print("=" * 80)

    for date_str in selected_dates:
        origin = pd.Timestamp(date_str)
        print(f"\n→ Forecast origin {origin} (history {args.history_days} days)")
        try:
            key, payload = generate_forecast_for_origin(
                df, origin, args.history_days, args.horizon
            )
        except ValueError as err:
            print(f"  ✗ Skipped: {err}")
            continue

        summary_da = payload["metrics"]["mae_da"]
        summary_rt = payload["metrics"]["mae_rt"]
        print(
            "  ✓ Forecast ready | "
            f"train rows DA {payload['training_samples_da']:,} | "
            f"MAE DA {summary_da:.2f} | MAE RT {summary_rt:.2f}"
            if summary_da is not None and summary_rt is not None
            else "  ✓ Forecast ready"
        )
        demo_results[key] = payload

    if not demo_results:
        raise SystemExit("No forecasts generated. Re-run with different dates or parameters.")

    OUTPUT_FILE.write_text(json.dumps(demo_results, indent=2))
    print("\n" + "=" * 80)
    print(f"SAVED {len(demo_results)} FORECAST SETS → {OUTPUT_FILE}")
    print("=" * 80)
    for key in demo_results:
        meta = demo_results[key]
        print(
            f"  • {key} | horizon {meta['horizon_hours']}h | "
            f"training rows {meta['training_samples_da']:,}/{meta['training_samples_rt']:,}"
        )


if __name__ == "__main__":
    main()

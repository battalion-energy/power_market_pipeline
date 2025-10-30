#!/usr/bin/env python3
"""
Export demo forecasts to a Parquet table for dashboard ingestion.

The Parquet schema mirrors the structure used elsewhere in the platform:
- one row per (forecast_origin, target_timestamp) pair
- wide columns for DA/RT quantiles and spike probabilities
- optional actuals for back-testing visuals
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

INPUT_PATH = Path("demo_forecasts.json")
OUTPUT_PATH = Path("output/demo_forecasts.parquet")
OUTPUT_PATH.parent.mkdir(exist_ok=True)


def load_forecasts(path: Path) -> dict:
    """Load forecasts whether stored as dict or list."""
    with path.open("r") as fh:
        payload = json.load(fh)

    if isinstance(payload, dict):
        return payload

    # Legacy list format from fast demo script.
    forecasts = {}
    for item in payload:
        origin = pd.to_datetime(item["forecast_origin"] if "forecast_origin" in item else item["origin_timestamp"])
        origin_iso = origin.isoformat()
        hourly = item.get("forecasts") or item.get("hourly_forecast", [])
        forecasts[origin_iso] = {
            "forecast_origin": origin_iso,
            "model_version": item.get("model_version", "unknown"),
            "horizon_hours": item.get("horizon_hours", len(hourly)),
            "forecasts": hourly,
            "metadata": item.get("metadata", {}),
        }
    return forecasts


def main() -> None:
    data = load_forecasts(INPUT_PATH)

    records = []
    for origin_iso, payload in data.items():
        origin_ts = pd.to_datetime(payload["forecast_origin"])
        model_version = payload.get("model_version", "unknown")
        meta = payload.get("metadata", {})
        horizon = payload.get("horizon_hours", 48)

        for entry in payload.get("forecasts", []):
            target_ts = pd.to_datetime(entry["timestamp"])
            hour_ahead = entry.get("hour") or int((target_ts - origin_ts).total_seconds() // 3600) + 1

            records.append(
                {
                    "forecast_origin": origin_ts,
                    "target_timestamp": target_ts,
                    "horizon_hours": hour_ahead,
                    "model_version": model_version,
                    "mae_da": meta.get("mae_da"),
                    "mae_rt": meta.get("mae_rt"),
                    "da_p10": entry.get("da_price_p10"),
                    "da_p25": entry.get("da_price_p25"),
                    "da_p50": entry.get("da_price_p50"),
                    "da_p75": entry.get("da_price_p75"),
                    "da_p90": entry.get("da_price_p90"),
                    "rt_p10": entry.get("rt_price_p10"),
                    "rt_p25": entry.get("rt_price_p25"),
                    "rt_p50": entry.get("rt_price_p50"),
                    "rt_p75": entry.get("rt_price_p75"),
                    "rt_p90": entry.get("rt_price_p90"),
                    "spike_prob_high": entry.get("spike_prob_high"),
                    "spike_prob_extreme": entry.get("spike_prob_extreme"),
                    "actual_da": entry.get("actual_da"),
                    "actual_rt": entry.get("actual_rt"),
                    "metadata_training_rows_da": meta.get("training_samples_da"),
                    "metadata_training_rows_rt": meta.get("training_samples_rt"),
                    "horizon_total_hours": horizon,
                }
            )

    if not records:
        raise SystemExit("No forecast records found in demo_forecasts.json")

    df = pd.DataFrame.from_records(records)
    df = df.sort_values(["forecast_origin", "target_timestamp"]).reset_index(drop=True)

    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"✓ Wrote {len(df):,} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

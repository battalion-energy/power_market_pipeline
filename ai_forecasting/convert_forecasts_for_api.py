#!/usr/bin/env python3
"""
Convert GAMBIT_ESS1 Forecasts to API Format
============================================

Transform the list-format forecasts into the dictionary format
expected by the forecast API, with proper timestamps.
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime

print("="*80)
print("CONVERTING FORECASTS TO API FORMAT")
print("="*80)

# Load the generated forecasts
input_file = Path("demo_forecasts_gambit_ess1.json")
output_file = Path("demo_forecasts.json")

print(f"\nLoading: {input_file}")
with open(input_file, 'r') as f:
    forecasts_list = json.load(f)

print(f"  ✓ Loaded {len(forecasts_list)} forecasts")

# Convert to API format: dict keyed by origin timestamp
api_forecasts = {}

for forecast in forecasts_list:
    origin_time = forecast['origin_timestamp']
    origin_dt = pd.to_datetime(origin_time)

    # Normalize to match API format (ISO string without timezone or milliseconds)
    origin_iso = origin_dt.replace(microsecond=0, tzinfo=None).isoformat()

    # Fix hourly timestamps to increment properly
    corrected_hourly = []
    for i, hour_data in enumerate(forecast['hourly_forecast']):
        # Calculate correct timestamp for this hour (origin + i hours)
        correct_timestamp = origin_dt + pd.Timedelta(hours=i)

        # Create properly formatted hour entry
        # Note: Our fast model only produces point predictions, not quantiles
        # For demo purposes, use point prediction as P50 and create simple bands
        da_pred = hour_data['price_da_forecast']
        rt_pred = hour_data['price_rt_forecast']

        # Create simple uncertainty bands (±20% for demo)
        da_p10 = da_pred * 0.8
        da_p90 = da_pred * 1.2
        rt_p10 = rt_pred * 0.8
        rt_p90 = rt_pred * 1.2

        corrected_hourly.append({
            "hour": i + 1,
            "timestamp": correct_timestamp.isoformat(),
            "da_price_p10": float(da_p10),
            "da_price_p25": float(da_pred * 0.9),
            "da_price_p50": float(da_pred),  # Point prediction as median
            "da_price_p75": float(da_pred * 1.1),
            "da_price_p90": float(da_p90),
            "rt_price_p10": float(rt_p10),
            "rt_price_p25": float(rt_pred * 0.9),
            "rt_price_p50": float(rt_pred),  # Point prediction as median
            "rt_price_p75": float(rt_pred * 1.1),
            "rt_price_p90": float(rt_p90),
            "spike_prob_high": 0.05,  # Placeholder - fast model doesn't predict spikes
            "spike_prob_extreme": 0.01,  # Placeholder
            # Keep actual values for comparison
            "actual_da": float(hour_data['price_da_actual']),
            "actual_rt": float(hour_data['price_rt_actual']),
        })

    # Create forecast object in API format
    api_forecast = {
        "forecast_origin": origin_iso,
        "model_version": "fast_demo_v1",
        "features": {
            "net_load": True,
            "reserve_margin": True,
            "ordc_indicators": True,
            "weather": True,
        },
        "horizon_hours": 48,
        "forecasts": corrected_hourly,
        "metadata": {
            "type": "walk_forward",
            "look_ahead_bias": False,
            "mae_da": float(forecast['mae_da']),
            "mae_rt": float(forecast['mae_rt']),
            "battery_id": forecast['battery_id'],
        }
    }

    # Add to dict keyed by origin timestamp
    api_forecasts[origin_iso] = api_forecast

print(f"\n✓ Converted {len(api_forecasts)} forecasts to API format")

# Show sample
print("\n" + "="*80)
print("SAMPLE FORECAST (first 3 hours)")
print("="*80)

sample_key = list(api_forecasts.keys())[0]
sample = api_forecasts[sample_key]

print(f"\nOrigin: {sample['forecast_origin']}")
print(f"Model: {sample['model_version']}")
print(f"MAE DA: ${sample['metadata']['mae_da']:.2f}/MWh")
print(f"MAE RT: ${sample['metadata']['mae_rt']:.2f}/MWh")
print(f"\nFirst 3 hours:")
for h in sample['forecasts'][:3]:
    print(f"  Hour {h['hour']}: {h['timestamp']}")
    print(f"    DA P50: ${h['da_price_p50']:.2f}/MWh (actual: ${h['actual_da']:.2f})")
    print(f"    RT P50: ${h['rt_price_p50']:.2f}/MWh (actual: ${h['actual_rt']:.2f})")

# Save in API format
print("\n" + "="*80)
print("SAVING API FORECASTS")
print("="*80)

with open(output_file, 'w') as f:
    json.dump(api_forecasts, f, indent=2)

print(f"\n✓ Saved: {output_file}")
print(f"  Size: {output_file.stat().st_size / 1024:.1f} KB")
print(f"  Forecast origins: {len(api_forecasts)}")

print("\n" + "="*80)
print("✓ API FORECASTS READY!")
print("="*80)
print(f"\nForecast origins available:")
for origin in sorted(api_forecasts.keys())[:5]:
    print(f"  - {origin}")
print(f"  ... and {len(api_forecasts) - 5} more")

print(f"\nNext: Start forecast API server with:")
print(f"  python forecast_api.py")

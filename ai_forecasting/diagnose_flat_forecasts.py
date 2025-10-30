#!/usr/bin/env python3
"""
Diagnose Why Forecasts Are Flat
=================================

CRITICAL ISSUE: Model produces flat forecasts (no diurnal patterns)
USER: "Even if there are modest spreads... flat forecasts are fucking useless!"

Investigate:
1. Model output variance - are predictions actually flat?
2. Target data variance - does training data have diurnal patterns?
3. Model architecture - can it capture temporal patterns?
4. Loss function - is it optimizing for mean instead of patterns?
5. Feature engineering - are temporal features working?
"""

import polars as pl
import pandas as pd
import numpy as np
import json
from pathlib import Path

print("="*80)
print("DIAGNOSING FLAT FORECAST ISSUE")
print("="*80)

# ============================================================================
# 1. CHECK DEMO FORECASTS - ARE THEY ACTUALLY FLAT?
# ============================================================================

print("\n" + "="*80)
print("1. ANALYZING DEMO FORECAST VARIANCE")
print("="*80)

forecast_file = Path("demo_forecasts.json")
with open(forecast_file, 'r') as f:
    forecasts = json.load(f)

print(f"Loaded {len(forecasts)} forecasts")

for i, (origin, forecast) in enumerate(list(forecasts.items())[:3], 1):
    print(f"\n{i}. Forecast origin: {origin}")
    print("-" * 60)

    # Extract predictions for 48 hours
    da_preds = [h['da_price_p50'] for h in forecast['forecasts']]
    rt_preds = [h['rt_price_p50'] for h in forecast['forecasts']]

    # Statistics
    da_mean = np.mean(da_preds)
    da_std = np.std(da_preds)
    da_min = np.min(da_preds)
    da_max = np.max(da_preds)
    da_range = da_max - da_min

    rt_mean = np.mean(rt_preds)
    rt_std = np.std(rt_preds)
    rt_min = np.min(rt_preds)
    rt_max = np.max(rt_preds)
    rt_range = rt_max - rt_min

    print(f"  DA Forecast:")
    print(f"    Mean: ${da_mean:.2f}, Std: ${da_std:.2f}")
    print(f"    Range: ${da_min:.2f} to ${da_max:.2f} (${da_range:.2f})")
    print(f"  RT Forecast:")
    print(f"    Mean: ${rt_mean:.2f}, Std: ${rt_std:.2f}")
    print(f"    Range: ${rt_min:.2f} to ${rt_max:.2f} (${rt_range:.2f})")

    # Show hourly pattern
    print(f"\n  First 24 hours (DA):")
    for h in range(0, 24, 4):
        print(f"    Hour {h:2d}: ${da_preds[h]:.2f}")

    # Check if truly flat (< $1 variance)
    if da_std < 1.0 and rt_std < 1.0:
        print(f"  ⚠️  FLAT FORECAST CONFIRMED (std < $1)")
    elif da_std < 5.0 and rt_std < 5.0:
        print(f"  ⚠️  VERY LOW VARIANCE (std < $5)")
    else:
        print(f"  ✓ Has some variance")

# ============================================================================
# 2. CHECK ACTUAL PRICE DATA - DOES IT HAVE DIURNAL PATTERNS?
# ============================================================================

print("\n" + "="*80)
print("2. ANALYZING ACTUAL PRICE DIURNAL PATTERNS")
print("="*80)

DATA_FILE = "/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/master_enhanced_with_net_load_reserves_2019_2025.parquet"
df = pl.read_parquet(DATA_FILE)

# Add hour of day
df = df.with_columns([
    pl.col('timestamp').dt.hour().alias('hour_of_day')
])

# Filter to 2024 (recent data)
df_2024 = df.filter(pl.col('timestamp').dt.year() == 2024)

print(f"\n2024 data: {len(df_2024):,} records")

# Average price by hour of day
hourly_avg = df_2024.group_by('hour_of_day').agg([
    pl.col('price_da').mean().alias('da_mean'),
    pl.col('price_mean').mean().alias('rt_mean'),
    pl.col('price_da').std().alias('da_std'),
    pl.col('price_mean').std().alias('rt_std'),
]).sort('hour_of_day')

print("\nActual Price Patterns by Hour of Day (2024):")
print("Hour | DA Mean | RT Mean | DA Std | RT Std")
print("-" * 60)
for row in hourly_avg.iter_rows():
    hour, da_mean, rt_mean, da_std, rt_std = row
    print(f"{hour:4d} | ${da_mean:6.2f} | ${rt_mean:6.2f} | ${da_std:5.2f} | ${rt_std:5.2f}")

# Calculate diurnal range
da_min_hour = hourly_avg.select(pl.col('da_mean').min())[0,0]
da_max_hour = hourly_avg.select(pl.col('da_mean').max())[0,0]
rt_min_hour = hourly_avg.select(pl.col('rt_mean').min())[0,0]
rt_max_hour = hourly_avg.select(pl.col('rt_mean').max())[0,0]

print(f"\nDiurnal Range (2024):")
print(f"  DA: ${da_min_hour:.2f} to ${da_max_hour:.2f} (range: ${da_max_hour - da_min_hour:.2f})")
print(f"  RT: ${rt_min_hour:.2f} to ${rt_max_hour:.2f} (range: ${rt_max_hour - rt_min_hour:.2f})")

if (da_max_hour - da_min_hour) > 5:
    print("  ✓ Strong diurnal pattern in actual data")
else:
    print("  ⚠️  Weak diurnal pattern in 2024")

# ============================================================================
# 3. CHECK TRAINING DATA STATISTICS
# ============================================================================

print("\n" + "="*80)
print("3. TRAINING DATA STATISTICS")
print("="*80)

# Overall variance
da_variance = df_2024.select(pl.col('price_da').var())[0,0]
rt_variance = df_2024.select(pl.col('price_mean').var())[0,0]
da_mean_overall = df_2024.select(pl.col('price_da').mean())[0,0]
rt_mean_overall = df_2024.select(pl.col('price_mean').mean())[0,0]

print(f"\n2024 Price Statistics:")
print(f"  DA: Mean = ${da_mean_overall:.2f}, Variance = {da_variance:.2f}, Std = ${np.sqrt(da_variance):.2f}")
print(f"  RT: Mean = ${rt_mean_overall:.2f}, Variance = {rt_variance:.2f}, Std = ${np.sqrt(rt_variance):.2f}")

# Check normalization impact
print(f"\n⚠️  POTENTIAL ISSUE: If model was normalized during training,")
print(f"    it may have learned to predict the mean (near zero in normalized space)")
print(f"    and lost the diurnal pattern signal.")

# ============================================================================
# 4. ROOT CAUSE HYPOTHESIS
# ============================================================================

print("\n" + "="*80)
print("4. PROBABLE ROOT CAUSES")
print("="*80)

print("""
LIKELY CAUSES OF FLAT FORECASTS:

1. NORMALIZATION ISSUE (Most Likely):
   - Model trained on StandardScaler normalized data
   - Predictions collapse to mean in normalized space
   - Denormalization doesn't recover patterns
   - FIX: Use different normalization or loss function

2. INSUFFICIENT TEMPORAL FEATURES:
   - hour_sin, hour_cos may not be enough
   - Model may need explicit hour-of-day embeddings
   - FIX: Add hour-of-day categorical encoding

3. LOSS FUNCTION:
   - MSE loss encourages predicting the mean
   - Doesn't penalize lack of variance
   - FIX: Use quantile loss or add variance penalty

4. MODEL CAPACITY:
   - 665K params may be too small
   - Can't capture complex temporal patterns
   - FIX: Larger model with more layers

5. EARLY STOPPING TOO AGGRESSIVE:
   - Stopped at epoch 10 (val loss 0.0201)
   - May not have learned patterns yet
   - FIX: Train longer, monitor pattern metrics

6. FEATURE ENGINEERING:
   - Missing key temporal drivers (hour-of-day load pattern)
   - FIX: Add hourly load forecast patterns

IMMEDIATE ACTION NEEDED:
- Retrain with different approach
- Focus on capturing patterns, not just mean
- Add explicit temporal features
- Train longer
- Use pattern-aware loss function
""")

print("\n" + "="*80)
print("✓ DIAGNOSIS COMPLETE")
print("="*80)
print("\nNext: Implement fixes and retrain model")
print("Priority: Get ANY diurnal pattern, even if accuracy is worse")

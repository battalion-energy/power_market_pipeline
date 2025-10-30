#!/usr/bin/env python3
"""
Create Enhanced Master Dataset with ORDC + Load Forecasts
Combines existing features with new ORDC and load forecast data
"""

import pandas as pd
import polars as pl
from pathlib import Path
from datetime import datetime
import numpy as np

# Paths
EXISTING_MASTER = Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/master_features_multihorizon_2019_2025.parquet")
ORDC_FILE = Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/processed/ordc_historical_hourly_2018_2025.parquet")
LOAD_FILE = Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/processed/load_forecasts_7day_2022_2025.parquet")
OUTPUT_DIR = Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data")

print("="*60)
print("Creating Enhanced Master Dataset")
print("="*60)
print(f"Started: {datetime.now()}")

# Load existing master dataset
print(f"\n{'='*60}")
print("Loading existing master dataset...")
df_master = pd.read_parquet(EXISTING_MASTER)

# Reset index to make timestamp a column
if df_master.index.name == 'timestamp':
    df_master = df_master.reset_index()
df_master['timestamp'] = pd.to_datetime(df_master['timestamp'])

print(f"Loaded: {len(df_master):,} records")
print(f"Date range: {df_master['timestamp'].min()} to {df_master['timestamp'].max()}")
print(f"Existing features: {len(df_master.columns)}")

# Load ORDC data
print(f"\n{'='*60}")
print("Loading ORDC data...")
df_ordc = pd.read_parquet(ORDC_FILE)
df_ordc['datetime'] = pd.to_datetime(df_ordc['datetime'])

print(f"Loaded: {len(df_ordc):,} records")
print(f"Date range: {df_ordc['datetime'].min()} to {df_ordc['datetime'].max()}")
print(f"ORDC columns: {list(df_ordc.columns)}")

# Load load forecast data
print(f"\n{'='*60}")
print("Loading load forecast data...")
df_load = pd.read_parquet(LOAD_FILE)
df_load['datetime'] = pd.to_datetime(df_load['datetime'])

print(f"Loaded: {len(df_load):,} records")
print(f"Date range: {df_load['datetime'].min()} to {df_load['datetime'].max()}")
print(f"Load forecast columns: {len(df_load.columns)}")

# Merge ORDC data
print(f"\n{'='*60}")
print("Merging ORDC data...")
df_enhanced = df_master.merge(
    df_ordc,
    left_on='timestamp',
    right_on='datetime',
    how='left',
    suffixes=('', '_ordc')
)
df_enhanced = df_enhanced.drop(columns=['datetime'], errors='ignore')

print(f"After ORDC merge: {len(df_enhanced):,} records, {len(df_enhanced.columns)} columns")

# Merge load forecast data
print(f"\n{'='*60}")
print("Merging load forecast data...")
df_enhanced = df_enhanced.merge(
    df_load,
    left_on='timestamp',
    right_on='datetime',
    how='left',
    suffixes=('', '_load')
)
df_enhanced = df_enhanced.drop(columns=['datetime'], errors='ignore')

print(f"After load merge: {len(df_enhanced):,} records, {len(df_enhanced.columns)} columns")

# Create derived ORDC features
print(f"\n{'='*60}")
print("Creating derived ORDC features...")

# Total ORDC price adder
df_enhanced['ordc_total_price_adder'] = (
    df_enhanced['ordc_reliability_price_adder_mean'].fillna(0)
)

# Lag features
for lag in [1, 3, 24]:
    col = f'ordc_total_price_adder_lag_{lag}h'
    df_enhanced[col] = df_enhanced['ordc_total_price_adder'].shift(lag)
    print(f"  Created: {col}")

# Rolling features
for window in [3, 24]:
    col_mean = f'ordc_total_price_adder_rolling_{window}h_mean'
    col_max = f'ordc_total_price_adder_rolling_{window}h_max'
    df_enhanced[col_mean] = df_enhanced['ordc_total_price_adder'].rolling(window).mean()
    df_enhanced[col_max] = df_enhanced['ordc_total_price_adder'].rolling(window).max()
    print(f"  Created: {col_mean}, {col_max}")

# Binary scarcity indicator
df_enhanced['ordc_scarcity_indicator'] = (df_enhanced['ordc_total_price_adder'] > 0).astype(int)
print(f"  Created: ordc_scarcity_indicator")

# ORDC statistics
ordc_nonzero = (df_enhanced['ordc_total_price_adder'] > 0).sum()
ordc_pct = 100 * ordc_nonzero / len(df_enhanced)
print(f"\nORDC scarcity events: {ordc_nonzero:,} ({ordc_pct:.2f}% of records)")

# Create derived load forecast features
print(f"\n{'='*60}")
print("Creating derived load forecast features...")

if 'load_forecast_mean' in df_enhanced.columns:
    # Load forecast trend
    df_enhanced['load_forecast_trend_24h'] = df_enhanced['load_forecast_mean'].diff(24)
    print(f"  Created: load_forecast_trend_24h")

    # Spread percentage
    df_enhanced['load_forecast_spread_pct'] = 100 * (
        df_enhanced['load_forecast_std'] / df_enhanced['load_forecast_mean'].replace(0, np.nan)
    )
    print(f"  Created: load_forecast_spread_pct")

    # Load statistics
    load_avail = df_enhanced['load_forecast_mean'].notna().sum()
    load_pct = 100 * load_avail / len(df_enhanced)
    print(f"\nLoad forecast coverage: {load_avail:,} records ({load_pct:.2f}%)")

# Summary statistics
print(f"\n{'='*60}")
print("Enhanced Dataset Summary")
print(f"{'='*60}")
print(f"\nTotal records: {len(df_enhanced):,}")
print(f"Date range: {df_enhanced['timestamp'].min()} to {df_enhanced['timestamp'].max()}")
print(f"Total features: {len(df_enhanced.columns)}")

# Count non-null values for new features
print(f"\nNew ORDC features coverage:")
ordc_cols = [col for col in df_enhanced.columns if 'ordc' in col.lower()]
for col in sorted(ordc_cols)[:10]:  # Show first 10
    non_null = df_enhanced[col].notna().sum()
    pct = 100 * non_null / len(df_enhanced)
    print(f"  {col}: {non_null:,} ({pct:.1f}%)")

print(f"\nNew load forecast features coverage:")
load_cols = [col for col in df_enhanced.columns if 'load_forecast' in col.lower()]
for col in sorted(load_cols)[:10]:  # Show first 10
    non_null = df_enhanced[col].notna().sum()
    pct = 100 * non_null / len(df_enhanced)
    print(f"  {col}: {non_null:,} ({pct:.1f}%)")

# Save enhanced dataset
output_file = OUTPUT_DIR / "master_features_enhanced_with_ordc_load_2019_2025.parquet"
print(f"\n{'='*60}")
print(f"Saving enhanced dataset...")
df_enhanced.to_parquet(output_file, index=False)

print(f"\n✓ Saved: {output_file}")
print(f"  Size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
print(f"  Records: {len(df_enhanced):,}")
print(f"  Features: {len(df_enhanced.columns)}")

# Save feature list
feature_list_file = OUTPUT_DIR / "enhanced_features_list.txt"
with open(feature_list_file, 'w') as f:
    f.write("Enhanced Master Dataset Features\n")
    f.write("="*60 + "\n\n")
    f.write(f"Total Features: {len(df_enhanced.columns)}\n\n")

    # Categorize features
    categories = {
        'Price Features': [c for c in df_enhanced.columns if 'price' in c.lower() and 'ordc' not in c.lower()],
        'ORDC Features': [c for c in df_enhanced.columns if 'ordc' in c.lower()],
        'Load Forecast Features': [c for c in df_enhanced.columns if 'load_forecast' in c.lower()],
        'Weather Features': [c for c in df_enhanced.columns if any(w in c.lower() for w in ['temp', 'wind', 'solar', 'precip', 'cloud', 'humid'])],
        'AS Features': [c for c in df_enhanced.columns if c in ['REGUP', 'REGDN', 'RRS', 'NSPIN', 'ECRS'] or 'as_' in c.lower()],
        'Temporal Features': [c for c in df_enhanced.columns if any(t in c.lower() for t in ['hour', 'day', 'month', 'year', 'season', 'weekend'])],
        'Spike Labels': [c for c in df_enhanced.columns if 'spike_' in c.lower()],
    }

    for category, features in categories.items():
        if features:
            f.write(f"\n{category} ({len(features)}):\n")
            for feat in sorted(features):
                f.write(f"  - {feat}\n")

print(f"✓ Saved feature list: {feature_list_file}")

print(f"\n{'='*60}")
print("COMPLETE!")
print(f"{'='*60}")
print(f"Finished: {datetime.now()}")
print(f"\nReady for enhanced model training!")
print(f"\nNext steps:")
print(f"1. Train spike model with ORDC features")
print(f"2. Train DA+RT model with load forecasts")
print(f"3. Compare to baseline models")

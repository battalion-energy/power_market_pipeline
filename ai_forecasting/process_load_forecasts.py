#!/usr/bin/env python3
"""
Process 7-Day Load Forecasts into Parquet
Critical for DA price forecasting
"""

import pandas as pd
import polars as pl
from pathlib import Path
from datetime import datetime
import glob

# Paths
LOAD_CSV_DIR = Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/Seven-Day Load Forecast by Model and Study Area/csv")
OUTPUT_DIR = Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*60)
print("Processing 7-Day Load Forecasts (2022-2025)")
print("="*60)
print(f"Started: {datetime.now()}")

# Find all CSV files
csv_files = sorted(LOAD_CSV_DIR.glob("*.csv"))
print(f"\nFound {len(csv_files):,} files")

if len(csv_files) == 0:
    print("ERROR: No CSV files found")
    exit(1)

# Sample first file to understand structure
print(f"\nSampling first file: {csv_files[0].name}")
sample = pd.read_csv(csv_files[0], nrows=10)
print(f"Columns: {sample.columns.tolist()}")
print(f"Sample data:\n{sample.head()}")

# Process files in batches
all_data = []
batch_size = 1000
total_files = len(csv_files)

for i in range(0, total_files, batch_size):
    batch_files = csv_files[i:i+batch_size]
    print(f"\nProcessing batch {i//batch_size + 1}/{(total_files + batch_size - 1)//batch_size} ({len(batch_files)} files)...")

    batch_data = []
    for csv_file in batch_files:
        try:
            df = pd.read_csv(csv_file)

            # Parse datetime
            # DeliveryDate is like "06/17/2022", HourEnding is like "1:00"
            df['DeliveryDate'] = pd.to_datetime(df['DeliveryDate'])
            df['hour'] = df['HourEnding'].str.replace(':00', '').astype(int)

            # Create datetime (hour ending, so subtract 1 for hour beginning)
            df['datetime'] = df['DeliveryDate'] + pd.to_timedelta(df['hour'] - 1, unit='h')

            # Keep relevant columns
            # Valley is the forecast value (MW)
            # Model indicates which forecast model (E, E1, E2, E3, etc.)
            df = df[['datetime', 'Valley', 'Model', 'DSTFlag']]

            batch_data.append(df)

        except Exception as e:
            print(f"  Error with {csv_file.name}: {e}")
            continue

    if batch_data:
        batch_combined = pd.concat(batch_data, ignore_index=True)
        all_data.append(batch_combined)
        print(f"  ✓ Processed {len(batch_combined):,} records")

if not all_data:
    print("ERROR: No data processed")
    exit(1)

# Combine all batches
print(f"\n{'='*60}")
print("Combining all batches...")
combined = pd.concat(all_data, ignore_index=True)
combined = combined.sort_values(['datetime', 'Model']).reset_index(drop=True)

print(f"Total records: {len(combined):,}")
print(f"Date range: {combined['datetime'].min()} to {combined['datetime'].max()}")
print(f"Models: {combined['Model'].unique()}")

# Pivot to have one row per datetime with columns for each model
print(f"\n{'='*60}")
print("Pivoting to wide format...")

# Create separate column for each forecast model
pivot = combined.pivot_table(
    index='datetime',
    columns='Model',
    values='Valley',
    aggfunc='first'  # Take first value if duplicates
).reset_index()

# Flatten column names
pivot.columns = ['datetime'] + [f'load_forecast_{col}' for col in pivot.columns[1:]]

# Calculate ensemble statistics
model_cols = [col for col in pivot.columns if col.startswith('load_forecast_')]
if model_cols:
    pivot['load_forecast_mean'] = pivot[model_cols].mean(axis=1)
    pivot['load_forecast_median'] = pivot[model_cols].median(axis=1)
    pivot['load_forecast_std'] = pivot[model_cols].std(axis=1)
    pivot['load_forecast_min'] = pivot[model_cols].min(axis=1)
    pivot['load_forecast_max'] = pivot[model_cols].max(axis=1)

print(f"Pivoted records: {len(pivot):,}")
print(f"Columns: {list(pivot.columns)}")

# Statistics
print(f"\n{'='*60}")
print("Load Forecast Statistics:")
print(f"{'='*60}")
print(f"\nEnsemble Mean Load (MW):")
print(f"  Mean:   {pivot['load_forecast_mean'].mean():.1f}")
print(f"  Median: {pivot['load_forecast_mean'].median():.1f}")
print(f"  Min:    {pivot['load_forecast_mean'].min():.1f}")
print(f"  Max:    {pivot['load_forecast_mean'].max():.1f}")

print(f"\nForecast Spread (MW):")
print(f"  Mean Std Dev: {pivot['load_forecast_std'].mean():.1f}")
print(f"  Mean Range:   {(pivot['load_forecast_max'] - pivot['load_forecast_min']).mean():.1f}")

# Save as parquet
output_file = OUTPUT_DIR / "load_forecasts_7day_2022_2025.parquet"
pivot.to_parquet(output_file, index=False)

print(f"\n{'='*60}")
print(f"✓ Saved: {output_file}")
print(f"  Size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
print(f"  Records: {len(pivot):,}")
print(f"{'='*60}")

print(f"\n{'='*60}")
print("COMPLETE!")
print(f"{'='*60}")
print(f"Finished: {datetime.now()}")
print(f"\nReady to merge with master dataset!")

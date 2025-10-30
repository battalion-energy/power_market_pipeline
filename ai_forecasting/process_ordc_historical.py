#!/usr/bin/env python3
"""
Process Historical ORDC Data (15-minute) into Parquet
CRITICAL for spike prediction - includes Winter Storm Uri 2021!
"""

import pandas as pd
import polars as pl
from pathlib import Path
from datetime import datetime

# Paths
ORDC_CSV_DIR = Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/Historical Real-Time ORDC and Reliability Deployment Prices for 15-minute Settlement Interval/csv")
OUTPUT_DIR = Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*60)
print("Processing Historical ORDC Data (2018-2025)")
print("="*60)
print(f"Started: {datetime.now()}")

# Find all ORDC CSV files
csv_files = sorted(ORDC_CSV_DIR.glob("RTM_ORDC_REL_DPLY_PRC_15MINT_*.csv"))
print(f"\nFound {len(csv_files)} files:")
for f in csv_files:
    print(f"  - {f.name}")

all_data = []

for csv_file in csv_files:
    year = csv_file.stem.split('_')[-1]
    print(f"\nProcessing {year}...")

    try:
        # Read CSV
        df = pd.read_csv(csv_file)

        print(f"  Loaded: {len(df):,} records")
        print(f"  Columns: {list(df.columns)}")

        # Parse datetime
        df['datetime'] = pd.to_datetime(df['DeliveryDate']) + pd.to_timedelta((df['DeliveryHour'] - 1) * 60 + (df['DeliveryInterval'] - 1) * 15, unit='m')

        # Rename columns for clarity
        df = df.rename(columns={
            'RTRSVPOR': 'ordc_online_reserves',      # Online reserves (MW)
            'RTRSVPOFF': 'ordc_offline_reserves',    # Offline reserves (MW)
            'RTRDP': 'ordc_reliability_price_adder'  # Reliability deployment price adder ($/MWh)
        })

        # Keep only what we need
        df = df[['datetime', 'ordc_online_reserves', 'ordc_offline_reserves', 'ordc_reliability_price_adder']]

        # Calculate derived features
        df['ordc_total_reserves'] = df['ordc_online_reserves'] + df['ordc_offline_reserves']
        df['ordc_distance_to_3000'] = (df['ordc_online_reserves'] - 3000).clip(lower=0)
        df['ordc_scarcity_indicator'] = (df['ordc_online_reserves'] < 2000).astype(int)
        df['ordc_critical_indicator'] = (df['ordc_online_reserves'] < 1000).astype(int)

        all_data.append(df)

        print(f"  ✓ Processed {len(df):,} records")
        print(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")

    except Exception as e:
        print(f"  ✗ Error: {e}")
        continue

# Combine all years
print(f"\n{'='*60}")
print("Combining all years...")
combined = pd.concat(all_data, ignore_index=True)
combined = combined.sort_values('datetime').reset_index(drop=True)

print(f"Total records: {len(combined):,}")
print(f"Date range: {combined['datetime'].min()} to {combined['datetime'].max()}")
print(f"\nColumns: {list(combined.columns)}")

# Show statistics
print(f"\n{'='*60}")
print("ORDC Statistics:")
print(f"{'='*60}")
print(f"\nOnline Reserves (MW):")
print(f"  Mean:   {combined['ordc_online_reserves'].mean():.1f}")
print(f"  Median: {combined['ordc_online_reserves'].median():.1f}")
print(f"  Min:    {combined['ordc_online_reserves'].min():.1f}")
print(f"  Max:    {combined['ordc_online_reserves'].max():.1f}")

print(f"\nScarcity Events:")
print(f"  Reserves < 3000 MW: {(combined['ordc_online_reserves'] < 3000).sum():,} intervals ({100 * (combined['ordc_online_reserves'] < 3000).mean():.2f}%)")
print(f"  Reserves < 2000 MW: {(combined['ordc_online_reserves'] < 2000).sum():,} intervals ({100 * (combined['ordc_online_reserves'] < 2000).mean():.2f}%)")
print(f"  Reserves < 1000 MW: {(combined['ordc_online_reserves'] < 1000).sum():,} intervals ({100 * (combined['ordc_online_reserves'] < 1000).mean():.2f}%)")

# Aggregate to hourly (since our other data is hourly)
print(f"\n{'='*60}")
print("Aggregating to hourly...")
print(f"{'='*60}")

combined['hour'] = combined['datetime'].dt.floor('h')
hourly = combined.groupby('hour').agg({
    'ordc_online_reserves': ['mean', 'min', 'max'],
    'ordc_offline_reserves': ['mean', 'min', 'max'],
    'ordc_reliability_price_adder': ['mean', 'max'],
    'ordc_total_reserves': 'mean',
    'ordc_distance_to_3000': 'min',  # Minimum distance (worst case)
    'ordc_scarcity_indicator': 'max',  # Any scarcity in the hour
    'ordc_critical_indicator': 'max',  # Any critical in the hour
}).reset_index()

# Flatten column names
hourly.columns = ['datetime'] + [f'{col[0]}_{col[1]}' if col[1] else col[0] for col in hourly.columns[1:]]

print(f"Hourly records: {len(hourly):,}")
print(f"Columns: {list(hourly.columns)}")

# Save as parquet
output_file = OUTPUT_DIR / "ordc_historical_hourly_2018_2025.parquet"
hourly.to_parquet(output_file, index=False)

print(f"\n{'='*60}")
print(f"✓ Saved: {output_file}")
print(f"  Size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
print(f"  Records: {len(hourly):,}")
print(f"{'='*60}")

# Also save 15-minute version for detailed analysis
output_15min = OUTPUT_DIR / "ordc_historical_15min_2018_2025.parquet"
combined.to_parquet(output_15min, index=False)
print(f"✓ Saved 15-min: {output_15min}")
print(f"  Size: {output_15min.stat().st_size / 1024 / 1024:.1f} MB")

print(f"\n{'='*60}")
print("COMPLETE!")
print(f"{'='*60}")
print(f"Finished: {datetime.now()}")
print(f"\nReady to merge with master dataset!")

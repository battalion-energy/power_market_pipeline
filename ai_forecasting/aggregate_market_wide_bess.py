#!/usr/bin/env python3
"""
Aggregate Market-Wide BESS Dispatch from Individual Resource Files
===================================================================

You already have hourly BESS dispatch data by individual resource!
This script aggregates ALL BESS resources to create a market-wide
timeseries showing total BESS activity.

Critical for price forecasting:
- Total BESS charging (increases demand → raises prices)
- Total BESS discharging (increases supply → lowers prices)
- BESS capacity growth 2019→2025 (GW scale)
"""

import polars as pl
from pathlib import Path
from datetime import datetime
import glob

print("="*80)
print("AGGREGATING MARKET-WIDE BESS DISPATCH")
print("="*80)
print(f"Started: {datetime.now()}")

# Paths
BESS_DISPATCH_DIR = Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/bess_analysis/hourly/dispatch")
OUTPUT_DIR = Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 1. EXPLORE ONE FILE TO UNDERSTAND STRUCTURE
# ============================================================================

print("\n" + "="*80)
print("1. EXPLORING FILE STRUCTURE")
print("="*80)

sample_files = list(BESS_DISPATCH_DIR.glob("ANCHOR_BESS1_2024_dispatch.parquet"))
if sample_files:
    sample = pl.read_parquet(sample_files[0])
    print(f"\nSample file: {sample_files[0].name}")
    print(f"Records: {len(sample):,}")
    print(f"Columns ({len(sample.columns)}): {sample.columns}")
    print("\nSample data:")
    print(sample.head(5))
else:
    print("⚠️  No BESS dispatch files found!")
    exit(1)

# ============================================================================
# 2. LOAD ALL BESS DISPATCH FILES
# ============================================================================

print("\n" + "="*80)
print("2. LOADING ALL BESS DISPATCH FILES")
print("="*80)

dispatch_files = list(BESS_DISPATCH_DIR.glob("*_dispatch.parquet"))
print(f"Found {len(dispatch_files)} BESS dispatch files")

all_dispatch = []
bess_resources = set()
years = set()

for file in dispatch_files:
    # Extract BESS name and year from filename
    # Format: BESS_NAME_YEAR_dispatch.parquet
    parts = file.stem.split('_')
    year = parts[-2]  # Year before "dispatch"
    bess_name = '_'.join(parts[:-2])  # Everything before year

    bess_resources.add(bess_name)
    years.add(year)

    # Load dispatch data
    try:
        df = pl.read_parquet(file)

        # Add BESS name and year
        df = df.with_columns([
            pl.lit(bess_name).alias('bess_name'),
            pl.lit(int(year)).alias('year')
        ])

        all_dispatch.append(df)

    except Exception as e:
        print(f"  ⚠️  Error loading {file.name}: {e}")
        continue

print(f"\n✓ Loaded {len(all_dispatch)} files")
print(f"  Unique BESS: {len(bess_resources)}")
print(f"  Years: {sorted(years)}")

# ============================================================================
# 3. CONCATENATE ALL DISPATCH DATA
# ============================================================================

print("\n" + "="*80)
print("3. CONCATENATING ALL DISPATCH DATA")
print("="*80)

df_all = pl.concat(all_dispatch, how='diagonal_relaxed')
print(f"Total records: {len(df_all):,}")

# Standardize timestamp column name
if 'local_date' in df_all.columns:
    df_all = df_all.rename({'local_date': 'timestamp'})
elif 'hour_ending' in df_all.columns:
    df_all = df_all.rename({'hour_ending': 'timestamp'})

# Show available columns
print(f"\nAvailable columns ({len(df_all.columns)}):")
for i, col in enumerate(df_all.columns, 1):
    print(f"  {i:2}. {col}")

# ============================================================================
# 4. AGGREGATE BY HOUR (MARKET-WIDE TOTALS)
# ============================================================================

print("\n" + "="*80)
print("4. AGGREGATING TO MARKET-WIDE HOURLY TOTALS")
print("="*80)

# Identify dispatch/power columns (may vary by file structure)
# Common patterns: 'rt_position_mw', 'rt_mw', 'basepoint', etc.
power_cols = [col for col in df_all.columns if any(x in col.lower() for x in ['_mw', 'mw_', 'basepoint', 'dispatch'])]
print(f"\nPotential power columns: {power_cols}")

# Use most likely dispatch column
if 'rt_position_mw' in df_all.columns:
    dispatch_col = 'rt_position_mw'
elif 'rt_mw' in df_all.columns:
    dispatch_col = 'rt_mw'
elif 'basepoint' in df_all.columns:
    dispatch_col = 'basepoint'
else:
    # Take first MW column
    dispatch_col = power_cols[0] if power_cols else None

if dispatch_col is None:
    print("❌ ERROR: Cannot identify dispatch power column!")
    print("Available columns:", df_all.columns)
    exit(1)

print(f"\nUsing dispatch column: '{dispatch_col}'")

# Aggregate by timestamp
df_hourly = df_all.group_by('timestamp').agg([
    # Total BESS dispatch (net: positive=discharge, negative=charge)
    pl.col(dispatch_col).sum().alias('bess_dispatch_MW'),

    # Separate charging and discharging
    pl.col(dispatch_col).filter(pl.col(dispatch_col) > 0).sum().alias('bess_discharging_MW'),
    pl.col(dispatch_col).filter(pl.col(dispatch_col) < 0).sum().alias('bess_charging_MW'),

    # Count of active BESS resources
    pl.col('bess_name').n_unique().alias('bess_active_count'),

    # Mean and extremes
    pl.col(dispatch_col).mean().alias('bess_dispatch_mean'),
    pl.col(dispatch_col).max().alias('bess_dispatch_max'),
    pl.col(dispatch_col).min().alias('bess_dispatch_min'),
])

# Sort by timestamp
df_hourly = df_hourly.sort('timestamp')

print(f"\nHourly aggregated records: {len(df_hourly):,}")

# ============================================================================
# 5. CALCULATE DERIVED FEATURES
# ============================================================================

print("\n" + "="*80)
print("5. CALCULATING DERIVED FEATURES")
print("="*80)

# Fill nulls in discharge/charge (when no activity)
df_hourly = df_hourly.with_columns([
    pl.col('bess_discharging_MW').fill_null(0),
    pl.col('bess_charging_MW').fill_null(0),
])

# Hour-over-hour changes
df_hourly = df_hourly.with_columns([
    (pl.col('bess_dispatch_MW') - pl.col('bess_dispatch_MW').shift(1)).alias('bess_dispatch_change_1h'),
])

# Rolling statistics (24-hour trends)
df_hourly = df_hourly.with_columns([
    pl.col('bess_dispatch_MW').rolling_mean(window_size=24).alias('bess_dispatch_roll_24h_mean'),
    pl.col('bess_discharging_MW').rolling_mean(window_size=24).alias('bess_discharge_roll_24h_mean'),
    pl.col('bess_charging_MW').rolling_mean(window_size=24).alias('bess_charge_roll_24h_mean'),
])

# Flags for significant events
df_hourly = df_hourly.with_columns([
    # Heavy charging (> 500 MW total)
    (pl.col('bess_charging_MW') < -500).cast(pl.Int8).alias('bess_heavy_charging_flag'),

    # Heavy discharging (> 500 MW total)
    (pl.col('bess_discharging_MW') > 500).cast(pl.Int8).alias('bess_heavy_discharging_flag'),

    # Net charging (negative net dispatch)
    (pl.col('bess_dispatch_MW') < 0).cast(pl.Int8).alias('bess_net_charging_flag'),

    # Rapid swing (> 200 MW change in 1 hour)
    (pl.col('bess_dispatch_change_1h').abs() > 200).cast(pl.Int8).alias('bess_rapid_swing_flag'),
])

print("✓ Derived features calculated")

# ============================================================================
# 6. GROWTH STATISTICS BY YEAR
# ============================================================================

print("\n" + "="*80)
print("BESS GROWTH STATISTICS BY YEAR")
print("="*80)

df_with_year = df_hourly.with_columns([
    pl.col('timestamp').dt.year().alias('year')
])

for year in sorted(df_with_year.select('year').unique().to_series().to_list()):
    if year is None:
        continue

    df_year = df_with_year.filter(pl.col('year') == year)

    if len(df_year) == 0:
        continue

    print(f"\n{year}:")
    print("-" * 40)

    mean_dispatch = df_year.select(pl.col('bess_dispatch_MW').mean())[0,0]
    max_discharge = df_year.select(pl.col('bess_discharging_MW').max())[0,0]
    max_charge = df_year.select(pl.col('bess_charging_MW').min())[0,0]
    mean_active = df_year.select(pl.col('bess_active_count').mean())[0,0]
    max_active = df_year.select(pl.col('bess_active_count').max())[0,0]

    print(f"  Mean net dispatch: {mean_dispatch:.2f} MW")
    print(f"  Max discharge: {max_discharge:.2f} MW")
    print(f"  Max charge: {max_charge:.2f} MW")
    print(f"  Mean active BESS: {mean_active:.1f}")
    print(f"  Max active BESS: {max_active}")

# ============================================================================
# 7. SAVE OUTPUT
# ============================================================================

print("\n" + "="*80)
print("SAVING OUTPUT")
print("="*80)

output_file = OUTPUT_DIR / "bess_market_wide_hourly_2019_2025.parquet"
df_hourly.write_parquet(output_file)

print(f"\n✓ Saved: {output_file}")
print(f"  Size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
print(f"  Records: {len(df_hourly):,}")
print(f"  Columns: {len(df_hourly.columns)}")

# Show date range
min_date = df_hourly.select(pl.col('timestamp').min())[0,0]
max_date = df_hourly.select(pl.col('timestamp').max())[0,0]
print(f"\nDate range: {min_date} to {max_date}")

# Show column list
print("\n" + "="*80)
print("OUTPUT COLUMNS")
print("="*80)
for i, col in enumerate(df_hourly.columns, 1):
    print(f"  {i:2}. {col}")

# Show sample
print("\n" + "="*80)
print("SAMPLE DATA (Recent)")
print("="*80)
sample = df_hourly.sort('timestamp', descending=True).head(10)
print(sample.select([
    'timestamp',
    'bess_dispatch_MW',
    'bess_discharging_MW',
    'bess_charging_MW',
    'bess_active_count'
]))

print("\n" + "="*80)
print("✓ MARKET-WIDE BESS AGGREGATION COMPLETE!")
print("="*80)
print(f"Finished: {datetime.now()}")
print(f"\nReady to merge into master dataset!")
print(f"\nKey insights:")
print(f"  - Aggregated {len(bess_resources)} BESS resources")
print(f"  - Growth 2019→2025 captured in active count")
print(f"  - Net dispatch shows charging vs discharging dynamics")
print(f"  - CRITICAL for next model revision!")

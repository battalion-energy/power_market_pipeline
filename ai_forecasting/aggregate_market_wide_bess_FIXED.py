#!/usr/bin/env python3
"""
Aggregate Market-Wide BESS Dispatch from Individual Resource Files - FIXED
==========================================================================

CRITICAL BUG FIX:
The original script used 'local_date' which is a DATE type (daily).
We need to use 'hour_start_local' which has proper hourly timestamps!

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

print("="*80)
print("AGGREGATING MARKET-WIDE BESS DISPATCH - FIXED VERSION")
print("="*80)
print(f"Started: {datetime.now()}")

# Paths
BESS_DISPATCH_DIR = Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/bess_analysis/hourly/dispatch")
OUTPUT_DIR = Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 1. LOAD ALL BESS DISPATCH FILES
# ============================================================================

print("\n" + "="*80)
print("1. LOADING ALL BESS DISPATCH FILES")
print("="*80)

dispatch_files = list(BESS_DISPATCH_DIR.glob("*_dispatch.parquet"))
print(f"Found {len(dispatch_files)} BESS dispatch files")

all_dispatch = []
bess_resources = set()
years = set()

for file in dispatch_files:
    # Extract BESS name and year from filename
    parts = file.stem.split('_')
    year = parts[-2]  # Year before "dispatch"
    bess_name = '_'.join(parts[:-2])  # Everything before year

    bess_resources.add(bess_name)
    years.add(year)

    # Load dispatch data
    try:
        df = pl.read_parquet(file)

        # BUG FIX: Use hour_start_local instead of local_date!
        if 'hour_start_local' in df.columns:
            df = df.with_columns([
                pl.col('hour_start_local').alias('timestamp')
            ])
        else:
            print(f"  ⚠️  No hour_start_local in {file.name}, skipping")
            continue

        # Add BESS name and year
        df = df.with_columns([
            pl.lit(bess_name).alias('bess_name'),
            pl.lit(int(year)).alias('year')
        ])

        # Select relevant columns
        df_clean = df.select([
            'timestamp',
            'bess_name',
            'year',
            pl.col('net_actual_mw_avg').alias('dispatch_mw'),  # Net discharge (positive) or charge (negative)
        ])

        all_dispatch.append(df_clean)

    except Exception as e:
        print(f"  ⚠️  Error loading {file.name}: {e}")
        continue

print(f"\n✓ Loaded {len(all_dispatch)} files")
print(f"  Unique BESS: {len(bess_resources)}")
print(f"  Years: {sorted(years)}")

# ============================================================================
# 2. CONCATENATE ALL DISPATCH DATA
# ============================================================================

print("\n" + "="*80)
print("2. CONCATENATING ALL DISPATCH DATA")
print("="*80)

df_all = pl.concat(all_dispatch, how='diagonal_relaxed')
print(f"Total records: {len(df_all):,}")

# Cast timestamp to nanosecond precision for consistency
df_all = df_all.with_columns([
    pl.col('timestamp').cast(pl.Datetime('ns'))
])

print(f"Unique hourly timestamps: {df_all.select('timestamp').unique().height:,}")

# ============================================================================
# 3. AGGREGATE BY HOUR (MARKET-WIDE TOTALS)
# ============================================================================

print("\n" + "="*80)
print("3. AGGREGATING TO MARKET-WIDE HOURLY TOTALS")
print("="*80)

# Aggregate by timestamp
df_hourly = df_all.group_by('timestamp').agg([
    # Total BESS dispatch (net: positive=discharge, negative=charge)
    pl.col('dispatch_mw').sum().alias('bess_dispatch_MW'),

    # Separate charging and discharging
    pl.col('dispatch_mw').filter(pl.col('dispatch_mw') > 0).sum().alias('bess_discharging_MW'),
    pl.col('dispatch_mw').filter(pl.col('dispatch_mw') < 0).sum().alias('bess_charging_MW'),

    # Count of active BESS resources
    pl.col('bess_name').n_unique().alias('bess_active_count'),

    # Mean and extremes
    pl.col('dispatch_mw').mean().alias('bess_dispatch_mean'),
    pl.col('dispatch_mw').max().alias('bess_dispatch_max'),
    pl.col('dispatch_mw').min().alias('bess_dispatch_min'),
])

# Sort by timestamp
df_hourly = df_hourly.sort('timestamp')

print(f"\nHourly aggregated records: {len(df_hourly):,}")
print(f"Unique timestamps: {df_hourly.select('timestamp').unique().height:,}")

# ============================================================================
# 4. CALCULATE DERIVED FEATURES
# ============================================================================

print("\n" + "="*80)
print("4. CALCULATING DERIVED FEATURES")
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
    # Heavy charging (> 5 GW)
    (pl.col('bess_charging_MW').abs() > 5000).cast(pl.Int8).alias('bess_heavy_charging_flag'),

    # Heavy discharging (> 10 GW)
    (pl.col('bess_discharging_MW') > 10000).cast(pl.Int8).alias('bess_heavy_discharging_flag'),

    # Net charging hour
    (pl.col('bess_dispatch_MW') < 0).cast(pl.Int8).alias('bess_net_charging_flag'),

    # Rapid swing (>2 GW change in 1 hour)
    (pl.col('bess_dispatch_change_1h').abs() > 2000).cast(pl.Int8).alias('bess_rapid_swing_flag'),
])

# ============================================================================
# 5. STATISTICS BY YEAR
# ============================================================================

print("\n" + "="*80)
print("BESS STATISTICS BY YEAR")
print("="*80)

# Add year column for analysis
df_hourly = df_hourly.with_columns([
    pl.col('timestamp').dt.year().alias('year')
])

yearly_stats = df_hourly.group_by('year').agg([
    pl.col('bess_dispatch_MW').mean().alias('mean_dispatch'),
    pl.col('bess_discharging_MW').max().alias('max_discharge'),
    pl.col('bess_active_count').max().alias('max_active_bess'),
]).sort('year')

print("\nYear | Mean Dispatch | Max Discharge | Active BESS")
print("-" * 60)
for row in yearly_stats.iter_rows():
    year, mean_disp, max_disc, active = row
    if year:
        print(f"{year} | {mean_disp:13,.2f} MW | {max_disc:13,.2f} MW | {active:11}")

# Drop year column (not needed in output)
df_hourly = df_hourly.drop('year')

# ============================================================================
# 6. SAVE OUTPUT
# ============================================================================

print("\n" + "="*80)
print("6. SAVING OUTPUT")
print("="*80)

output_file = OUTPUT_DIR / "bess_market_wide_hourly_2019_2025.parquet"
df_hourly.write_parquet(output_file)

print(f"\n✓ Saved: {output_file}")
print(f"  Size: {output_file.stat().st_size / 1024:.1f} KB")
print(f"  Records: {len(df_hourly):,}")
print(f"  Unique hourly timestamps: {df_hourly.select('timestamp').unique().height:,}")
print(f"  Columns: {df_hourly.columns}")

# Show sample
print("\n" + "="*80)
print("SAMPLE DATA (First 24 hours)")
print("="*80)
sample = df_hourly.select([
    'timestamp',
    'bess_dispatch_MW',
    'bess_discharging_MW',
    'bess_charging_MW',
    'bess_active_count'
]).head(24)
print(sample)

# Date range
min_ts = df_hourly.select(pl.col('timestamp').min())[0,0]
max_ts = df_hourly.select(pl.col('timestamp').max())[0,0]
print(f"\nDate range: {min_ts} to {max_ts}")

print("\n" + "="*80)
print("✓ COMPLETE!")
print("="*80)
print(f"Finished: {datetime.now()}")
print(f"\nReady to merge with master dataset!")
print(f"Use: pl.read_parquet('{output_file}')")

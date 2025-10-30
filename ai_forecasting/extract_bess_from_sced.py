#!/usr/bin/env python3
"""
Extract BESS (Battery Energy Storage System) Dispatch from SCED Data
=====================================================================

CRITICAL MISSING FEATURE identified by user!

BESS changes market dynamics:
- Charges during low prices (acts like load) → increases demand → raises prices
- Discharges during high prices (acts like generation) → increases supply → lowers prices
- Growth: 2019 (almost none) → 2025 (GW scale) = FUNDAMENTAL MARKET CHANGE

Extract from SCED 5-minute data:
- ResourceName contains 'BESS', 'BATTERY', or '_ESS_'
- BasePoint: Positive = discharge (generation), Negative = charge (load)
- Aggregate to hourly
- Track capacity growth over time
"""

import polars as pl
from pathlib import Path
from datetime import datetime
import gc

print("="*80)
print("EXTRACTING BESS DISPATCH FROM SCED DATA")
print("="*80)
print(f"Started: {datetime.now()}")

SCED_DIR = Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/rollup_files/SCED_Gen_Resources")
OUTPUT_DIR = Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 1. EXTRACT BESS DATA YEAR BY YEAR
# ============================================================================

print("\n" + "="*80)
print("EXTRACTING BESS BY YEAR")
print("="*80)

years = range(2019, 2026)  # 2019-2025
all_bess = []

for year in years:
    year_file = SCED_DIR / f"{year}.parquet"

    if not year_file.exists():
        print(f"\n{year}: File not found, skipping")
        continue

    print(f"\n{year}:")
    print("-" * 40)

    # Load year data
    df_year = pl.read_parquet(year_file)
    print(f"  Loaded: {len(df_year):,} records")

    # Filter for BESS resources
    # Look for ResourceName containing BESS, BATTERY, or _ESS_
    bess_filter = (
        pl.col('ResourceName').str.contains('BESS', literal=False) |
        pl.col('ResourceName').str.contains('BATTERY', literal=False) |
        pl.col('ResourceName').str.contains('_ESS_', literal=False)
    )

    df_bess = df_year.filter(bess_filter)
    print(f"  BESS records: {len(df_bess):,}")

    if len(df_bess) == 0:
        print(f"  ⚠️  No BESS found in {year}")
        continue

    # Select key columns
    df_bess = df_bess.select([
        'datetime',
        'ResourceName',
        'BasePoint',  # MW dispatch (positive=discharge, negative=charge)
        'TelemeteredNetOutput',  # Actual output
        'HSL',  # High Sustainable Limit (capacity)
        'LSL',  # Low Sustainable Limit
    ])

    # Convert datetime if needed
    if df_bess.schema['datetime'] != pl.Datetime:
        df_bess = df_bess.with_columns([
            pl.col('datetime').cast(pl.Datetime).alias('datetime')
        ])

    # Unique BESS resources
    unique_bess = df_bess.select('ResourceName').unique().height
    print(f"  Unique BESS: {unique_bess}")

    # Aggregate stats
    total_dispatch = df_bess.select(pl.col('BasePoint').sum())[0,0]
    mean_dispatch = df_bess.select(pl.col('BasePoint').mean())[0,0]
    max_discharge = df_bess.select(pl.col('BasePoint').max())[0,0]
    min_charge = df_bess.select(pl.col('BasePoint').min())[0,0]

    print(f"  Total dispatch: {total_dispatch:,.0f} MWh")
    print(f"  Mean dispatch: {mean_dispatch:.2f} MW")
    print(f"  Max discharge: {max_discharge:.2f} MW")
    print(f"  Max charge: {min_charge:.2f} MW (negative)")

    all_bess.append(df_bess)

    # Clean up
    del df_year, df_bess
    gc.collect()

if not all_bess:
    print("\n❌ ERROR: No BESS data found in any year!")
    print("Check if SCED files have BESS resources.")
    exit(1)

# Concatenate all years
print("\n" + "="*80)
print("COMBINING ALL YEARS")
print("="*80)

df_bess_all = pl.concat(all_bess, how='vertical')
print(f"Total BESS records (all years): {len(df_bess_all):,}")

del all_bess
gc.collect()

# ============================================================================
# 2. AGGREGATE TO HOURLY
# ============================================================================

print("\n" + "="*80)
print("AGGREGATING 5-MINUTE DATA TO HOURLY")
print("="*80)

# Round to hour
df_bess_all = df_bess_all.with_columns([
    pl.col('datetime').dt.truncate('1h').alias('timestamp')
])

# Aggregate by hour
# Calculate:
# - Total BESS dispatch (sum of all BasePoints)
# - Charging vs discharging split
# - Number of active BESS resources
# - Total BESS capacity

df_hourly = df_bess_all.group_by('timestamp').agg([
    # Total BESS dispatch (net: positive=discharge, negative=charge)
    pl.col('BasePoint').sum().alias('bess_dispatch_MW'),

    # Separate charging and discharging
    pl.col('BasePoint').filter(pl.col('BasePoint') > 0).sum().alias('bess_discharging_MW'),
    pl.col('BasePoint').filter(pl.col('BasePoint') < 0).sum().alias('bess_charging_MW'),

    # Count of active BESS resources
    pl.col('ResourceName').n_unique().alias('bess_active_count'),

    # Total capacity (sum of HSL for all active BESS)
    pl.col('HSL').sum().alias('bess_total_capacity_MW'),

    # Mean and max dispatch
    pl.col('BasePoint').mean().alias('bess_dispatch_mean'),
    pl.col('BasePoint').max().alias('bess_dispatch_max'),
    pl.col('BasePoint').min().alias('bess_dispatch_min'),
])

# Sort by timestamp
df_hourly = df_hourly.sort('timestamp')

print(f"Hourly records: {len(df_hourly):,}")

# ============================================================================
# 3. CALCULATE DERIVED FEATURES
# ============================================================================

print("\n" + "="*80)
print("CALCULATING DERIVED FEATURES")
print("="*80)

# Hour-over-hour changes
df_hourly = df_hourly.with_columns([
    (pl.col('bess_dispatch_MW') - pl.col('bess_dispatch_MW').shift(1)).alias('bess_dispatch_change_1h'),
    (pl.col('bess_total_capacity_MW') - pl.col('bess_total_capacity_MW').shift(1)).alias('bess_capacity_growth_1h'),
])

# Rolling statistics (24-hour trends)
df_hourly = df_hourly.with_columns([
    pl.col('bess_dispatch_MW').rolling_mean(window_size=24).alias('bess_dispatch_roll_24h_mean'),
    pl.col('bess_discharging_MW').rolling_mean(window_size=24).alias('bess_discharge_roll_24h_mean'),
    pl.col('bess_charging_MW').rolling_mean(window_size=24).alias('bess_charge_roll_24h_mean'),
])

# Flags for significant events
df_hourly = df_hourly.with_columns([
    # Heavy charging (> 500 MW)
    (pl.col('bess_charging_MW') < -500).cast(pl.Int8).alias('bess_heavy_charging_flag'),

    # Heavy discharging (> 500 MW)
    (pl.col('bess_discharging_MW') > 500).cast(pl.Int8).alias('bess_heavy_discharging_flag'),

    # Net charging (negative dispatch)
    (pl.col('bess_dispatch_MW') < 0).cast(pl.Int8).alias('bess_net_charging_flag'),

    # Rapid swing (> 200 MW change in 1 hour)
    (pl.col('bess_dispatch_change_1h').abs() > 200).cast(pl.Int8).alias('bess_rapid_swing_flag'),
])

print("✓ Derived features calculated")

# ============================================================================
# 4. STATISTICS BY YEAR
# ============================================================================

print("\n" + "="*80)
print("BESS GROWTH STATISTICS BY YEAR")
print("="*80)

df_hourly_with_year = df_hourly.with_columns([
    pl.col('timestamp').dt.year().alias('year')
])

for year in range(2019, 2026):
    df_year = df_hourly_with_year.filter(pl.col('year') == year)

    if len(df_year) == 0:
        print(f"\n{year}: No data")
        continue

    print(f"\n{year}:")
    print("-" * 40)

    # Get stats
    mean_capacity = df_year.select(pl.col('bess_total_capacity_MW').mean())[0,0]
    max_capacity = df_year.select(pl.col('bess_total_capacity_MW').max())[0,0]
    mean_active = df_year.select(pl.col('bess_active_count').mean())[0,0]
    max_active = df_year.select(pl.col('bess_active_count').max())[0,0]

    mean_dispatch = df_year.select(pl.col('bess_dispatch_MW').mean())[0,0]
    max_discharge = df_year.select(pl.col('bess_discharging_MW').max())[0,0]
    max_charge = df_year.select(pl.col('bess_charging_MW').min())[0,0]

    print(f"  Mean BESS capacity: {mean_capacity:,.0f} MW")
    print(f"  Max BESS capacity: {max_capacity:,.0f} MW")
    print(f"  Mean active BESS: {mean_active:.1f}")
    print(f"  Max active BESS: {max_active}")
    print(f"  Mean net dispatch: {mean_dispatch:.2f} MW")
    print(f"  Max discharge: {max_discharge:.2f} MW")
    print(f"  Max charge: {max_charge:.2f} MW")

# ============================================================================
# 5. SAVE OUTPUT
# ============================================================================

print("\n" + "="*80)
print("SAVING OUTPUT")
print("="*80)

output_file = OUTPUT_DIR / "bess_dispatch_hourly_2019_2025.parquet"
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
    'bess_active_count',
    'bess_total_capacity_MW'
]))

print("\n" + "="*80)
print("✓ BESS EXTRACTION COMPLETE!")
print("="*80)
print(f"Finished: {datetime.now()}")
print(f"\nReady to merge into master dataset!")
print(f"\nKey insights:")
print(f"  - BESS growth 2019→2025 captured")
print(f"  - Net dispatch tracks charging vs discharging")
print(f"  - Capacity growth over time documented")
print(f"  - CRITICAL for next model revision!")

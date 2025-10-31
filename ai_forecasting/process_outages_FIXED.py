#!/usr/bin/env python3
"""
Process Generator Outage Capacity Data - FIXED
==============================================

Correctly processes raw outage files with proper timestamp parsing.
"""

import polars as pl
from pathlib import Path
from datetime import datetime

print("="*80)
print("PROCESSING GENERATOR OUTAGE DATA - FIXED")
print("="*80)
print(f"Started: {datetime.now()}")

RAW_DIR = Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_clean_batch_dataset/parquet/Hourly Resource Outage Capacity")
OUTPUT_DIR = Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("\n" + "="*80)
print("1. LOADING AND PROCESSING RAW OUTAGE FILES")
print("="*80)

raw_files = sorted(RAW_DIR.glob("*.parquet"))
print(f"Found {len(raw_files)} files")

all_outages = []
for file in raw_files:
    year = file.stem.split('_')[-1]
    df = pl.read_parquet(file)
    
    # Parse timestamps from Date + HourEnding (same fix as load data)
    df = df.with_columns([
        pl.col('Date').str.strptime(pl.Date, '%m/%d/%Y').alias('date'),
        pl.col('HourEnding').cast(pl.Int32).alias('hour_ending'),  # Already an integer
    ])
    
    # Create proper timestamp
    df = df.with_columns([
        pl.when(pl.col('hour_ending') == 24)
        .then(pl.col('date').cast(pl.Datetime) + pl.duration(days=1))
        .otherwise(pl.col('date').cast(pl.Datetime) + pl.duration(hours=pl.col('hour_ending')))
        .alias('timestamp')
    ])
    
    # Select and rename columns (cast to Float64 for consistency)
    df_clean = df.select([
        'timestamp',
        pl.col('TotalResourceMW').cast(pl.Float64).alias('outage_total_MW'),
        pl.col('TotalIRRMW').cast(pl.Float64).alias('outage_renewable_MW'),
        pl.col('TotalNewEquipResourceMW').cast(pl.Float64).alias('outage_new_equip_MW'),
    ])

    # Fill nulls with 0
    df_clean = df_clean.with_columns([
        pl.col('outage_total_MW').fill_null(0),
        pl.col('outage_renewable_MW').fill_null(0),
        pl.col('outage_new_equip_MW').fill_null(0),
    ])
    
    all_outages.append(df_clean)
    print(f"  ✓ {year}: {len(df_clean):,} records")

df_outages = pl.concat(all_outages).sort('timestamp')
print(f"\nTotal outage records: {len(df_outages):,}")

# ============================================================================
# 2. CALCULATE DERIVED FEATURES
# ============================================================================
print("\n" + "="*80)
print("2. CALCULATING DERIVED FEATURES")
print("="*80)

df_outages = df_outages.with_columns([
    (pl.col('outage_total_MW') - pl.col('outage_renewable_MW')).alias('outage_thermal_MW'),
    (pl.col('outage_total_MW') - pl.col('outage_total_MW').shift(1)).alias('outage_change_1h'),
    pl.col('outage_total_MW').rolling_mean(window_size=3).alias('outage_roll_3h_mean'),
    pl.col('outage_total_MW').rolling_max(window_size=3).alias('outage_roll_3h_max'),
    pl.col('outage_total_MW').rolling_mean(window_size=24).alias('outage_roll_24h_mean'),
    pl.col('outage_total_MW').rolling_max(window_size=24).alias('outage_roll_24h_max'),
])

df_outages = df_outages.with_columns([
    (pl.col('outage_total_MW') > 15000).cast(pl.Int8).alias('high_outage_flag'),
    (pl.col('outage_total_MW') > 20000).cast(pl.Int8).alias('critical_outage_flag'),
    (pl.col('outage_thermal_MW') > 10000).cast(pl.Int8).alias('large_thermal_outage_flag'),
    (pl.col('outage_change_1h') > 2000).cast(pl.Int8).alias('sudden_outage_flag'),
])

# ============================================================================
# 3. STATISTICS
# ============================================================================
print("\n" + "="*80)
print("OUTAGE STATISTICS")
print("="*80)

stats_by_year = df_outages.group_by(
    pl.col('timestamp').dt.year().alias('year')
).agg([
    pl.col('outage_total_MW').mean().alias('mean'),
    pl.col('outage_total_MW').max().alias('max'),
    pl.len().alias('hours'),
]).sort('year')

print("\nOutage MW by year:")
print(stats_by_year)

# ============================================================================
# 4. SAVE OUTPUT
# ============================================================================
print("\n" + "="*80)
print("4. SAVING OUTPUT")
print("="*80)

output_file = OUTPUT_DIR / "generator_outages_2018_2025_FIXED.parquet"
df_outages.write_parquet(output_file)

print(f"\n✓ Saved: {output_file}")
print(f"  Size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
print(f"  Records: {len(df_outages):,}")
print(f"  Date range: {df_outages['timestamp'].min()} to {df_outages['timestamp'].max()}")

print("\n" + "="*80)
print("✓ COMPLETE - OUTAGE DATA FIXED!")
print("="*80)
print(f"Finished: {datetime.now()}")

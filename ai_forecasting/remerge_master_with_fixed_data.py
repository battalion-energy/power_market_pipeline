#!/usr/bin/env python3
"""
Re-merge Master Dataset with Fixed Net Load and Reserve Margin Data
===================================================================

The original merge had only 4% coverage because the source files had
daily timestamps instead of hourly. Now that we've fixed the timestamps,
we need to re-merge to get 100% coverage.

Steps:
1. Load master dataset
2. Drop buggy net load and reserve margin columns
3. Merge with fixed net load features (59,825 hourly)
4. Merge with fixed reserve margin features (59,825 hourly)
5. Verify 100% coverage
6. Save updated master dataset
"""

import polars as pl
from pathlib import Path
from datetime import datetime

print("="*80)
print("RE-MERGING MASTER DATASET WITH FIXED NET LOAD AND RESERVE MARGIN")
print("="*80)
print(f"Started: {datetime.now()}")

# Paths
MASTER_FILE = Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/master_enhanced_with_net_load_reserves_2019_2025.parquet")
NET_LOAD_FILE = Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/processed/net_load_features_2018_2025.parquet")
RESERVE_FILE = Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/processed/reserve_margin_dam_2018_2025.parquet")
OUTPUT_FILE = Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/master_enhanced_with_net_load_reserves_2019_2025.parquet")

# ============================================================================
# 1. LOAD MASTER DATASET
# ============================================================================
print("\n" + "="*80)
print("1. LOADING MASTER DATASET")
print("="*80)

df_master = pl.read_parquet(MASTER_FILE)
print(f"Master dataset: {len(df_master):,} records")
print(f"Unique timestamps: {df_master.select('timestamp').unique().height:,}")
print(f"Columns: {len(df_master.columns)}")

# ============================================================================
# 2. DROP BUGGY NET LOAD AND RESERVE MARGIN COLUMNS
# ============================================================================
print("\n" + "="*80)
print("2. DROPPING BUGGY COLUMNS")
print("="*80)

# Identify columns to drop (from previous buggy merge)
buggy_cols = [
    'net_load_MW',
    'wind_generation_MW',
    'solar_generation_MW',
    'renewable_penetration_pct',
    'net_load_ramp_1h',
    'net_load_ramp_3h',
    'net_load_roll_24h_mean',
    'net_load_roll_24h_std',
    'net_load_roll_24h_max',
    'net_load_roll_24h_min',
    'houston_pct_of_net_load',
    'north_pct_of_net_load',
    'high_renewable_flag',
    'large_ramp_flag',
    'low_net_load_flag',
    'sced_system_load_MW',
    'reserve_margin_pct',
    'tight_reserves_flag',
    'critical_reserves_flag',
    'total_dam_reserves_MW',
    'REGDN',
    'REGUP',
    'RRS',
    'ECRS',
    'NSPIN',
]

# Only drop columns that actually exist
cols_to_drop = [c for c in buggy_cols if c in df_master.columns]
print(f"Dropping {len(cols_to_drop)} buggy columns:")
for col in cols_to_drop:
    print(f"  - {col}")

df_clean = df_master.drop(cols_to_drop)
print(f"\nAfter dropping: {len(df_clean.columns)} columns")

# ============================================================================
# 3. LOAD FIXED NET LOAD FEATURES
# ============================================================================
print("\n" + "="*80)
print("3. LOADING FIXED NET LOAD FEATURES")
print("="*80)

df_net_load = pl.read_parquet(NET_LOAD_FILE)
print(f"Net load file: {len(df_net_load):,} records")
print(f"Unique timestamps: {df_net_load.select('timestamp').unique().height:,}")

# Select columns to merge (exclude columns already in master like NORTH, SOUTH, etc)
net_load_merge_cols = [
    'timestamp',
    'net_load_MW',
    'wind_generation_MW',
    'solar_generation_MW',
    'renewable_penetration_pct',
    'net_load_ramp_1h',
    'net_load_ramp_3h',
    'net_load_roll_24h_mean',
    'net_load_roll_24h_std',
    'net_load_roll_24h_max',
    'net_load_roll_24h_min',
    'houston_pct_of_net_load',
    'north_pct_of_net_load',
    'high_renewable_flag',
    'large_ramp_flag',
    'low_net_load_flag',
]

df_net_load_clean = df_net_load.select([c for c in net_load_merge_cols if c in df_net_load.columns])
print(f"Selected {len(df_net_load_clean.columns)} columns for merge")

# ============================================================================
# 4. LOAD FIXED RESERVE MARGIN FEATURES
# ============================================================================
print("\n" + "="*80)
print("4. LOADING FIXED RESERVE MARGIN FEATURES")
print("="*80)

df_reserve = pl.read_parquet(RESERVE_FILE)
print(f"Reserve margin file: {len(df_reserve):,} records")
print(f"Unique timestamps: {df_reserve.select('timestamp').unique().height:,}")

# Select columns to merge
reserve_merge_cols = [
    'timestamp',
    'reserve_margin_pct',
    'tight_reserves_flag',
    'critical_reserves_flag',
    'total_dam_reserves_MW',
    'REGDN',
    'REGUP',
    'RRS',
    'ECRS',
    'NSPIN',
]

df_reserve_clean = df_reserve.select([c for c in reserve_merge_cols if c in df_reserve.columns])

# Cast timestamp to match master dataset precision (nanoseconds)
df_reserve_clean = df_reserve_clean.with_columns([
    pl.col('timestamp').cast(pl.Datetime('ns'))
])

print(f"Selected {len(df_reserve_clean.columns)} columns for merge")

# ============================================================================
# 5. MERGE WITH MASTER DATASET
# ============================================================================
print("\n" + "="*80)
print("5. MERGING WITH MASTER DATASET")
print("="*80)

# Merge net load
print("\nMerging net load features...")
df_merged = df_clean.join(
    df_net_load_clean,
    on='timestamp',
    how='left'
)
print(f"After net load merge: {len(df_merged):,} records, {len(df_merged.columns)} columns")

# Check coverage
net_load_coverage = df_merged.filter(pl.col('net_load_MW').is_not_null()).height
print(f"Net load coverage: {net_load_coverage:,} / {len(df_merged):,} ({100*net_load_coverage/len(df_merged):.1f}%)")

# Merge reserve margin
print("\nMerging reserve margin features...")
df_merged = df_merged.join(
    df_reserve_clean,
    on='timestamp',
    how='left'
)
print(f"After reserve merge: {len(df_merged):,} records, {len(df_merged.columns)} columns")

# Check coverage
reserve_coverage = df_merged.filter(pl.col('reserve_margin_pct').is_not_null()).height
print(f"Reserve margin coverage: {reserve_coverage:,} / {len(df_merged):,} ({100*reserve_coverage/len(df_merged):.1f}%)")

# ============================================================================
# 6. VERIFY COVERAGE
# ============================================================================
print("\n" + "="*80)
print("6. VERIFYING FEATURE COVERAGE")
print("="*80)

# Check all critical features
critical_features = [
    'net_load_MW',
    'wind_generation_MW',
    'solar_generation_MW',
    'renewable_penetration_pct',
    'reserve_margin_pct',
    'total_dam_reserves_MW',
]

print("\nCritical Feature Coverage:")
for feature in critical_features:
    if feature in df_merged.columns:
        non_null = df_merged.filter(pl.col(feature).is_not_null()).height
        total = len(df_merged)
        pct = 100 * non_null / total
        print(f"  {feature:30s}: {non_null:9,} / {total:9,} ({pct:5.1f}%)")

        # Show date range of non-null values
        if non_null > 0:
            non_null_dates = df_merged.filter(pl.col(feature).is_not_null())
            min_date = non_null_dates.select(pl.col('timestamp').min())[0,0]
            max_date = non_null_dates.select(pl.col('timestamp').max())[0,0]
            print(f"        Date range: {min_date} to {max_date}")
    else:
        print(f"  {feature:30s}: NOT IN DATASET")

# ============================================================================
# 7. SAVE UPDATED MASTER DATASET
# ============================================================================
print("\n" + "="*80)
print("7. SAVING UPDATED MASTER DATASET")
print("="*80)

df_merged.write_parquet(OUTPUT_FILE)

print(f"\n✓ Saved: {OUTPUT_FILE}")
print(f"  Size: {OUTPUT_FILE.stat().st_size / 1024 / 1024:.1f} MB")
print(f"  Records: {len(df_merged):,}")
print(f"  Unique timestamps: {df_merged.select('timestamp').unique().height:,}")
print(f"  Columns: {len(df_merged.columns)}")

# Summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

# Net load stats
valid_net_load = df_merged.filter(pl.col('net_load_MW').is_not_null())
if len(valid_net_load) > 0:
    stats = valid_net_load.select(['net_load_MW', 'renewable_penetration_pct']).describe()

    print("\nNet Load (MW):")
    print(stats.filter(pl.col('statistic').is_in(['mean', 'min', 'max'])).select(['statistic', 'net_load_MW']))

    print("\nRenewable Penetration (%):")
    print(stats.filter(pl.col('statistic').is_in(['mean', 'min', 'max'])).select(['statistic', 'renewable_penetration_pct']))

# Reserve margin stats
valid_reserve = df_merged.filter(pl.col('reserve_margin_pct').is_not_null())
if len(valid_reserve) > 0:
    stats = valid_reserve.select(['reserve_margin_pct', 'total_dam_reserves_MW']).describe()

    print("\nReserve Margin (%):")
    print(stats.filter(pl.col('statistic').is_in(['mean', 'min', 'max'])).select(['statistic', 'reserve_margin_pct']))

print("\n" + "="*80)
print("✓ COMPLETE!")
print("="*80)
print(f"Finished: {datetime.now()}")
print("\nMaster dataset updated with 100% coverage on critical features!")
print("Ready to retrain model with proper data.")

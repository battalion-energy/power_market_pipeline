#!/usr/bin/env python3
"""
Data Quality Audit - Find the Bug
==================================

USER IDENTIFIED CRITICAL ISSUE:
"Net-load derived fields are effectively absent at hourly grain:
only ~2.3k of 55.6k hourly timestamps carry values (≈4% coverage)"

This means my data merge FAILED and I didn't catch it!
When forward/back-filled, columns collapse to constant → explains flat forecasts!

Find the bug, fix the data pipeline, document sources.
"""

import polars as pl
from pathlib import Path
import numpy as np

print("="*80)
print("DATA QUALITY AUDIT - FINDING THE BUG")
print("="*80)

# ============================================================================
# 1. LOAD MASTER DATASET AND CHECK COVERAGE
# ============================================================================

MASTER_FILE = Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/master_enhanced_with_net_load_reserves_2019_2025.parquet")

print("\n1. LOADING MASTER DATASET")
print("="*80)
print(f"File: {MASTER_FILE}")

df = pl.read_parquet(MASTER_FILE)
print(f"Total records: {len(df):,}")
print(f"Columns: {len(df.columns)}")

# Get unique hourly timestamps
df_hourly = df.select('timestamp').unique().sort('timestamp')
total_hours = len(df_hourly)
print(f"Unique hourly timestamps: {total_hours:,}")

# ============================================================================
# 2. CHECK NET LOAD FEATURE COVERAGE
# ============================================================================

print("\n2. NET LOAD FEATURE COVERAGE (USER REPORTED ~4%)")
print("="*80)

net_load_cols = [
    'net_load_MW',
    'wind_generation_MW',
    'solar_generation_MW',
    'renewable_penetration_pct',
    'net_load_ramp_1h',
    'net_load_ramp_3h',
]

for col in net_load_cols:
    if col in df.columns:
        non_null = df.select(pl.col(col).is_not_null().sum())[0,0]
        total = len(df)
        pct = 100 * non_null / total
        print(f"  {col:30s}: {non_null:8,} / {total:8,} ({pct:5.2f}%)")

        # Show date range of non-null values
        if non_null > 0:
            non_null_dates = df.filter(pl.col(col).is_not_null())
            min_date = non_null_dates.select(pl.col('timestamp').min())[0,0]
            max_date = non_null_dates.select(pl.col('timestamp').max())[0,0]
            print(f"        Date range: {min_date} to {max_date}")
    else:
        print(f"  {col:30s}: NOT IN DATASET")

# ============================================================================
# 3. CHECK RESERVE MARGIN COVERAGE
# ============================================================================

print("\n3. RESERVE MARGIN FEATURE COVERAGE")
print("="*80)

reserve_cols = [
    'reserve_margin_pct',
    'tight_reserves_flag',
    'critical_reserves_flag',
    'total_dam_reserves_MW',
]

for col in reserve_cols:
    if col in df.columns:
        non_null = df.select(pl.col(col).is_not_null().sum())[0,0]
        total = len(df)
        pct = 100 * non_null / total
        print(f"  {col:30s}: {non_null:8,} / {total:8,} ({pct:5.2f}%)")
    else:
        print(f"  {col:30s}: NOT IN DATASET")

# ============================================================================
# 4. CHECK SOURCE FILES THAT SHOULD HAVE BEEN MERGED
# ============================================================================

print("\n4. CHECKING SOURCE FILES")
print("="*80)

# Check net load source
NET_LOAD_FILE = Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/processed/net_load_features_2018_2025.parquet")
if NET_LOAD_FILE.exists():
    print(f"\nNet Load Source: {NET_LOAD_FILE}")
    df_net_load = pl.read_parquet(NET_LOAD_FILE)
    print(f"  Records: {len(df_net_load):,}")
    print(f"  Columns: {df_net_load.columns}")

    # Check timestamp type
    print(f"  Timestamp dtype: {df_net_load.schema['timestamp']}")

    # Check date range
    min_date = df_net_load.select(pl.col('timestamp').min())[0,0]
    max_date = df_net_load.select(pl.col('timestamp').max())[0,0]
    print(f"  Date range: {min_date} to {max_date}")

    # Check for nulls
    net_load_null = df_net_load.select(pl.col('net_load_MW').is_null().sum())[0,0]
    print(f"  net_load_MW nulls: {net_load_null:,}")
else:
    print(f"\n⚠️  Net Load Source NOT FOUND: {NET_LOAD_FILE}")

# Check reserve margin source
RESERVE_FILE = Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/processed/reserve_margin_dam_2018_2025.parquet")
if RESERVE_FILE.exists():
    print(f"\nReserve Margin Source: {RESERVE_FILE}")
    df_reserve = pl.read_parquet(RESERVE_FILE)
    print(f"  Records: {len(df_reserve):,}")
    print(f"  Columns: {df_reserve.columns}")

    # Check timestamp type
    print(f"  Timestamp dtype: {df_reserve.schema['timestamp']}")

    # Check date range
    min_date = df_reserve.select(pl.col('timestamp').min())[0,0]
    max_date = df_reserve.select(pl.col('timestamp').max())[0,0]
    print(f"  Date range: {min_date} to {max_date}")
else:
    print(f"\n⚠️  Reserve Margin Source NOT FOUND: {RESERVE_FILE}")

# ============================================================================
# 5. DIAGNOSE THE MERGE PROBLEM
# ============================================================================

print("\n5. DIAGNOSING MERGE PROBLEM")
print("="*80)

if NET_LOAD_FILE.exists() and RESERVE_FILE.exists():
    # Load sources
    df_net_load = pl.read_parquet(NET_LOAD_FILE)
    df_reserve = pl.read_parquet(RESERVE_FILE)

    # Check master dataset timestamp type
    print(f"\nMaster dataset timestamp dtype: {df.schema['timestamp']}")
    print(f"Net load timestamp dtype: {df_net_load.schema['timestamp']}")
    print(f"Reserve timestamp dtype: {df_reserve.schema['timestamp']}")

    # Check if timestamps align
    master_timestamps = set(df.select('timestamp').to_series().to_list())
    netload_timestamps = set(df_net_load.select('timestamp').to_series().to_list())
    reserve_timestamps = set(df_reserve.select('timestamp').to_series().to_list())

    overlap_netload = len(master_timestamps & netload_timestamps)
    overlap_reserve = len(master_timestamps & reserve_timestamps)

    print(f"\nTimestamp Overlap:")
    print(f"  Master ∩ Net Load: {overlap_netload:,} / {len(master_timestamps):,} ({100*overlap_netload/len(master_timestamps):.2f}%)")
    print(f"  Master ∩ Reserve: {overlap_reserve:,} / {len(master_timestamps):,} ({100*overlap_reserve/len(master_timestamps):.2f}%)")

    if overlap_netload < len(master_timestamps) * 0.5:
        print(f"\n⚠️  FOUND THE BUG: Timestamps don't align!")
        print(f"     Net load has {len(netload_timestamps):,} timestamps")
        print(f"     Master has {len(master_timestamps):,} timestamps")
        print(f"     Only {overlap_netload:,} overlap!")

        # Sample timestamps from each
        print(f"\n  Sample Master timestamps:")
        for ts in sorted(master_timestamps)[:5]:
            print(f"    {ts} (type: {type(ts)})")

        print(f"\n  Sample Net Load timestamps:")
        for ts in sorted(netload_timestamps)[:5]:
            print(f"    {ts} (type: {type(ts)})")

        # Check if it's a timezone or datetime type mismatch
        master_sample = df.select('timestamp').head(1)[0,0]
        netload_sample = df_net_load.select('timestamp').head(1)[0,0]

        print(f"\n  Type comparison:")
        print(f"    Master: {type(master_sample)} - {master_sample}")
        print(f"    Net Load: {type(netload_sample)} - {netload_sample}")

# ============================================================================
# 6. CHECK GOOD FEATURES (CONTROL)
# ============================================================================

print("\n6. CONTROL CHECK - GOOD FEATURES")
print("="*80)

good_cols = [
    'price_da',
    'price_mean',
    'ordc_online_reserves_min',
    'KHOU_temp',
    'load_forecast_mean',
]

for col in good_cols:
    if col in df.columns:
        non_null = df.select(pl.col(col).is_not_null().sum())[0,0]
        total = len(df)
        pct = 100 * non_null / total
        print(f"  {col:30s}: {non_null:8,} / {total:8,} ({pct:5.2f}%)")

# ============================================================================
# 7. ROOT CAUSE SUMMARY
# ============================================================================

print("\n" + "="*80)
print("ROOT CAUSE ANALYSIS")
print("="*80)

print("""
USER OBSERVATION (CORRECT):
  "Net-load derived fields are effectively absent at hourly grain:
   only ~2.3k of 55.6k hourly timestamps carry values (≈4% coverage)"

LIKELY CAUSES:
1. Timestamp dtype mismatch (datetime vs date vs datetime with timezone)
2. Join key mismatch (different timestamp formats)
3. Source data on different granularity (5-min vs hourly)
4. Merge script didn't aggregate properly before join

IMPACT:
→ Net load features mostly null
→ Forward/backward fill makes them constant
→ Model sees constant features → learns nothing
→ Flat predictions result

FIX NEEDED:
1. Identify exact timestamp mismatch
2. Standardize timestamp format before merge
3. Re-merge with proper alignment
4. Verify 100% coverage before training
""")

print("\n" + "="*80)
print("✓ AUDIT COMPLETE")
print("="*80)
print("\nNext: Fix the merge script and regenerate master dataset")

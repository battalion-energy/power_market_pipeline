#!/usr/bin/env python3
"""
Create Enhanced Master Dataset with Net Load and Reserve Margin Features
========================================================================

Merges:
1. Existing master dataset (prices, ORDC, load forecasts, weather)
2. Net load features (system_load - wind - solar)
3. Reserve margin features (DAM reserves / load)

This will be the final dataset for training the 48-hour DA+RT forecast model.
"""

import polars as pl
from pathlib import Path
from datetime import datetime

print("="*80)
print("CREATING ENHANCED MASTER DATASET")
print("="*80)
print(f"Started: {datetime.now()}")

# Paths
CURRENT_MASTER = Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/master_features_enhanced_with_ordc_load_2019_2025.parquet")
NET_LOAD = Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/processed/net_load_features_2018_2025.parquet")
RESERVE_MARGIN = Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/processed/reserve_margin_dam_2018_2025.parquet")
OUTPUT_DIR = Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data")

# ============================================================================
# 1. LOAD ALL DATASETS
# ============================================================================
print("\n" + "="*80)
print("1. LOADING DATASETS")
print("="*80)

print("Loading current master dataset...")
df_master = pl.read_parquet(CURRENT_MASTER)
print(f"  ✓ Master: {len(df_master):,} records, {len(df_master.columns)} columns")

print("Loading net load features...")
df_net_load = pl.read_parquet(NET_LOAD)
print(f"  ✓ Net load: {len(df_net_load):,} records, {len(df_net_load.columns)} columns")

print("Loading reserve margin...")
df_reserve = pl.read_parquet(RESERVE_MARGIN)
print(f"  ✓ Reserve margin: {len(df_reserve):,} records, {len(df_reserve.columns)} columns")

# ============================================================================
# 2. PREPARE FOR MERGE
# ============================================================================
print("\n" + "="*80)
print("2. PREPARING FOR MERGE")
print("="*80)

# Select columns to merge from net load (avoid duplicates)
net_load_cols = [
    'timestamp',
    'wind_generation_MW',
    'solar_generation_MW',
    'net_load_MW',
    'renewable_penetration_pct',
    'net_load_ramp_1h',
    'net_load_ramp_3h',
    'net_load_roll_24h_mean',
    'net_load_roll_24h_std',
    'high_renewable_flag',
    'large_ramp_flag',
]

df_net_load_clean = df_net_load.select([c for c in net_load_cols if c in df_net_load.columns])
print(f"  Net load columns to merge: {len(df_net_load_clean.columns) - 1}")

# Select columns from reserve margin
reserve_cols = [
    'timestamp',
    'total_dam_reserves_MW',
    'reserve_margin_pct',
    'tight_reserves_flag',
    'critical_reserves_flag',
    'REGDN',
    'REGUP',
    'RRS',
    'ECRS',
    'NSPIN',
]

df_reserve_clean = df_reserve.select([c for c in reserve_cols if c in df_reserve.columns])
print(f"  Reserve margin columns to merge: {len(df_reserve_clean.columns) - 1}")

# ============================================================================
# 3. MERGE ALL DATASETS
# ============================================================================
print("\n" + "="*80)
print("3. MERGING DATASETS")
print("="*80)

# Merge net load
print("Merging net load features...")
df_enhanced = df_master.join(
    df_net_load_clean,
    on='timestamp',
    how='left'
)
print(f"  ✓ After net load merge: {len(df_enhanced):,} records, {len(df_enhanced.columns)} columns")

# Merge reserve margin
print("Merging reserve margin features...")
df_enhanced = df_enhanced.join(
    df_reserve_clean,
    on='timestamp',
    how='left'
)
print(f"  ✓ After reserve margin merge: {len(df_enhanced):,} records, {len(df_enhanced.columns)} columns")

# ============================================================================
# 4. DATA QUALITY CHECK
# ============================================================================
print("\n" + "="*80)
print("4. DATA QUALITY CHECK")
print("="*80)

# Check for nulls in critical new columns
critical_cols = ['net_load_MW', 'wind_generation_MW', 'reserve_margin_pct']
for col in critical_cols:
    if col in df_enhanced.columns:
        null_count = df_enhanced.select(pl.col(col).is_null().sum())[col][0]
        total = len(df_enhanced)
        print(f"  {col:30s}: {total - null_count:6,}/{total:6,} non-null ({100*(total-null_count)/total:5.1f}%)")

# Show date range
min_date = df_enhanced.select(pl.col('timestamp').min())[0,0]
max_date = df_enhanced.select(pl.col('timestamp').max())[0,0]
print(f"\nDate range: {min_date} to {max_date}")

# Count complete records (all critical features non-null)
complete_mask = pl.lit(True)
for col in ['price_da', 'price_mean', 'net_load_MW']:
    if col in df_enhanced.columns:
        complete_mask = complete_mask & pl.col(col).is_not_null()

complete_records = df_enhanced.filter(complete_mask).height
print(f"Complete records (all critical features): {complete_records:,} ({100*complete_records/len(df_enhanced):.1f}%)")

# ============================================================================
# 5. SAVE ENHANCED DATASET
# ============================================================================
print("\n" + "="*80)
print("5. SAVING ENHANCED DATASET")
print("="*80)

output_file = OUTPUT_DIR / "master_enhanced_with_net_load_reserves_2019_2025.parquet"
df_enhanced.write_parquet(output_file)

print(f"\n✓ Saved: {output_file}")
print(f"  Size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
print(f"  Records: {len(df_enhanced):,}")
print(f"  Columns: {len(df_enhanced.columns)}")

# ============================================================================
# 6. SELECT 15 DEMO DATES FOR GAMBIT_ESS1
# ============================================================================
print("\n" + "="*80)
print("6. SELECTING 15 DEMO DATES FOR GAMBIT_ESS1")
print("="*80)

# Strategy: Select dates with diverse conditions
# - Different seasons
# - Mix of high/low prices
# - Different renewable penetration levels
# - Include some scarcity events
# - Spread across 2023-2024 (most recent complete years)

demo_dates = [
    # Q1 2023 - Winter
    "2023-01-15",  # Mid-winter
    "2023-02-20",  # Late winter

    # Q2 2023 - Spring
    "2023-04-10",  # Spring
    "2023-05-15",  # Late spring

    # Q3 2023 - Summer (high load, high prices)
    "2023-07-20",  # Mid-summer
    "2023-08-15",  # Peak summer

    # Q4 2023 - Fall
    "2023-10-01",  # Early fall
    "2023-11-15",  # Late fall

    # Q1 2024 - Winter
    "2024-01-01",  # New year
    "2024-02-20",  # GAMBIT_ESS1 original demo date

    # Q2 2024 - Spring
    "2024-03-15",  # Early spring
    "2024-04-10",  # Mid spring

    # Q3 2024 - Summer
    "2024-06-15",  # Early summer
    "2024-08-01",  # Peak summer

    # Q4 2024 - Fall
    "2024-10-01",  # Fall
]

print(f"Selected {len(demo_dates)} demo dates:")
for i, date in enumerate(demo_dates, 1):
    # Check if data exists for this date
    ts = pl.datetime(int(date[:4]), int(date[5:7]), int(date[8:10]))
    data_exists = df_enhanced.filter(
        (pl.col('timestamp') >= ts) &
        (pl.col('timestamp') < ts + pl.duration(days=2))
    ).height > 0

    status = "✓" if data_exists else "✗"
    print(f"  {i:2}. {date}  {status}")

# Save demo dates list
demo_dates_file = OUTPUT_DIR / "demo_dates_gambit_ess1.txt"
with open(demo_dates_file, 'w') as f:
    for date in demo_dates:
        f.write(f"{date}\n")

print(f"\n✓ Saved demo dates: {demo_dates_file}")

print("\n" + "="*80)
print("✓ COMPLETE!")
print("="*80)
print(f"Finished: {datetime.now()}")
print(f"\nEnhanced dataset ready for training!")
print(f"Next step: Train model with: {output_file}")

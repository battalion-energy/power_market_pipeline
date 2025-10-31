#!/usr/bin/env python3
"""
Merge BESS and Outages into Master Dataset
===========================================

Now that we have:
- Fixed net load and reserve margin (100% coverage)
- Fixed BESS data (40,398 hourly timestamps)
- Fixed outages data (57,330 hourly timestamps)

Merge everything into the master dataset for training.
"""

import polars as pl
from pathlib import Path
from datetime import datetime

print("="*80)
print("MERGING BESS AND OUTAGES INTO MASTER DATASET")
print("="*80)
print(f"Started: {datetime.now()}")

# Paths
MASTER_FILE = Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/master_enhanced_with_net_load_reserves_2019_2025.parquet")
BESS_FILE = Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/processed/bess_market_wide_hourly_2019_2025.parquet")
OUTAGE_FILE = Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/processed/generator_outages_2018_2025.parquet")

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
# 2. LOAD BESS DATA
# ============================================================================
print("\n" + "="*80)
print("2. LOADING BESS DATA")
print("="*80)

df_bess = pl.read_parquet(BESS_FILE)
print(f"BESS data: {len(df_bess):,} records")
print(f"Unique timestamps: {df_bess.select('timestamp').unique().height:,}")
print(f"Columns to merge: {df_bess.columns}")

# ============================================================================
# 3. LOAD OUTAGES DATA
# ============================================================================
print("\n" + "="*80)
print("3. LOADING OUTAGES DATA")
print("="*80)

df_outage = pl.read_parquet(OUTAGE_FILE)
print(f"Outages data: {len(df_outage):,} records")
print(f"Unique timestamps: {df_outage.select('timestamp').unique().height:,}")
print(f"Columns to merge: {df_outage.columns}")

# ============================================================================
# 4. MERGE BESS INTO MASTER
# ============================================================================
print("\n" + "="*80)
print("4. MERGING BESS DATA")
print("="*80)

df_merged = df_master.join(
    df_bess,
    on='timestamp',
    how='left'
)

print(f"After BESS merge: {len(df_merged):,} records, {len(df_merged.columns)} columns")

# Check coverage
bess_coverage = df_merged.filter(pl.col('bess_dispatch_MW').is_not_null()).height
total = len(df_merged)
print(f"BESS coverage: {bess_coverage:,} / {total:,} ({100*bess_coverage/total:.1f}%)")

# ============================================================================
# 5. MERGE OUTAGES INTO MASTER
# ============================================================================
print("\n" + "="*80)
print("5. MERGING OUTAGES DATA")
print("="*80)

df_merged = df_merged.join(
    df_outage,
    on='timestamp',
    how='left'
)

print(f"After outages merge: {len(df_merged):,} records, {len(df_merged.columns)} columns")

# Check coverage
outage_coverage = df_merged.filter(pl.col('outage_total_MW').is_not_null()).height
print(f"Outages coverage: {outage_coverage:,} / {total:,} ({100*outage_coverage/total:.1f}%)")

# ============================================================================
# 6. VERIFY ALL CRITICAL FEATURES
# ============================================================================
print("\n" + "="*80)
print("6. VERIFYING ALL CRITICAL FEATURES")
print("="*80)

critical_features = [
    'net_load_MW',
    'wind_generation_MW',
    'solar_generation_MW',
    'renewable_penetration_pct',
    'reserve_margin_pct',
    'total_dam_reserves_MW',
    'bess_dispatch_MW',
    'bess_active_count',
    'outage_total_MW',
    'outage_thermal_MW',
]

print("\nCritical Feature Coverage:")
for feature in critical_features:
    if feature in df_merged.columns:
        non_null = df_merged.filter(pl.col(feature).is_not_null()).height
        total = len(df_merged)
        pct = 100 * non_null / total
        print(f"  {feature:30s}: {non_null:9,} / {total:9,} ({pct:5.1f}%)")
    else:
        print(f"  {feature:30s}: NOT IN DATASET")

# ============================================================================
# 7. SAVE UPDATED MASTER DATASET
# ============================================================================
print("\n" + "="*80)
print("7. SAVING UPDATED MASTER DATASET")
print("="*80)

output_file = MASTER_FILE
df_merged.write_parquet(output_file)

print(f"\n✓ Saved: {output_file}")
print(f"  Size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
print(f"  Records: {len(df_merged):,}")
print(f"  Unique timestamps: {df_merged.select('timestamp').unique().height:,}")
print(f"  Columns: {len(df_merged.columns)}")

print("\n" + "="*80)
print("✓ COMPLETE!")
print("="*80)
print(f"Finished: {datetime.now()}")
print("\nMaster dataset updated with BESS and outages!")
print("All critical features integrated. Ready for model training.")

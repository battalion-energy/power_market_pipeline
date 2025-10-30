#!/usr/bin/env python3
"""
Compute Reserve Margin from DAM Ancillary Service Plan and Actual System Load
============================================================================

Reserve Margin = (Total Planned Reserves / Actual System Load) × 100%

This is a CRITICAL feature for price forecasting because:
- Low reserve margins (< 10%) → High scarcity risk → Price spikes
- High reserve margins (> 20%) → Oversupply → Low prices
- Reserve margin captures the supply-demand tightness

ERCOT typically targets ~13.75% reserve margin.
"""

import polars as pl
from pathlib import Path
from datetime import datetime

print("="*80)
print("COMPUTING RESERVE MARGIN FROM DAM AS PLAN AND ACTUAL LOAD")
print("="*80)
print(f"Started: {datetime.now()}")

# Paths
AS_PLAN_DIR = Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_clean_batch_dataset/parquet/DAM Ancillary Service Plan")
ACTUAL_LOAD_DIR = Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_clean_batch_dataset/parquet/Actual System Load by Forecast Zone")
OUTPUT_DIR = Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 1. LOAD DAM ANCILLARY SERVICE PLAN
# ============================================================================
print("\n" + "="*80)
print("1. LOADING DAM ANCILLARY SERVICE PLAN")
print("="*80)

as_files = sorted(AS_PLAN_DIR.glob("*.parquet"))
print(f"Found {len(as_files)} files")

all_as = []
for file in as_files:
    year = file.stem.split('_')[-1]
    df = pl.read_parquet(file)
    all_as.append(df)
    print(f"  ✓ {year}: {len(df):,} records")

df_as = pl.concat(all_as)
print(f"\nTotal AS records: {len(df_as):,}")

# Pivot to wide format (one row per timestamp with all AS types)
print("\nPivoting ancillary services to wide format...")
df_as_pivot = df_as.pivot(
    index='datetime_local',
    on='AncillaryType',
    values='Quantity',
    aggregate_function='first'
).fill_null(0)

# Calculate total reserves
as_cols = ['REGDN', 'REGUP', 'RRS', 'ECRS', 'NSPIN']
df_as_pivot = df_as_pivot.with_columns(
    pl.sum_horizontal(as_cols).alias('total_dam_reserves_MW')
)

print(f"✓ Pivoted to {len(df_as_pivot):,} timestamps")
print(f"  AS columns: {as_cols}")

# Show AS statistics
print("\nAncillary Services Statistics (MW):")
for col in as_cols + ['total_dam_reserves_MW']:
    stats = df_as_pivot.select(col).describe()
    mean = stats.filter(pl.col('statistic') == 'mean')[col][0]
    min_val = stats.filter(pl.col('statistic') == 'min')[col][0]
    max_val = stats.filter(pl.col('statistic') == 'max')[col][0]
    print(f"  {col:25s}: {mean:8.0f} MW (range: {min_val:6.0f} - {max_val:6.0f})")

# ============================================================================
# 2. LOAD ACTUAL SYSTEM LOAD
# ============================================================================
print("\n" + "="*80)
print("2. LOADING ACTUAL SYSTEM LOAD")
print("="*80)

load_files = sorted(ACTUAL_LOAD_DIR.glob("*.parquet"))
print(f"Found {len(load_files)} files")

all_load = []
for file in load_files:
    year = file.stem.split('_')[-1]
    df = pl.read_parquet(file)
    all_load.append(df)
    print(f"  ✓ {year}: {len(df):,} records")

df_load = pl.concat(all_load)
print(f"\nTotal load records: {len(df_load):,}")

# Show load statistics
stats = df_load.select('TOTAL').describe()
mean_load = stats.filter(pl.col('statistic') == 'mean')['TOTAL'][0]
min_load = stats.filter(pl.col('statistic') == 'min')['TOTAL'][0]
max_load = stats.filter(pl.col('statistic') == 'max')['TOTAL'][0]
print(f"\nSystem Load Statistics:")
print(f"  Mean: {mean_load:8,.0f} MW")
print(f"  Min:  {min_load:8,.0f} MW")
print(f"  Max:  {max_load:8,.0f} MW")

# ============================================================================
# 3. MERGE AND CALCULATE RESERVE MARGIN
# ============================================================================
print("\n" + "="*80)
print("3. CALCULATING RESERVE MARGIN")
print("="*80)

# Normalize timestamps for joining
df_as_clean = df_as_pivot.select([
    pl.col('datetime_local').dt.replace_time_zone(None).alias('timestamp'),
    'REGDN',
    'REGUP',
    'RRS',
    'ECRS',
    'NSPIN',
    'total_dam_reserves_MW'
])

df_load_clean = df_load.select([
    pl.col('datetime_local').dt.replace_time_zone(None).alias('timestamp'),
    pl.col('TOTAL').alias('actual_system_load_MW'),
    'NORTH',
    'SOUTH',
    'WEST',
    'HOUSTON'
])

# Join on timestamp
df_merged = df_load_clean.join(
    df_as_clean,
    on='timestamp',
    how='left'
)

print(f"Merged records: {len(df_merged):,}")
print(f"Records with both load and reserves: {df_merged.filter(pl.col('total_dam_reserves_MW').is_not_null()).height:,}")

# Calculate reserve margin
df_merged = df_merged.with_columns([
    (pl.col('total_dam_reserves_MW') / pl.col('actual_system_load_MW') * 100).alias('reserve_margin_pct')
])

# Flag tight conditions
df_merged = df_merged.with_columns([
    (pl.col('reserve_margin_pct') < 10).cast(pl.Int8).alias('tight_reserves_flag'),  # < 10% is tight
    (pl.col('reserve_margin_pct') < 7).cast(pl.Int8).alias('critical_reserves_flag')  # < 7% is critical
])

# Show statistics
print("\n" + "="*80)
print("RESERVE MARGIN STATISTICS")
print("="*80)

valid_data = df_merged.filter(pl.col('reserve_margin_pct').is_not_null())
stats = valid_data.select([
    'actual_system_load_MW',
    'total_dam_reserves_MW',
    'reserve_margin_pct'
]).describe()

print("\nSystem Load (MW):")
print(stats.filter(pl.col('statistic').is_in(['mean', 'min', 'max'])).select(['statistic', 'actual_system_load_MW']))

print("\nTotal DAM Reserves (MW):")
print(stats.filter(pl.col('statistic').is_in(['mean', 'min', 'max'])).select(['statistic', 'total_dam_reserves_MW']))

print("\nReserve Margin (%):")
print(stats.filter(pl.col('statistic').is_in(['mean', 'min', 'max', '25%', '50%', '75%'])).select(['statistic', 'reserve_margin_pct']))

# Count tight/critical periods
tight_count = valid_data.filter(pl.col('tight_reserves_flag') == 1).height
critical_count = valid_data.filter(pl.col('critical_reserves_flag') == 1).height
total_count = valid_data.height

print(f"\nScarcity Events:")
print(f"  Tight reserves (< 10%):    {tight_count:6,} hours ({100*tight_count/total_count:5.2f}%)")
print(f"  Critical reserves (< 7%):  {critical_count:6,} hours ({100*critical_count/total_count:5.2f}%)")

# ============================================================================
# 4. SAVE OUTPUT
# ============================================================================
print("\n" + "="*80)
print("4. SAVING OUTPUT")
print("="*80)

output_file = OUTPUT_DIR / "reserve_margin_dam_2018_2025.parquet"
df_merged.write_parquet(output_file)

print(f"\n✓ Saved: {output_file}")
print(f"  Size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
print(f"  Records: {len(df_merged):,}")
print(f"  Columns: {df_merged.columns}")

# Show sample
print("\n" + "="*80)
print("SAMPLE DATA")
print("="*80)
sample = valid_data.select([
    'timestamp',
    'actual_system_load_MW',
    'total_dam_reserves_MW',
    'reserve_margin_pct',
    'tight_reserves_flag',
    'critical_reserves_flag'
]).head(10)
print(sample)

print("\n" + "="*80)
print("✓ COMPLETE!")
print("="*80)
print(f"Finished: {datetime.now()}")
print(f"\nReady to merge with master dataset!")
print(f"Use: pl.read_parquet('{output_file}')")

#!/usr/bin/env python3
"""
Compute Net Load Features (System Load - Wind - Solar Generation)
================================================================

Net Load = System Load - Wind Generation - Solar Generation

This is CRITICAL for price forecasting because:
- Prices driven by net load, not gross load
- High wind at night → Low net load → Low prices (even if load is high)
- Solar drop at sunset → Net load ramp → Price spikes ("duck curve")
- Low net load with high reserves → Negative prices
- High net load with low reserves → Price spikes to $1000+/MWh

Also compute:
- Net load ramp (hour-over-hour change)
- Reserve margin on net load basis
"""

import polars as pl
from pathlib import Path
from datetime import datetime

print("="*80)
print("COMPUTING NET LOAD FEATURES FROM ACTUAL GENERATION DATA")
print("="*80)
print(f"Started: {datetime.now()}")

# Paths
GEN_DATA_DIR = Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_clean_batch_dataset/parquet/2-Day Real Time Gen and Load Data Reports")
ACTUAL_LOAD_DIR = Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_clean_batch_dataset/parquet/Actual System Load by Forecast Zone")
OUTPUT_DIR = Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 1. LOAD GENERATION DATA (5-minute SCED intervals)
# ============================================================================
print("\n" + "="*80)
print("1. LOADING REAL-TIME GENERATION DATA")
print("="*80)

gen_files = sorted(GEN_DATA_DIR.glob("*.parquet"))
print(f"Found {len(gen_files)} files")

all_gen = []
for file in gen_files:
    year = file.stem.split('_')[-1]
    df = pl.read_parquet(file)

    # Standardize column names (they changed over the years)
    # Select only the columns we need with consistent naming
    df_clean = df.select([
        pl.col('datetime_local'),
        pl.col([c for c in df.columns if 'BASE POINT WGR' in c or 'BASE POINT  WGR' in c][0] if any('WGR' in c for c in df.columns) else pl.lit(None)).alias('wind_gen'),
        pl.col([c for c in df.columns if 'BASE POINT PVGR' in c or 'BASE POINT  PVGR' in c][0] if any('PVGR' in c for c in df.columns) else pl.lit(None)).alias('solar_gen'),
        pl.col([c for c in df.columns if 'AGG LOAD' in c][0] if any('AGG LOAD' in c for c in df.columns) else pl.lit(None)).alias('agg_load'),
    ])

    all_gen.append(df_clean)
    print(f"  ✓ {year}: {len(df_clean):,} records")

df_gen = pl.concat(all_gen)
print(f"\nTotal generation records: {len(df_gen):,}")

# Show key columns
print(f"\nStandardized columns:")
print(f"  wind_gen   = Wind generation (MW)")
print(f"  solar_gen  = Solar/PV generation (MW)")
print(f"  agg_load   = Aggregate system load (MW)")

# ============================================================================
# 2. AGGREGATE TO HOURLY AND CALCULATE NET LOAD
# ============================================================================
print("\n" + "="*80)
print("2. AGGREGATING TO HOURLY AND CALCULATING NET LOAD")
print("="*80)

# Normalize timestamp and aggregate to hourly
df_gen = df_gen.with_columns([
    pl.col('datetime_local').dt.replace_time_zone(None).alias('timestamp')
])

# Floor to hour and aggregate
df_hourly = df_gen.group_by(
    pl.col('timestamp').dt.truncate('1h')
).agg([
    pl.col('wind_gen').mean().alias('wind_generation_MW'),
    pl.col('solar_gen').mean().alias('solar_generation_MW'),
    pl.col('agg_load').mean().alias('sced_system_load_MW'),
]).sort('timestamp')

# Handle nulls (fill with 0 for generation, which is reasonable)
df_hourly = df_hourly.with_columns([
    pl.col('wind_generation_MW').fill_null(0),
    pl.col('solar_generation_MW').fill_null(0),
])

print(f"Aggregated to {len(df_hourly):,} hourly records")

# ============================================================================
# 3. MERGE WITH ACTUAL SYSTEM LOAD (for validation)
# ============================================================================
print("\n" + "="*80)
print("3. MERGING WITH ACTUAL SYSTEM LOAD FOR VALIDATION")
print("="*80)

load_files = sorted(ACTUAL_LOAD_DIR.glob("*.parquet"))
all_load = []
for file in load_files:
    df = pl.read_parquet(file)
    all_load.append(df)

df_load = pl.concat(all_load)
df_load_clean = df_load.select([
    pl.col('datetime_local').dt.replace_time_zone(None).alias('timestamp'),
    pl.col('TOTAL').alias('actual_system_load_MW'),
    'NORTH',
    'SOUTH',
    'WEST',
    'HOUSTON'
])

# Merge
df_merged = df_load_clean.join(
    df_hourly,
    on='timestamp',
    how='left'
)

print(f"Merged records: {len(df_merged):,}")

# ============================================================================
# 4. CALCULATE NET LOAD AND DERIVED FEATURES
# ============================================================================
print("\n" + "="*80)
print("4. CALCULATING NET LOAD AND DERIVED FEATURES")
print("="*80)

df_merged = df_merged.with_columns([
    # Net load calculation
    (pl.col('actual_system_load_MW') -
     pl.col('wind_generation_MW').fill_null(0) -
     pl.col('solar_generation_MW').fill_null(0)).alias('net_load_MW'),

    # Renewable penetration
    ((pl.col('wind_generation_MW').fill_null(0) + pl.col('solar_generation_MW').fill_null(0)) /
     pl.col('actual_system_load_MW') * 100).alias('renewable_penetration_pct'),
])

# Calculate net load ramps and volatility
df_merged = df_merged.with_columns([
    # Hour-over-hour net load ramp
    (pl.col('net_load_MW') - pl.col('net_load_MW').shift(1)).alias('net_load_ramp_1h'),

    # 3-hour net load change
    (pl.col('net_load_MW') - pl.col('net_load_MW').shift(3)).alias('net_load_ramp_3h'),

    # Net load rolling statistics (24-hour window)
    pl.col('net_load_MW').rolling_mean(window_size=24).alias('net_load_roll_24h_mean'),
    pl.col('net_load_MW').rolling_std(window_size=24).alias('net_load_roll_24h_std'),
    pl.col('net_load_MW').rolling_max(window_size=24).alias('net_load_roll_24h_max'),
    pl.col('net_load_MW').rolling_min(window_size=24).alias('net_load_roll_24h_min'),
])

# Net load vs forecast zone loads
df_merged = df_merged.with_columns([
    (pl.col('HOUSTON') / pl.col('net_load_MW') * 100).alias('houston_pct_of_net_load'),
    (pl.col('NORTH') / pl.col('net_load_MW') * 100).alias('north_pct_of_net_load'),
])

# Flag extreme conditions
df_merged = df_merged.with_columns([
    # High renewable penetration (> 40%)
    (pl.col('renewable_penetration_pct') > 40).cast(pl.Int8).alias('high_renewable_flag'),

    # Large net load ramp (> 3000 MW/hour)
    (pl.col('net_load_ramp_1h').abs() > 3000).cast(pl.Int8).alias('large_ramp_flag'),

    # Solar curtailment risk (negative net load possible in future)
    (pl.col('net_load_MW') < 10000).cast(pl.Int8).alias('low_net_load_flag'),
])

# ============================================================================
# 5. STATISTICS AND VALIDATION
# ============================================================================
print("\n" + "="*80)
print("NET LOAD STATISTICS")
print("="*80)

valid_data = df_merged.filter(pl.col('net_load_MW').is_not_null())

stats = valid_data.select([
    'actual_system_load_MW',
    'wind_generation_MW',
    'solar_generation_MW',
    'net_load_MW',
    'renewable_penetration_pct'
]).describe()

print("\nSystem Load (MW):")
print(stats.filter(pl.col('statistic').is_in(['mean', 'min', 'max'])).select(['statistic', 'actual_system_load_MW']))

print("\nWind Generation (MW):")
print(stats.filter(pl.col('statistic').is_in(['mean', 'min', 'max'])).select(['statistic', 'wind_generation_MW']))

print("\nSolar Generation (MW):")
print(stats.filter(pl.col('statistic').is_in(['mean', 'min', 'max'])).select(['statistic', 'solar_generation_MW']))

print("\nNet Load (MW):")
print(stats.filter(pl.col('statistic').is_in(['mean', 'min', 'max', '25%', '50%', '75%'])).select(['statistic', 'net_load_MW']))

print("\nRenewable Penetration (%):")
print(stats.filter(pl.col('statistic').is_in(['mean', 'min', 'max', '25%', '50%', '75%'])).select(['statistic', 'renewable_penetration_pct']))

# Count extreme events
high_renewable = valid_data.filter(pl.col('high_renewable_flag') == 1).height
large_ramp = valid_data.filter(pl.col('large_ramp_flag') == 1).height
low_net_load = valid_data.filter(pl.col('low_net_load_flag') == 1).height
total = valid_data.height

print(f"\nExtreme Events:")
print(f"  High renewable (>40%):      {high_renewable:6,} hours ({100*high_renewable/total:5.2f}%)")
print(f"  Large ramps (>3000 MW/h):   {large_ramp:6,} hours ({100*large_ramp/total:5.2f}%)")
print(f"  Low net load (<10000 MW):   {low_net_load:6,} hours ({100*low_net_load/total:5.2f}%)")

# ============================================================================
# 6. SAVE OUTPUT
# ============================================================================
print("\n" + "="*80)
print("6. SAVING OUTPUT")
print("="*80)

output_file = OUTPUT_DIR / "net_load_features_2018_2025.parquet"
df_merged.write_parquet(output_file)

print(f"\n✓ Saved: {output_file}")
print(f"  Size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
print(f"  Records: {len(df_merged):,}")
print(f"  Columns: {len(df_merged.columns)}")

# Show sample
print("\n" + "="*80)
print("SAMPLE DATA")
print("="*80)
sample = valid_data.select([
    'timestamp',
    'actual_system_load_MW',
    'wind_generation_MW',
    'solar_generation_MW',
    'net_load_MW',
    'renewable_penetration_pct',
    'net_load_ramp_1h'
]).head(10)
print(sample)

print("\n" + "="*80)
print("✓ COMPLETE!")
print("="*80)
print(f"Finished: {datetime.now()}")
print(f"\nReady to merge with master dataset!")
print(f"Use: pl.read_parquet('{output_file}')")

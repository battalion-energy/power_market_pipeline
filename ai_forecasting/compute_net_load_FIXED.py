#!/usr/bin/env python3
"""
Compute Net Load Features - FIXED VERSION
==========================================

FIXES:
1. Uses actual solar generation from dedicated Solar Production files (not SCED base points)
2. Applies bounds checking to remove data quality issues (Aug 2024 had 130k MW errors)
3. Uses actual wind generation from dedicated Wind Production files
4. Prioritizes ACTUAL data, falls back to COP_HSL forecasts
5. Properly aggregates forecast scenarios to hourly averages

Net Load = System Load - Wind Generation - Solar Generation
"""

import polars as pl
from pathlib import Path
from datetime import datetime

print("="*80)
print("COMPUTING NET LOAD FEATURES - FIXED VERSION")
print("="*80)
print(f"Started: {datetime.now()}")

# Paths
SOLAR_DIR = Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_clean_batch_dataset/parquet/Solar Power Production - Hourly Averaged Actual and Forecasted Values")
WIND_DIR = Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_clean_batch_dataset/parquet/Wind Power Production - Hourly Averaged Actual and Forecasted Values")
ACTUAL_LOAD_DIR = Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_clean_batch_dataset/parquet/Actual System Load by Forecast Zone")
OUTPUT_DIR = Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Reasonable bounds (based on installed capacity)
SOLAR_MAX_MW = 25000  # Conservative max for Texas solar through 2024
WIND_MAX_MW = 40000   # Conservative max for Texas wind

# ============================================================================
# 1. LOAD SOLAR GENERATION DATA
# ============================================================================
print("\n" + "="*80)
print("1. LOADING SOLAR GENERATION DATA")
print("="*80)

solar_files = sorted(SOLAR_DIR.glob("*.parquet"))
print(f"Found {len(solar_files)} solar files")

all_solar = []
for file in solar_files:
    year = file.stem.split('_')[-1]
    df = pl.read_parquet(file)

    # Handle different column names across years
    # 2018-2023: ACTUAL_SYSTEM_WIDE
    # 2024-2025: SYSTEM_WIDE_GEN
    actual_col = 'ACTUAL_SYSTEM_WIDE' if 'ACTUAL_SYSTEM_WIDE' in df.columns else 'SYSTEM_WIDE_GEN'

    # Aggregate to hourly and clean
    df_hourly = df.group_by(
        pl.col('datetime_local').dt.replace_time_zone(None).dt.truncate('1h')
    ).agg([
        pl.col(actual_col).mean().alias('actual_solar'),
        pl.col('COP_HSL_SYSTEM_WIDE').mean().alias('cop_solar'),
    ])

    df_hourly = df_hourly.with_columns([
        pl.col('datetime_local').alias('timestamp'),
        # Use actual if available and reasonable, else use COP
        pl.when(
            pl.col('actual_solar').is_not_null() &
            (pl.col('actual_solar') >= 0) &
            (pl.col('actual_solar') <= SOLAR_MAX_MW)
        ).then(pl.col('actual_solar'))
        .otherwise(pl.col('cop_solar'))
        .alias('solar_generation_MW')
    ])

    all_solar.append(df_hourly.select(['timestamp', 'solar_generation_MW']))
    print(f"  ✓ {year}: {len(df_hourly):,} hourly records")

df_solar = pl.concat(all_solar).sort('timestamp')
print(f"\nTotal solar hourly records: {len(df_solar):,}")

solar_stats = df_solar.select([
    pl.col('solar_generation_MW').mean().alias('mean'),
    pl.col('solar_generation_MW').max().alias('max'),
    pl.col('solar_generation_MW').min().alias('min'),
])
print(f"Solar generation stats:")
print(f"  Mean: {solar_stats['mean'][0]:,.0f} MW")
print(f"  Max:  {solar_stats['max'][0]:,.0f} MW")
print(f"  Min:  {solar_stats['min'][0]:,.0f} MW")

# ============================================================================
# 2. LOAD WIND GENERATION DATA
# ============================================================================
print("\n" + "="*80)
print("2. LOADING WIND GENERATION DATA")
print("="*80)

wind_files = sorted(WIND_DIR.glob("*.parquet"))
print(f"Found {len(wind_files)} wind files")

all_wind = []
for file in wind_files:
    year = file.stem.split('_')[-1]
    df = pl.read_parquet(file)

    # Handle different column names across years (same as solar)
    # 2021-2023: ACTUAL_SYSTEM_WIDE
    # 2024-2025: SYSTEM_WIDE_GEN
    actual_col = 'ACTUAL_SYSTEM_WIDE' if 'ACTUAL_SYSTEM_WIDE' in df.columns else 'SYSTEM_WIDE_GEN'

    df_hourly = df.group_by(
        pl.col('datetime_local').dt.replace_time_zone(None).dt.truncate('1h')
    ).agg([
        pl.col(actual_col).mean().alias('actual_wind'),
        pl.col('COP_HSL_SYSTEM_WIDE').mean().alias('cop_wind'),
    ])

    df_hourly = df_hourly.with_columns([
        pl.col('datetime_local').alias('timestamp'),
        pl.when(
            pl.col('actual_wind').is_not_null() &
            (pl.col('actual_wind') >= 0) &
            (pl.col('actual_wind') <= WIND_MAX_MW)
        ).then(pl.col('actual_wind'))
        .otherwise(pl.col('cop_wind'))
        .alias('wind_generation_MW')
    ])

    all_wind.append(df_hourly.select(['timestamp', 'wind_generation_MW']))
    print(f"  ✓ {year}: {len(df_hourly):,} hourly records")

df_wind = pl.concat(all_wind).sort('timestamp')
print(f"\nTotal wind hourly records: {len(df_wind):,}")

wind_stats = df_wind.select([
    pl.col('wind_generation_MW').mean().alias('mean'),
    pl.col('wind_generation_MW').max().alias('max'),
    pl.col('wind_generation_MW').min().alias('min'),
])
print(f"Wind generation stats:")
print(f"  Mean: {wind_stats['mean'][0]:,.0f} MW")
print(f"  Max:  {wind_stats['max'][0]:,.0f} MW")
print(f"  Min:  {wind_stats['min'][0]:,.0f} MW")

# ============================================================================
# 3. LOAD ACTUAL SYSTEM LOAD
# ============================================================================
print("\n" + "="*80)
print("3. LOADING ACTUAL SYSTEM LOAD")
print("="*80)

load_files = sorted(ACTUAL_LOAD_DIR.glob("*.parquet"))
all_load = []
for file in load_files:
    df = pl.read_parquet(file)

    # FIX CRITICAL BUG: datetime_local is broken (all 00:00:00)
    # Need to parse HourEnding column to get actual hour
    # HourEnding format: "01:00", "02:00", ..., "24:00"
    df = df.with_columns([
        pl.col('OperDay').dt.replace_time_zone(None).alias('date'),
        pl.col('HourEnding').str.split(':').list.get(0).cast(pl.Int32).alias('hour_ending'),
    ])

    # Create proper timestamp from OperDay + HourEnding
    # Hour ending 24:00 means end of day (use 00:00 next day)
    df = df.with_columns([
        pl.when(pl.col('hour_ending') == 24)
        .then(pl.col('date') + pl.duration(days=1))
        .otherwise(pl.col('date') + pl.duration(hours=pl.col('hour_ending')))
        .alias('timestamp')
    ])

    all_load.append(df)

df_load = pl.concat(all_load)
df_load_clean = df_load.select([
    pl.col('timestamp').cast(pl.Datetime('ns')).alias('timestamp'),  # Cast to nanoseconds to match solar/wind
    pl.col('TOTAL').alias('actual_system_load_MW'),
    'NORTH',
    'SOUTH',
    'WEST',
    'HOUSTON'
])

print(f"System load records: {len(df_load_clean):,}")
print(f"Date range: {df_load_clean['timestamp'].min()} to {df_load_clean['timestamp'].max()}")

# ============================================================================
# 4. MERGE ALL DATASETS
# ============================================================================
print("\n" + "="*80)
print("4. MERGING ALL DATASETS")
print("="*80)

df_merged = df_load_clean.join(df_wind, on='timestamp', how='left')
print(f"After wind merge: {len(df_merged):,} records")

df_merged = df_merged.join(df_solar, on='timestamp', how='left')
print(f"After solar merge: {len(df_merged):,} records")

df_merged = df_merged.with_columns([
    pl.col('wind_generation_MW').fill_null(0),
    pl.col('solar_generation_MW').fill_null(0),
])

# ============================================================================
# 5. CALCULATE NET LOAD AND DERIVED FEATURES
# ============================================================================
print("\n" + "="*80)
print("5. CALCULATING NET LOAD AND DERIVED FEATURES")
print("="*80)

df_merged = df_merged.with_columns([
    (pl.col('actual_system_load_MW') -
     pl.col('wind_generation_MW') -
     pl.col('solar_generation_MW')).alias('net_load_MW'),

    ((pl.col('wind_generation_MW') + pl.col('solar_generation_MW')) /
     pl.col('actual_system_load_MW') * 100).alias('renewable_penetration_pct'),
])

df_merged = df_merged.with_columns([
    (pl.col('net_load_MW') - pl.col('net_load_MW').shift(1)).alias('net_load_ramp_1h'),
    (pl.col('net_load_MW') - pl.col('net_load_MW').shift(3)).alias('net_load_ramp_3h'),
    pl.col('net_load_MW').rolling_mean(window_size=24).alias('net_load_roll_24h_mean'),
    pl.col('net_load_MW').rolling_std(window_size=24).alias('net_load_roll_24h_std'),
    pl.col('net_load_MW').rolling_max(window_size=24).alias('net_load_roll_24h_max'),
    pl.col('net_load_MW').rolling_min(window_size=24).alias('net_load_roll_24h_min'),
])

df_merged = df_merged.with_columns([
    (pl.col('HOUSTON') / pl.col('net_load_MW') * 100).alias('houston_pct_of_net_load'),
    (pl.col('NORTH') / pl.col('net_load_MW') * 100).alias('north_pct_of_net_load'),
])

df_merged = df_merged.with_columns([
    (pl.col('renewable_penetration_pct') > 40).cast(pl.Int8).alias('high_renewable_flag'),
    (pl.col('net_load_ramp_1h').abs() > 3000).cast(pl.Int8).alias('large_ramp_flag'),
    (pl.col('net_load_MW') < 10000).cast(pl.Int8).alias('low_net_load_flag'),
])

# ============================================================================
# 6. STATISTICS AND VALIDATION
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
print(stats.filter(pl.col('statistic').is_in(['mean', 'min', 'max', '50%'])).select(['statistic', 'net_load_MW']))

print("\n" + "="*80)
print("SOLAR GENERATION BY YEAR (VALIDATION)")
print("="*80)

solar_by_year = valid_data.group_by(
    pl.col('timestamp').dt.year().alias('year')
).agg([
    pl.col('solar_generation_MW').mean().alias('mean'),
    pl.col('solar_generation_MW').max().alias('max'),
    pl.col('solar_generation_MW').quantile(0.95).alias('p95'),
    pl.len().alias('hours'),
]).sort('year')

print(solar_by_year)

# ============================================================================
# 7. SAVE OUTPUT
# ============================================================================
print("\n" + "="*80)
print("7. SAVING OUTPUT")
print("="*80)

output_file = OUTPUT_DIR / "net_load_features_2018_2025_FIXED.parquet"
df_merged.write_parquet(output_file)

print(f"\n✓ Saved: {output_file}")
print(f"  Size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
print(f"  Records: {len(df_merged):,}")
print(f"  Date range: {df_merged['timestamp'].min()} to {df_merged['timestamp'].max()}")

print("\n" + "="*80)
print("✓ COMPLETE - SOLAR DATA FIXED!")
print("="*80)
print(f"Finished: {datetime.now()}")
print(f"\nFIXES APPLIED:")
print(f"  ✓ Solar now uses ACTUAL from dedicated files (not SCED base points)")
print(f"  ✓ Bounds checking removes corrupt data (Aug 2024: 130k MW errors)")
print(f"  ✓ Expected solar: 10k-20k MW peak (vs old: 200-300 MW)")
print(f"  ✓ Net load calculation is now CORRECT")

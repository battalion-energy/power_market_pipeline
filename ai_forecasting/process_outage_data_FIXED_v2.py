#!/usr/bin/env python3
"""
Process Generator Outage Data - FIXED VERSION 2
==============================================

Updated to handle the actual column structure from extracted CSV files:
- Date + HourEnding columns (not datetime_local)
- TotalResourceMW, TotalIRRMW, TotalNewEquipResourceMW as totals

Generator outages are CRITICAL for price forecasting:
- High outages (>20 GW) → Supply shortfall → Price spikes
- Thermal outages more critical than renewable
- Sudden large outages → Scarcity pricing
"""

import polars as pl
from pathlib import Path
from datetime import datetime

print("="*80)
print("PROCESSING GENERATOR OUTAGE DATA - FIXED VERSION 2")
print("="*80)
print(f"Started: {datetime.now()}")

# Paths
OUTAGE_DIR = Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_clean_batch_dataset/parquet/Hourly Resource Outage Capacity")
OUTPUT_DIR = Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 1. LOAD ALL OUTAGE FILES
# ============================================================================

print("\n" + "="*80)
print("1. LOADING HOURLY RESOURCE OUTAGE CAPACITY FILES")
print("="*80)

outage_files = sorted(OUTAGE_DIR.glob("*.parquet"))
print(f"Found {len(outage_files)} files")

all_outages = []
for file in outage_files:
    year = file.stem.split('_')[-1]
    df = pl.read_parquet(file)

    print(f"\nProcessing {year}:")
    print(f"  Columns: {df.columns[:5]}... ({len(df.columns)} total)")
    print(f"  Rows: {len(df):,}")

    # Create timestamp from Date and HourEnding
    # Date format is MM/DD/YYYY, HourEnding is 1-24
    df = df.with_columns([
        # Parse date string and combine with hour
        pl.col('Date').str.strptime(pl.Date, "%m/%d/%Y").alias('date'),
        # HourEnding 1-24 needs to be converted to hour 0-23
        (pl.col('HourEnding') - 1).alias('hour')
    ])

    # Combine date and hour into timestamp
    df = df.with_columns([
        (pl.col('date').cast(pl.Datetime) + pl.duration(hours=pl.col('hour'))).alias('timestamp')
    ])

    # Select only the columns we need
    cols_to_select = [
        pl.col('timestamp'),
    ]

    # Total outage capacity
    if 'TotalResourceMW' in df.columns:
        cols_to_select.append(pl.col('TotalResourceMW').cast(pl.Float64).alias('outage_total_MW'))

    # Renewable outages (IRR = Intermittent Renewable Resources)
    if 'TotalIRRMW' in df.columns:
        cols_to_select.append(pl.col('TotalIRRMW').cast(pl.Float64).alias('outage_renewable_MW'))

    # New equipment outages
    if 'TotalNewEquipResourceMW' in df.columns:
        cols_to_select.append(pl.col('TotalNewEquipResourceMW').cast(pl.Float64).alias('outage_new_equip_MW'))

    df_clean = df.select(cols_to_select)

    all_outages.append(df_clean)

    unique_hours = df_clean.select('timestamp').unique().height
    print(f"  ✓ {year}: {len(df_clean):,} records, {unique_hours:,} unique hours")

df_all = pl.concat(all_outages, how='diagonal_relaxed')
print(f"\nTotal outage records: {len(df_all):,}")

# ============================================================================
# 2. AGGREGATE TO HOURLY TOTALS
# ============================================================================

print("\n" + "="*80)
print("2. AGGREGATING TO HOURLY TOTALS")
print("="*80)

# Note: Each hour should have only one record per file, but we aggregate
# to be safe and to combine any duplicates across files
agg_exprs = [
    pl.col('outage_total_MW').max().alias('outage_total_MW'),
]

if 'outage_renewable_MW' in df_all.columns:
    agg_exprs.append(pl.col('outage_renewable_MW').max().alias('outage_renewable_MW'))

if 'outage_new_equip_MW' in df_all.columns:
    agg_exprs.append(pl.col('outage_new_equip_MW').max().alias('outage_new_equip_MW'))

df_hourly = df_all.group_by('timestamp').agg(agg_exprs).sort('timestamp')

print(f"Hourly aggregated records: {len(df_hourly):,}")
print(f"Unique timestamps: {df_hourly.select('timestamp').unique().height:,}")

# ============================================================================
# 3. CALCULATE THERMAL OUTAGES
# ============================================================================

print("\n" + "="*80)
print("3. CALCULATING THERMAL OUTAGES")
print("="*80)

# Thermal outages (total minus renewable)
df_hourly = df_hourly.with_columns([
    (pl.col('outage_total_MW') - pl.col('outage_renewable_MW').fill_null(0)).alias('outage_thermal_MW')
])

# ============================================================================
# 4. CALCULATE DERIVED FEATURES
# ============================================================================

print("\n" + "="*80)
print("4. CALCULATING DERIVED FEATURES")
print("="*80)

# Hour-over-hour changes
df_hourly = df_hourly.with_columns([
    (pl.col('outage_total_MW') - pl.col('outage_total_MW').shift(1)).alias('outage_change_1h'),
    (pl.col('outage_thermal_MW') - pl.col('outage_thermal_MW').shift(1)).alias('outage_thermal_change_1h'),
])

# Rolling statistics
df_hourly = df_hourly.with_columns([
    pl.col('outage_total_MW').rolling_mean(window_size=3).alias('outage_roll_3h_mean'),
    pl.col('outage_total_MW').rolling_max(window_size=3).alias('outage_roll_3h_max'),
    pl.col('outage_total_MW').rolling_mean(window_size=24).alias('outage_roll_24h_mean'),
    pl.col('outage_total_MW').rolling_max(window_size=24).alias('outage_roll_24h_max'),
])

# Flags for extreme conditions
df_hourly = df_hourly.with_columns([
    # High outages (>20 GW)
    (pl.col('outage_total_MW') > 20000).cast(pl.Int8).alias('high_outage_flag'),

    # Critical outages (>30 GW)
    (pl.col('outage_total_MW') > 30000).cast(pl.Int8).alias('critical_outage_flag'),

    # Large thermal outage (>15 GW thermal offline)
    (pl.col('outage_thermal_MW') > 15000).cast(pl.Int8).alias('large_thermal_outage_flag'),

    # Sudden outage spike (>5 GW change in 1 hour)
    (pl.col('outage_change_1h').abs() > 5000).cast(pl.Int8).alias('sudden_outage_flag'),
])

# ============================================================================
# 5. STATISTICS AND VALIDATION
# ============================================================================

print("\n" + "="*80)
print("OUTAGE STATISTICS")
print("="*80)

valid_data = df_hourly.filter(pl.col('outage_total_MW').is_not_null())

stats = valid_data.select([
    'outage_total_MW',
    'outage_thermal_MW',
    'outage_renewable_MW',
]).describe()

print("\nTotal Outages (MW):")
print(stats.filter(pl.col('statistic').is_in(['mean', 'min', 'max', '50%', '75%'])).select(['statistic', 'outage_total_MW']))

print("\nThermal Outages (MW):")
print(stats.filter(pl.col('statistic').is_in(['mean', 'min', 'max', '50%', '75%'])).select(['statistic', 'outage_thermal_MW']))

print("\nRenewable Outages (MW):")
print(stats.filter(pl.col('statistic').is_in(['mean', 'min', 'max', '50%', '75%'])).select(['statistic', 'outage_renewable_MW']))

# Count extreme events
high_outage = valid_data.filter(pl.col('high_outage_flag') == 1).height
critical_outage = valid_data.filter(pl.col('critical_outage_flag') == 1).height
large_thermal = valid_data.filter(pl.col('large_thermal_outage_flag') == 1).height
sudden_outage = valid_data.filter(pl.col('sudden_outage_flag') == 1).height
total_valid = valid_data.height

print(f"\nExtreme Outage Events:")
print(f"  High outages (>20 GW):       {high_outage:6,} hours ({100*high_outage/total_valid:5.2f}%)")
print(f"  Critical outages (>30 GW):   {critical_outage:6,} hours ({100*critical_outage/total_valid:5.2f}%)")
print(f"  Large thermal (>15 GW):      {large_thermal:6,} hours ({100*large_thermal/total_valid:5.2f}%)")
print(f"  Sudden spikes (>5 GW/h):     {sudden_outage:6,} hours ({100*sudden_outage/total_valid:5.2f}%)")

# ============================================================================
# 6. SAVE OUTPUT
# ============================================================================

print("\n" + "="*80)
print("6. SAVING OUTPUT")
print("="*80)

output_file = OUTPUT_DIR / "generator_outages_2019_2025.parquet"
df_hourly.write_parquet(output_file)

print(f"\n✓ Saved: {output_file}")
print(f"  Size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
print(f"  Records: {len(df_hourly):,}")
print(f"  Unique hourly timestamps: {df_hourly.select('timestamp').unique().height:,}")
print(f"  Columns: {len(df_hourly.columns)}")

# Show sample
print("\n" + "="*80)
print("SAMPLE DATA (First 24 hours)")
print("="*80)
sample = valid_data.select([
    'timestamp',
    'outage_total_MW',
    'outage_thermal_MW',
    'outage_renewable_MW',
    'high_outage_flag',
    'critical_outage_flag'
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
print(f"\nOutput file for downstream aggregation:")
print(f"  {output_file}")
print(f"\nThis file should be merged with the master ML dataset at:")
print(f"  /pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/processed/")

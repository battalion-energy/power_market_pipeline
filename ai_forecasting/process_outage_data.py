#!/usr/bin/env python3
"""
Process Generator Outage Capacity Data
======================================

CRITICAL for price forecasting:
- Large thermal outages reduce available supply → price spikes
- 20,000+ MW out = significant scarcity risk
- Combine with load and reserves for complete supply picture

Features to create:
- Total outage capacity (MW)
- Thermal vs renewable outages
- Hour-over-hour change in outages (sudden outages)
- Rolling statistics (outage trends)
"""

import polars as pl
from pathlib import Path
from datetime import datetime

print("="*80)
print("PROCESSING GENERATOR OUTAGE CAPACITY DATA")
print("="*80)
print(f"Started: {datetime.now()}")

# Paths
OUTAGE_FILE = Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_clean_batch_dataset/parquet/Hourly Resource Outage Capacity/Hourly Resource Outage Capacity_1970.parquet")
OUTPUT_DIR = Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 1. LOAD OUTAGE DATA
# ============================================================================
print("\n" + "="*80)
print("1. LOADING OUTAGE DATA")
print("="*80)

df = pl.read_parquet(OUTAGE_FILE)
print(f"Loaded: {len(df):,} records")

# Use datetime_local and drop timezone for consistency
df = df.with_columns([
    pl.col('datetime_local').dt.replace_time_zone(None).alias('timestamp')
])

# Select key columns
df_clean = df.select([
    'timestamp',
    'TotalResourceMW',              # Total generation on outage
    'TotalIRRMW',                   # Intermittent renewables on outage
    'TotalNewEquipResourceMW',      # New equipment on outage
    'TotalResourceMWZoneSouth',
    'TotalResourceMWZoneNorth',
    'TotalResourceMWZoneWest',
    'TotalResourceMWZoneHouston',
])

# Rename for clarity
df_clean = df_clean.rename({
    'TotalResourceMW': 'outage_total_MW',
    'TotalIRRMW': 'outage_renewable_MW',
    'TotalNewEquipResourceMW': 'outage_new_equip_MW',
    'TotalResourceMWZoneSouth': 'outage_south_MW',
    'TotalResourceMWZoneNorth': 'outage_north_MW',
    'TotalResourceMWZoneWest': 'outage_west_MW',
    'TotalResourceMWZoneHouston': 'outage_houston_MW',
})

print(f"✓ Cleaned: {len(df_clean):,} records")

# ============================================================================
# 2. CALCULATE DERIVED FEATURES
# ============================================================================
print("\n" + "="*80)
print("2. CALCULATING DERIVED FEATURES")
print("="*80)

# Thermal outages (non-renewable)
df_clean = df_clean.with_columns([
    (pl.col('outage_total_MW') - pl.col('outage_renewable_MW').fill_null(0)).alias('outage_thermal_MW')
])

# Outage changes (sudden unplanned outages)
df_clean = df_clean.with_columns([
    (pl.col('outage_total_MW') - pl.col('outage_total_MW').shift(1)).alias('outage_change_1h'),
    (pl.col('outage_thermal_MW') - pl.col('outage_thermal_MW').shift(1)).alias('outage_thermal_change_1h'),
])

# Rolling statistics (3-hour and 24-hour trends)
df_clean = df_clean.with_columns([
    pl.col('outage_total_MW').rolling_mean(window_size=3).alias('outage_roll_3h_mean'),
    pl.col('outage_total_MW').rolling_max(window_size=3).alias('outage_roll_3h_max'),
    pl.col('outage_total_MW').rolling_mean(window_size=24).alias('outage_roll_24h_mean'),
    pl.col('outage_total_MW').rolling_max(window_size=24).alias('outage_roll_24h_max'),
])

# Flags for extreme outage events
df_clean = df_clean.with_columns([
    # High outages (> 15,000 MW)
    (pl.col('outage_total_MW') > 15000).cast(pl.Int8).alias('high_outage_flag'),

    # Critical outages (> 20,000 MW)
    (pl.col('outage_total_MW') > 20000).cast(pl.Int8).alias('critical_outage_flag'),

    # Large thermal outages (> 10,000 MW)
    (pl.col('outage_thermal_MW') > 10000).cast(pl.Int8).alias('large_thermal_outage_flag'),

    # Sudden outage spike (> 2,000 MW increase)
    (pl.col('outage_change_1h') > 2000).cast(pl.Int8).alias('sudden_outage_flag'),
])

print(f"✓ Added derived features")

# ============================================================================
# 3. STATISTICS
# ============================================================================
print("\n" + "="*80)
print("OUTAGE STATISTICS")
print("="*80)

valid_data = df_clean.filter(pl.col('outage_total_MW').is_not_null())

stats = valid_data.select([
    'outage_total_MW',
    'outage_thermal_MW',
    'outage_renewable_MW',
]).describe()

print("\nTotal Outages (MW):")
print(stats.filter(pl.col('statistic').is_in(['mean', 'min', 'max', '50%'])).select(['statistic', 'outage_total_MW']))

print("\nThermal Outages (MW):")
print(stats.filter(pl.col('statistic').is_in(['mean', 'min', 'max', '50%'])).select(['statistic', 'outage_thermal_MW']))

print("\nRenewable Outages (MW):")
print(stats.filter(pl.col('statistic').is_in(['mean', 'min', 'max', '50%'])).select(['statistic', 'outage_renewable_MW']))

# Count extreme events
high_outage = valid_data.filter(pl.col('high_outage_flag') == 1).height
critical_outage = valid_data.filter(pl.col('critical_outage_flag') == 1).height
large_thermal = valid_data.filter(pl.col('large_thermal_outage_flag') == 1).height
sudden_outage = valid_data.filter(pl.col('sudden_outage_flag') == 1).height
total = valid_data.height

print(f"\nExtreme Outage Events:")
print(f"  High outages (>15,000 MW):    {high_outage:6,} hours ({100*high_outage/total:5.2f}%)")
print(f"  Critical outages (>20,000 MW): {critical_outage:6,} hours ({100*critical_outage/total:5.2f}%)")
print(f"  Large thermal (>10,000 MW):    {large_thermal:6,} hours ({100*large_thermal/total:5.2f}%)")
print(f"  Sudden outages (>2,000 MW/h):  {sudden_outage:6,} hours ({100*sudden_outage/total:5.2f}%)")

# ============================================================================
# 4. SAVE OUTPUT
# ============================================================================
print("\n" + "="*80)
print("4. SAVING OUTPUT")
print("="*80)

output_file = OUTPUT_DIR / "generator_outages_2018_2025.parquet"
df_clean.write_parquet(output_file)

print(f"\n✓ Saved: {output_file}")
print(f"  Size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
print(f"  Records: {len(df_clean):,}")
print(f"  Columns: {len(df_clean.columns)}")

# Show date range
min_date = df_clean.select(pl.col('timestamp').min())[0,0]
max_date = df_clean.select(pl.col('timestamp').max())[0,0]
print(f"\nDate range: {min_date} to {max_date}")

# Show sample
print("\n" + "="*80)
print("SAMPLE DATA")
print("="*80)
sample = valid_data.select([
    'timestamp',
    'outage_total_MW',
    'outage_thermal_MW',
    'outage_renewable_MW',
    'high_outage_flag',
    'critical_outage_flag'
]).head(10)
print(sample)

print("\n" + "="*80)
print("✓ COMPLETE!")
print("="*80)
print(f"Finished: {datetime.now()}")
print(f"\nReady to merge with master dataset and dashboard!")
print(f"Use: pl.read_parquet('{output_file}')")

#!/usr/bin/env python3
"""
Create Clean Dashboard Metrics Dataset
========================================

Extract key metrics for visualization and business logic.
This is NOT the training dataset - it's a clean, focused dataset
for displaying in graphs alongside forecasts.

Key Metrics:
- Prices (DA and RT)
- Load (actual, forecast, net load)
- Generation (wind, solar, renewables %)
- Reserve margin and ancillary services
- ORDC scarcity indicators
- Weather (temperature)
"""

import polars as pl
from pathlib import Path
from datetime import datetime

print("="*80)
print("CREATING DASHBOARD METRICS DATASET")
print("="*80)
print(f"Started: {datetime.now()}")

# Load enhanced master dataset
MASTER_FILE = Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/master_enhanced_with_net_load_reserves_2019_2025.parquet")
OUTPUT_DIR = Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data")

print("\nLoading enhanced master dataset...")
df = pl.read_parquet(MASTER_FILE)
print(f"  ✓ Loaded: {len(df):,} records, {len(df.columns)} columns")

# ============================================================================
# SELECT KEY METRICS FOR DASHBOARD
# ============================================================================
print("\n" + "="*80)
print("SELECTING KEY METRICS FOR DASHBOARD")
print("="*80)

# Core columns for visualization
dashboard_cols = [
    # Timestamp
    'timestamp',

    # Prices
    'price_da',          # Day-Ahead market price ($/MWh)
    'price_mean',        # Real-Time market price ($/MWh)

    # Load
    'net_load_MW',                # Net load (load - wind - solar)
    'load_forecast_mean',         # Day-ahead load forecast

    # Generation
    'wind_generation_MW',         # Wind generation
    'solar_generation_MW',        # Solar/PV generation
    'renewable_penetration_pct',  # (Wind + Solar) / Load * 100

    # Net Load Dynamics
    'net_load_ramp_1h',          # Hour-over-hour net load change
    'net_load_ramp_3h',          # 3-hour net load change

    # Reserve Margin
    'reserve_margin_pct',        # (Total Reserves / Load) * 100
    'total_dam_reserves_MW',     # Total DAM ancillary services
    'tight_reserves_flag',       # 1 if reserve margin < 10%
    'critical_reserves_flag',    # 1 if reserve margin < 7%

    # Ancillary Services (individual)
    'REGDN',                     # Regulation Down
    'REGUP',                     # Regulation Up
    'RRS',                       # Responsive Reserve Service
    'ECRS',                      # ERCOT Contingency Reserve Service
    'NSPIN',                     # Non-Spinning Reserve

    # ORDC Scarcity Indicators
    'ordc_online_reserves_mean',      # Online reserves (hourly mean)
    'ordc_online_reserves_min',       # Online reserves (hourly min)
    'ordc_scarcity_indicator_max',    # 1 if any 15-min interval < 2000 MW
    'ordc_critical_indicator_max',    # 1 if any 15-min interval < 1000 MW
    'ordc_reliability_price_adder_max',  # Max reliability price adder in hour
]

# Check which columns exist
available_cols = [col for col in dashboard_cols if col in df.columns]
missing_cols = [col for col in dashboard_cols if col not in df.columns]

print(f"\nAvailable columns: {len(available_cols)}/{len(dashboard_cols)}")
if missing_cols:
    print(f"Missing columns: {missing_cols}")

# Select available columns
df_dashboard = df.select(available_cols)

print(f"\n✓ Selected {len(df_dashboard.columns)} key metrics")

# ============================================================================
# ADD DERIVED METRICS
# ============================================================================
print("\n" + "="*80)
print("ADDING DERIVED METRICS")
print("="*80)

# Add price spike indicators
df_dashboard = df_dashboard.with_columns([
    # DA price spike (> $100/MWh)
    (pl.col('price_da') > 100).cast(pl.Int8).alias('da_price_spike_flag'),

    # RT price spike (> $100/MWh)
    (pl.col('price_mean') > 100).cast(pl.Int8).alias('rt_price_spike_flag'),

    # Extreme price event (> $500/MWh)
    (pl.col('price_mean') > 500).cast(pl.Int8).alias('extreme_price_flag'),

    # Price spread (RT - DA)
    (pl.col('price_mean') - pl.col('price_da')).alias('rt_da_spread'),
])

# Add temporal features for analysis
df_dashboard = df_dashboard.with_columns([
    pl.col('timestamp').dt.hour().alias('hour_of_day'),
    pl.col('timestamp').dt.weekday().alias('day_of_week'),
    pl.col('timestamp').dt.month().alias('month'),
    pl.col('timestamp').dt.year().alias('year'),
])

print(f"✓ Added derived metrics")
print(f"  Total columns: {len(df_dashboard.columns)}")

# ============================================================================
# DATA QUALITY SUMMARY
# ============================================================================
print("\n" + "="*80)
print("DATA QUALITY SUMMARY")
print("="*80)

# Check completeness of critical metrics
critical_metrics = [
    'price_da',
    'price_mean',
    'net_load_MW',
    'wind_generation_MW',
    'reserve_margin_pct'
]

for col in critical_metrics:
    if col in df_dashboard.columns:
        non_null = df_dashboard.select(pl.col(col).is_not_null().sum())[col][0]
        total = len(df_dashboard)
        pct = 100 * non_null / total
        print(f"  {col:30s}: {non_null:7,}/{total:7,} ({pct:5.1f}%)")

# Date range
min_date = df_dashboard.select(pl.col('timestamp').min())[0,0]
max_date = df_dashboard.select(pl.col('timestamp').max())[0,0]
print(f"\nDate range: {min_date} to {max_date}")

# ============================================================================
# SAVE DASHBOARD DATASET
# ============================================================================
print("\n" + "="*80)
print("SAVING DASHBOARD DATASET")
print("="*80)

output_file = OUTPUT_DIR / "ercot_dashboard_metrics_2019_2025.parquet"
df_dashboard.write_parquet(output_file)

print(f"\n✓ Saved: {output_file}")
print(f"  Size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
print(f"  Records: {len(df_dashboard):,}")
print(f"  Columns: {len(df_dashboard.columns)}")

# Show column list
print("\n" + "="*80)
print("DASHBOARD COLUMNS")
print("="*80)
for i, col in enumerate(df_dashboard.columns, 1):
    print(f"  {i:2}. {col}")

# Show sample data
print("\n" + "="*80)
print("SAMPLE DATA (Recent)")
print("="*80)
sample = df_dashboard.sort('timestamp', descending=True).head(5)
print(sample.select([
    'timestamp',
    'price_da',
    'price_mean',
    'net_load_MW',
    'wind_generation_MW',
    'reserve_margin_pct'
]))

print("\n" + "="*80)
print("✓ COMPLETE!")
print("="*80)
print(f"Finished: {datetime.now()}")
print(f"\nDashboard-ready dataset created!")
print(f"Use this for graphs, API responses, and business logic.")

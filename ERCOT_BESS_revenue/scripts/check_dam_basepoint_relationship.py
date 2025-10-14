#!/usr/bin/env python3
"""
Check if SCED BasePoint = DAM Award + RT Adjustment

This is CRITICAL - if BasePoint includes DAM, we're double counting!
"""

import polars as pl
from pathlib import Path

ROLLUP_DIR = Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/rollup_files")

# Test with RRANCHES_UNIT2
gen = "RRANCHES_UNIT2"
load = "RRANCHES_LD2"

print("=" * 100)
print("TESTING: Does SCED BasePoint include DAM Awards?")
print("=" * 100)
print()

# Load May 2024 data
print("Loading May 2024 DAM Gen data...")
df_dam_gen = pl.read_parquet(ROLLUP_DIR / "DAM_Gen_Resources/2024.parquet").filter(
    pl.col("ResourceName") == gen
).with_columns([
    pl.col("DeliveryDate").cast(pl.Date),
    pl.col("HourEnding").str.slice(0, 2).cast(pl.Int32).alias("hour_int")
]).filter(
    (pl.col("DeliveryDate").dt.month() == 5) & (pl.col("DeliveryDate").dt.year() == 2024)
).select([
    "DeliveryDate",
    "HourEnding",
    "hour_int",
    pl.col("AwardedQuantity").alias("dam_energy_mw")
])

print(f"DAM Gen intervals: {len(df_dam_gen):,}")
print("Sample DAM awards:")
print(df_dam_gen.head(10))
print()

print("Loading May 2024 SCED Gen data...")
df_sced_gen = pl.read_parquet(ROLLUP_DIR / "SCED_Gen_Resources/2024.parquet").filter(
    pl.col("ResourceName") == gen
).with_columns([
    pl.col("SCEDTimeStamp").str.strptime(pl.Datetime, "%m/%d/%Y %H:%M:%S")
        .dt.replace_time_zone("America/Chicago")
        .dt.convert_time_zone("UTC")
        .alias("sced_dt")
]).filter(
    (pl.col("sced_dt").dt.month() == 5) & (pl.col("sced_dt").dt.year() == 2024)
).select([
    "SCEDTimeStamp",
    "sced_dt",
    pl.col("BasePoint").alias("sced_basepoint")
])

print(f"SCED Gen intervals: {len(df_sced_gen):,}")
print()

# Aggregate SCED to hourly to compare with DAM
print("Aggregating SCED to hourly...")
df_sced_hourly = df_sced_gen.with_columns([
    pl.col("sced_dt").dt.date().alias("date"),
    (pl.col("sced_dt").dt.hour() + 1).alias("hour_ending")  # ERCOT HE convention
]).group_by(["date", "hour_ending"]).agg([
    pl.col("sced_basepoint").mean().alias("avg_sced_mw"),
    pl.col("sced_basepoint").min().alias("min_sced_mw"),
    pl.col("sced_basepoint").max().alias("max_sced_mw"),
    pl.len().alias("intervals")
])

print(f"SCED hourly aggregations: {len(df_sced_hourly):,}")
print()

# Join DAM with SCED hourly
print("Joining DAM with hourly SCED...")
df_comparison = df_dam_gen.join(
    df_sced_hourly,
    left_on=["DeliveryDate", "hour_int"],
    right_on=["date", "hour_ending"],
    how="inner"
).with_columns([
    (pl.col("avg_sced_mw") - pl.col("dam_energy_mw")).alias("rt_delta_avg"),
    (pl.col("min_sced_mw") - pl.col("dam_energy_mw")).alias("rt_delta_min"),
    (pl.col("max_sced_mw") - pl.col("dam_energy_mw")).alias("rt_delta_max")
])

print(f"Joined rows: {len(df_comparison):,}")
print()

print("=" * 100)
print("COMPARISON: DAM Award vs SCED BasePoint (May 2024)")
print("=" * 100)
print()

print("Sample hours (first 20):")
print(df_comparison.select([
    "DeliveryDate",
    "HourEnding",
    "dam_energy_mw",
    "avg_sced_mw",
    "min_sced_mw",
    "max_sced_mw",
    "rt_delta_avg"
]).head(20))
print()

# Statistics
print("=" * 100)
print("STATISTICS")
print("=" * 100)

stats = df_comparison.select([
    pl.col("dam_energy_mw").mean().alias("avg_dam"),
    pl.col("avg_sced_mw").mean().alias("avg_sced"),
    pl.col("rt_delta_avg").mean().alias("avg_delta"),
    pl.col("rt_delta_avg").abs().mean().alias("avg_abs_delta"),
    pl.col("rt_delta_avg").min().alias("min_delta"),
    pl.col("rt_delta_avg").max().alias("max_delta")
])

print(stats)
print()

# Count how many hours have DAM close to SCED
df_comparison = df_comparison.with_columns([
    (pl.col("dam_energy_mw").abs() < 0.01).alias("dam_zero"),
    (pl.col("avg_sced_mw").abs() < 0.01).alias("sced_zero"),
    (pl.col("rt_delta_avg").abs() < 1.0).alias("delta_small")
])

counts = df_comparison.select([
    pl.col("dam_zero").sum().alias("hours_with_zero_dam"),
    pl.col("sced_zero").sum().alias("hours_with_zero_sced"),
    pl.col("delta_small").sum().alias("hours_with_small_delta"),
    pl.len().alias("total_hours")
])

print("Counts:")
print(counts)
print()

# Check correlation
correlation = df_comparison.select([
    pl.corr("dam_energy_mw", "avg_sced_mw").alias("correlation")
]).item()

print(f"Correlation between DAM and SCED: {correlation:.4f}")
print()

print("=" * 100)
print("INTERPRETATION")
print("=" * 100)
print()

if correlation > 0.8:
    print("⚠️  HIGH CORRELATION - SCED BasePoint appears to INCLUDE DAM component!")
    print()
    print("This means:")
    print("  1. We're DOUBLE COUNTING DAM energy in our current calculation")
    print("  2. RT revenue should be: (SCED BasePoint - DAM Award) × RT Price")
    print("  3. This explains why revenues are so low!")
    print()
else:
    print("✓ LOW CORRELATION - SCED BasePoint appears INDEPENDENT of DAM")
    print()
    print("This means:")
    print("  1. Current calculation is correct")
    print("  2. DAM and RT are separate dispatch instructions")
    print()

# Show example of high-discharge hour
print("=" * 100)
print("EXAMPLE: Hour with high discharge")
print("=" * 100)

high_discharge = df_comparison.filter(
    pl.col("avg_sced_mw") > 50
).head(5)

print(high_discharge.select([
    "DeliveryDate",
    "HourEnding",
    "dam_energy_mw",
    "avg_sced_mw",
    "min_sced_mw",
    "max_sced_mw",
    "rt_delta_avg"
]))
print()

# Check the same for Load Resource
print()
print("=" * 100)
print("CHECKING LOAD RESOURCE")
print("=" * 100)
print()

print("Loading May 2024 DAM Load data...")
df_dam_load = pl.read_parquet(ROLLUP_DIR / "DAM_Load_Resources/2024.parquet").filter(
    pl.col("Load Resource Name") == load
).with_columns([
    pl.col("Delivery Date").cast(pl.Date).alias("DeliveryDate"),
    pl.col("Hour Ending").str.slice(0, 2).cast(pl.Int32).alias("hour_int")
]).filter(
    (pl.col("DeliveryDate").dt.month() == 5) & (pl.col("DeliveryDate").dt.year() == 2024)
).select([
    "DeliveryDate",
    "Hour Ending",
    "hour_int",
    pl.col("Awarded Quantity").alias("dam_energy_mw")
])

print(f"DAM Load intervals: {len(df_dam_load):,}")
print()

print("Loading May 2024 SCED Load data...")
df_sced_load = pl.read_parquet(ROLLUP_DIR / "SCED_Load_Resources/2024.parquet").filter(
    pl.col("ResourceName") == load
).with_columns([
    pl.col("SCEDTimeStamp").str.strptime(pl.Datetime, "%m/%d/%Y %H:%M:%S")
        .dt.replace_time_zone("America/Chicago")
        .dt.convert_time_zone("UTC")
        .alias("sced_dt")
]).filter(
    (pl.col("sced_dt").dt.month() == 5) & (pl.col("sced_dt").dt.year() == 2024)
).select([
    "SCEDTimeStamp",
    "sced_dt",
    pl.col("BasePoint").alias("sced_basepoint")
])

print(f"SCED Load intervals: {len(df_sced_load):,}")
print()

# Aggregate and compare
df_sced_load_hourly = df_sced_load.with_columns([
    pl.col("sced_dt").dt.date().alias("date"),
    (pl.col("sced_dt").dt.hour() + 1).alias("hour_ending")
]).group_by(["date", "hour_ending"]).agg([
    pl.col("sced_basepoint").mean().alias("avg_sced_mw")
])

df_load_comparison = df_dam_load.join(
    df_sced_load_hourly,
    left_on=["DeliveryDate", "hour_int"],
    right_on=["date", "hour_ending"],
    how="inner"
).with_columns([
    (pl.col("avg_sced_mw") - pl.col("dam_energy_mw")).alias("rt_delta_avg")
])

load_correlation = df_load_comparison.select([
    pl.corr("dam_energy_mw", "avg_sced_mw").alias("correlation")
]).item()

print(f"Load Resource - Correlation between DAM and SCED: {load_correlation:.4f}")
print()

print("Sample Load comparison:")
print(df_load_comparison.select([
    "DeliveryDate",
    "Hour Ending",
    "dam_energy_mw",
    "avg_sced_mw",
    "rt_delta_avg"
]).head(10))

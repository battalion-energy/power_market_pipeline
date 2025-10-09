#!/usr/bin/env python3
"""
CRITICAL CHECK: Verify SCED interval frequency

Are SCED BasePoints every 5 minutes or every 15 minutes?
This determines whether we use 5/60 or 15/60 in our calculations.
"""

import polars as pl
from pathlib import Path

ROLLUP_DIR = Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/rollup_files")

print("=" * 100)
print("VERIFYING SCED INTERVAL FREQUENCY")
print("=" * 100)
print()

# Load one hour of SCED data for one battery
df_sced = pl.read_parquet(ROLLUP_DIR / "SCED_Gen_Resources/2024.parquet").filter(
    pl.col("ResourceName") == "RRANCHES_UNIT2"
).with_columns([
    pl.col("SCEDTimeStamp").str.strptime(pl.Datetime, "%m/%d/%Y %H:%M:%S")
        .dt.replace_time_zone("America/Chicago")
        .dt.convert_time_zone("UTC")
        .alias("sced_dt")
]).filter(
    (pl.col("sced_dt").dt.month() == 5) &
    (pl.col("sced_dt").dt.year() == 2024) &
    (pl.col("sced_dt").dt.day() == 2) &
    (pl.col("sced_dt").dt.hour() == 20)  # Hour 20 UTC (afternoon in Texas)
).sort("sced_dt")

print("SCED intervals for May 2, 2024, Hour 20 UTC:")
print(df_sced.select(["SCEDTimeStamp", "sced_dt", "BasePoint"]))
print()

# Calculate time differences
df_intervals = df_sced.with_columns([
    pl.col("sced_dt").diff().dt.total_seconds().alias("seconds_since_prev")
])

print("Time differences between consecutive SCED intervals:")
print(df_intervals.select(["SCEDTimeStamp", "seconds_since_prev"]).head(20))
print()

# Count intervals per hour
total_intervals = len(df_sced)
print(f"Total SCED intervals in this hour: {total_intervals}")
print()

if total_intervals <= 4:
    print("✓ ~4 intervals per hour → SCED is 15-MINUTE")
    print("  Correct factor: 15/60 hours")
elif total_intervals >= 10:
    print("⚠️  ~12 intervals per hour → SCED is 5-MINUTE")
    print("  Correct factor: 5/60 hours")
    print()
    print("  CRITICAL ERROR IN CURRENT CODE:")
    print("  We're using 15/60 on 5-minute data → TRIPLE COUNTING!")
else:
    print(f"? Unclear - {total_intervals} intervals is unusual")

print()
print("=" * 100)
print("CHECKING RT PRICE INTERVALS")
print("=" * 100)
print()

# Check RT price intervals for same hour
df_rt = pl.read_parquet(ROLLUP_DIR / "RT_prices/2024.parquet").filter(
    pl.col("SettlementPointName") == "RRANCHES_ALL"
).with_columns([
    pl.from_epoch(pl.col("datetime"), time_unit='ms').alias("price_dt")
]).filter(
    (pl.col("price_dt").dt.month() == 5) &
    (pl.col("price_dt").dt.year() == 2024) &
    (pl.col("price_dt").dt.day() == 2) &
    (pl.col("price_dt").dt.hour() == 20)
).sort("price_dt")

print("RT price intervals for May 2, 2024, Hour 20 UTC:")
print(df_rt.select(["price_dt", "SettlementPointPrice"]))
print()

rt_intervals = len(df_rt)
print(f"Total RT price intervals in this hour: {rt_intervals}")

if rt_intervals == 4:
    print("✓ RT prices are 15-MINUTE intervals (as expected)")
else:
    print(f"? RT has {rt_intervals} intervals - check DST or data quality")

print()
print("=" * 100)
print("CORRECT CALCULATION METHOD")
print("=" * 100)
print()

if total_intervals >= 10:
    print("METHOD 1 (5-min granularity):")
    print("  For each 5-min SCED interval:")
    print("    revenue = BasePoint × RT_Price_15min × (5/60)")
    print("  Join: Round each 5-min timestamp to parent 15-min interval")
    print()
    print("METHOD 2 (aggregate to 15-min first):")
    print("  For each 15-min interval:")
    print("    MWh_15 = sum(BasePoint_5min × 5/60 for 3 intervals)")
    print("    revenue = MWh_15 × RT_Price_15min")
    print()
    print("  Both methods give same answer.")
else:
    print("SCED appears to be 15-min already:")
    print("  revenue = BasePoint × RT_Price × (15/60)")

print()
print("=" * 100)
print("CHECKING FOR SMNE FIELD")
print("=" * 100)
print()

# Check if SMNE exists in SCED Gen
df_gen_schema = pl.read_parquet(ROLLUP_DIR / "SCED_Gen_Resources/2024.parquet", n_rows=0)
print("SCED_Gen_Resources columns:")
for col in df_gen_schema.columns:
    print(f"  {col}")

print()
if "SMNE" in df_gen_schema.columns or any("Metered" in col for col in df_gen_schema.columns):
    print("✓ Found metered field - should use for discharge")
else:
    print("✗ No SMNE found - BasePoint is best available for discharge")

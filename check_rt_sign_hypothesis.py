#!/usr/bin/env python3
"""
Test hypothesis: Are RT charging/discharging economics SWITCHED?

Theory: In RT settlement, maybe:
- Load Resource BasePoint > 0 means REDUCING load → GET PAID (not pay)
- Gen Resource BasePoint > 0 means INCREASING gen → PAY (not get paid)

Or the mapping is backwards, or the sign convention is opposite.
"""

import polars as pl
from pathlib import Path

ROLLUP_DIR = Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/rollup_files")

# Test with RRANCHES_UNIT2 - should be profitable if sign is flipped
gen = "RRANCHES_UNIT2"
load = "RRANCHES_LD2"
node = "RRANCHES_ALL"

print("=" * 100)
print("TESTING RT SIGN HYPOTHESIS")
print("=" * 100)
print()
print(f"Battery: RRANCHES_UNIT2")
print(f"Gen Resource:  {gen}")
print(f"Load Resource: {load}")
print(f"Resource Node: {node}")
print()

# Load May 2024 data
print("Loading May 2024 data...")
df_gen = pl.read_parquet(ROLLUP_DIR / "SCED_Gen_Resources/2024.parquet").filter(
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
    pl.col("BasePoint").alias("gen_basepoint")
])

df_load = pl.read_parquet(ROLLUP_DIR / "SCED_Load_Resources/2024.parquet").filter(
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
    pl.col("BasePoint").alias("load_basepoint")
])

df_prices = pl.read_parquet(ROLLUP_DIR / "RT_prices/2024.parquet").filter(
    pl.col("SettlementPointName") == node
).with_columns([
    pl.from_epoch(pl.col("datetime"), time_unit='ms').alias("price_dt")
]).filter(
    (pl.col("price_dt").dt.month() == 5) & (pl.col("price_dt").dt.year() == 2024)
).select([
    pl.col("datetime").alias("price_datetime"),
    pl.col("SettlementPointPrice").alias("rt_price")
])

print(f"Gen intervals:   {len(df_gen):,}")
print(f"Load intervals:  {len(df_load):,}")
print(f"Price intervals: {len(df_prices):,}")
print()

# Join and calculate - CURRENT METHOD
print("=" * 100)
print("METHOD 1: CURRENT LOGIC")
print("  Gen BasePoint > 0 → Discharge → Revenue = MW × Price")
print("  Load BasePoint > 0 → Charge → Cost = MW × Price")
print("  Net = Discharge Revenue - Charge Cost")
print("=" * 100)

df_gen_joined = df_gen.with_columns([
    pl.col("sced_dt").dt.truncate("15m").dt.epoch("ms").alias("rounded_datetime")
]).join(
    df_prices,
    left_on="rounded_datetime",
    right_on="price_datetime",
    how="left"
).filter(pl.col("rt_price").is_not_null())

df_load_joined = df_load.with_columns([
    pl.col("sced_dt").dt.truncate("15m").dt.epoch("ms").alias("rounded_datetime")
]).join(
    df_prices,
    left_on="rounded_datetime",
    right_on="price_datetime",
    how="left"
).filter(pl.col("rt_price").is_not_null())

discharge_revenue = df_gen_joined.select(
    (pl.col("gen_basepoint") * pl.col("rt_price") * 15.0 / 60.0).sum()
).item()

charge_cost = df_load_joined.select(
    (pl.col("load_basepoint") * pl.col("rt_price") * 15.0 / 60.0).sum()
).item()

rt_net_current = discharge_revenue - charge_cost

print(f"Discharge Revenue: ${discharge_revenue:,.2f}")
print(f"Charge Cost:       ${charge_cost:,.2f}")
print(f"RT Net:            ${rt_net_current:,.2f}")
print()

# HYPOTHESIS 1: Flip the signs (charge becomes revenue, discharge becomes cost)
print("=" * 100)
print("METHOD 2: FLIPPED SIGN HYPOTHESIS")
print("  Gen BasePoint > 0 → Discharge → COST = MW × Price (PAY to discharge?)")
print("  Load BasePoint > 0 → Charge → REVENUE = MW × Price (GET PAID to charge?)")
print("  Net = Charge Revenue - Discharge Cost")
print("=" * 100)

rt_net_flipped = charge_cost - discharge_revenue  # Flip the subtraction

print(f"Charge Revenue:    ${charge_cost:,.2f}")
print(f"Discharge Cost:    ${discharge_revenue:,.2f}")
print(f"RT Net:            ${rt_net_flipped:,.2f}")
print()

# HYPOTHESIS 2: Resources are swapped (Gen is actually charging, Load is actually discharging)
print("=" * 100)
print("METHOD 3: SWAPPED RESOURCE HYPOTHESIS")
print("  Gen Resource is actually CHARGING (not discharging)")
print("  Load Resource is actually DISCHARGING (not charging)")
print("  Net = Load Revenue - Gen Cost")
print("=" * 100)

rt_net_swapped = charge_cost - discharge_revenue  # Same calculation but different interpretation

print(f"Discharge Revenue (from Load): ${charge_cost:,.2f}")
print(f"Charge Cost (from Gen):        ${discharge_revenue:,.2f}")
print(f"RT Net:                        ${rt_net_swapped:,.2f}")
print()

# HYPOTHESIS 3: Both are revenues (opposite signs in settlement)
print("=" * 100)
print("METHOD 4: BOTH AS REVENUES HYPOTHESIS")
print("  Gen BasePoint > 0 → Revenue (positive)")
print("  Load BasePoint > 0 → Revenue (negative, paying for energy)")
print("  Net = Gen Revenue + Load Revenue (where Load Revenue is negative)")
print("=" * 100)

# In this model, charge_cost should be negative
rt_net_both_revenue = discharge_revenue + (-charge_cost)

print(f"Gen Revenue:       ${discharge_revenue:,.2f}")
print(f"Load Revenue:      ${-charge_cost:,.2f}")
print(f"RT Net:            ${rt_net_both_revenue:,.2f}")
print()

# Summary
print("=" * 100)
print("SUMMARY - Which makes sense?")
print("=" * 100)
print(f"Current Method:        ${rt_net_current:,.2f}      (NEGATIVE - battery loses money)")
print(f"Flipped Sign:          ${rt_net_flipped:,.2f}     (POSITIVE - battery makes money!)")
print(f"Swapped Resources:     ${rt_net_swapped:,.2f}     (Same as flipped)")
print(f"Both as Revenues:      ${rt_net_both_revenue:,.2f}      (Same as current)")
print()

print("If flipped sign is correct, this battery would be PROFITABLE in RT!")
print()

# Check a few specific intervals to see the pattern
print("=" * 100)
print("SAMPLE INTERVALS - Looking for pattern")
print("=" * 100)

# Find intervals where both gen and load have non-zero basepoints
df_both = df_gen_joined.join(
    df_load_joined.select(["sced_dt", "load_basepoint"]),
    on="sced_dt",
    how="inner"
).filter(
    (pl.col("gen_basepoint") > 0) & (pl.col("load_basepoint") > 0)
).head(20)

if len(df_both) > 0:
    print("Intervals where BOTH Gen and Load have positive BasePoint:")
    print(df_both.select([
        "SCEDTimeStamp",
        "gen_basepoint",
        "load_basepoint",
        "rt_price",
        (pl.col("gen_basepoint") * pl.col("rt_price") * 15/60).alias("gen_$"),
        (pl.col("load_basepoint") * pl.col("rt_price") * 15/60).alias("load_$")
    ]))
else:
    print("No intervals where both are positive - checking high discharge intervals...")

    df_high_discharge = df_gen_joined.filter(
        pl.col("gen_basepoint") > 50
    ).head(10)

    print("High discharge intervals (Gen BasePoint > 50 MW):")
    print(df_high_discharge.select([
        "SCEDTimeStamp",
        "gen_basepoint",
        "rt_price",
        (pl.col("gen_basepoint") * pl.col("rt_price") * 15/60).alias("gen_$")
    ]))

    print()

    df_high_charge = df_load_joined.filter(
        pl.col("load_basepoint") > 50
    ).head(10)

    print("High charge intervals (Load BasePoint > 50 MW):")
    print(df_high_charge.select([
        "SCEDTimeStamp",
        "load_basepoint",
        "rt_price",
        (pl.col("load_basepoint") * pl.col("rt_price") * 15/60).alias("load_$")
    ]))

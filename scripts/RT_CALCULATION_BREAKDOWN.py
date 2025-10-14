#!/usr/bin/env python3
"""
RT Revenue Calculation Breakdown - Detailed Analysis with Real Data

This script extracts ACTUAL data from parquet files to show EXACTLY how
RT revenue is calculated for specific batteries in specific time periods.

NO FAKE DATA - All values come directly from ERCOT parquet files.
"""

import polars as pl
from pathlib import Path
from datetime import datetime

# Data directory
ROLLUP_DIR = Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/rollup_files")

# Example batteries to analyze
EXAMPLES = [
    {
        "name": "RRANCHES_UNIT2",
        "gen": "RRANCHES_UNIT2",
        "load": "RRANCHES_LD2",
        "node": "RRANCHES_ALL",
        "capacity": 150.0,
        "note": "Top RT performer - positive arbitrage"
    },
    {
        "name": "CHISMGRD_BES1",
        "gen": "CHISMGRD_BES1",
        "load": "CHISMGRD_LD1",
        "node": "CHISMGRD_RN",
        "capacity": 9.99,
        "note": "Worst RT performer - massive negative arbitrage"
    },
    {
        "name": "BATCAVE_BES1",
        "gen": "BATCAVE_BES1",
        "load": "BATCAVE_LD1",
        "node": "BATCAVE_RN",
        "capacity": 155.2,
        "note": "Negative RT, >100% efficiency"
    }
]

print("=" * 100)
print("RT REVENUE CALCULATION - COMPLETE BREAKDOWN")
print("=" * 100)
print()

print("## 1. DATA SOURCES")
print()
print("### File Locations:")
print(f"   SCED_Gen_Resources:  {ROLLUP_DIR / 'SCED_Gen_Resources/2024.parquet'}")
print(f"   SCED_Load_Resources: {ROLLUP_DIR / 'SCED_Load_Resources/2024.parquet'}")
print(f"   RT_prices:           {ROLLUP_DIR / 'RT_prices/2024.parquet'}")
print()

# Check file schemas
print("### SCED_Gen_Resources Schema:")
df_gen_sample = pl.read_parquet(ROLLUP_DIR / "SCED_Gen_Resources/2024.parquet", n_rows=0)
print(f"   Columns: {df_gen_sample.columns}")
print()

print("### SCED_Load_Resources Schema:")
df_load_sample = pl.read_parquet(ROLLUP_DIR / "SCED_Load_Resources/2024.parquet", n_rows=0)
print(f"   Columns: {df_load_sample.columns}")
print()

print("### RT_prices Schema:")
df_price_sample = pl.read_parquet(ROLLUP_DIR / "RT_prices/2024.parquet", n_rows=0)
print(f"   Columns: {df_price_sample.columns}")
print()

print("=" * 100)
print("## 2. CALCULATION METHODOLOGY")
print("=" * 100)
print()
print("### Step 1: Load SCED Discharge Data (Gen Resource)")
print("   - Source: SCED_Gen_Resources parquet")
print("   - Filter: ResourceName == 'GEN_RESOURCE_NAME'")
print("   - Extract: SCEDTimeStamp, BasePoint (MW discharge)")
print("   - Timezone: Parse as Central Time, convert to UTC")
print()

print("### Step 2: Load SCED Charging Data (Load Resource)")
print("   - Source: SCED_Load_Resources parquet")
print("   - Filter: ResourceName == 'LOAD_RESOURCE_NAME'")
print("   - Extract: SCEDTimeStamp, BasePoint (MW charge)")
print("   - Timezone: Parse as Central Time, convert to UTC")
print()

print("### Step 3: Load RT Prices")
print("   - Source: RT_prices parquet")
print("   - Filter: SettlementPointName == 'RESOURCE_NODE'")
print("   - Extract: datetime (UTC), SettlementPointPrice ($/MWh)")
print("   - Granularity: 15-minute intervals")
print()

print("### Step 4: Join Discharge with Prices")
print("   - Round SCED timestamp to nearest 15-min")
print("   - Join on rounded timestamp")
print("   - Calculate: discharge_revenue = Σ(discharge_MW × price × 15/60)")
print()

print("### Step 5: Join Charging with Prices")
print("   - Round SCED timestamp to nearest 15-min")
print("   - Join on rounded timestamp")
print("   - Calculate: charge_cost = Σ(charge_MW × price × 15/60)")
print()

print("### Step 6: Calculate Net Revenue")
print("   - RT_net = discharge_revenue - charge_cost")
print("   - Efficiency = discharge_MWh / charge_MWh")
print()

print("=" * 100)
print("## 3. DETAILED EXAMPLES - ACTUAL DATA")
print("=" * 100)
print()

for example in EXAMPLES:
    print(f"\n{'=' * 100}")
    print(f"### {example['name']}: {example['note']}")
    print(f"{'=' * 100}")
    print(f"Gen Resource:   {example['gen']}")
    print(f"Load Resource:  {example['load']}")
    print(f"Resource Node:  {example['node']}")
    print(f"Capacity:       {example['capacity']} MW")
    print()

    # Load data for this battery
    print("Loading SCED discharge data...")
    df_gen = pl.read_parquet(ROLLUP_DIR / "SCED_Gen_Resources/2024.parquet").filter(
        pl.col("ResourceName") == example['gen']
    ).with_columns([
        pl.col("SCEDTimeStamp").str.strptime(pl.Datetime, "%m/%d/%Y %H:%M:%S")
            .dt.replace_time_zone("America/Chicago")
            .dt.convert_time_zone("UTC")
            .alias("sced_dt")
    ]).filter(
        pl.col("sced_dt").dt.year() == 2024
    ).select([
        "SCEDTimeStamp",
        "sced_dt",
        pl.col("BasePoint").alias("discharge_mw")
    ])

    print(f"   Total discharge intervals: {len(df_gen):,}")
    print(f"   Date range: {df_gen.select(pl.col('sced_dt').min()).item()} to {df_gen.select(pl.col('sced_dt').max()).item()}")
    print()

    print("Loading SCED charging data...")
    df_load = pl.read_parquet(ROLLUP_DIR / "SCED_Load_Resources/2024.parquet").filter(
        pl.col("ResourceName") == example['load']
    ).with_columns([
        pl.col("SCEDTimeStamp").str.strptime(pl.Datetime, "%m/%d/%Y %H:%M:%S")
            .dt.replace_time_zone("America/Chicago")
            .dt.convert_time_zone("UTC")
            .alias("sced_dt")
    ]).filter(
        pl.col("sced_dt").dt.year() == 2024
    ).select([
        "SCEDTimeStamp",
        "sced_dt",
        pl.col("BasePoint").alias("charge_mw")
    ])

    print(f"   Total charging intervals: {len(df_load):,}")
    print(f"   Date range: {df_load.select(pl.col('sced_dt').min()).item()} to {df_load.select(pl.col('sced_dt').max()).item()}")
    print()

    print("Loading RT prices...")
    df_prices = pl.read_parquet(ROLLUP_DIR / "RT_prices/2024.parquet").filter(
        pl.col("SettlementPointName") == example['node']
    ).select([
        pl.col("datetime").alias("price_datetime"),
        pl.col("SettlementPointPrice").alias("rt_price")
    ])

    print(f"   Total price intervals: {len(df_prices):,}")
    if len(df_prices) > 0:
        # Convert epoch ms to datetime for display
        price_min_dt = df_prices.select(pl.from_epoch(pl.col('price_datetime'), time_unit='ms')).min().item()
        price_max_dt = df_prices.select(pl.from_epoch(pl.col('price_datetime'), time_unit='ms')).max().item()
        print(f"   Date range: {price_min_dt} to {price_max_dt}")
        price_stats = df_prices.select([
            pl.col("rt_price").min().alias("min"),
            pl.col("rt_price").mean().alias("avg"),
            pl.col("rt_price").max().alias("max")
        ])
        print(f"   Price range: ${price_stats['min'][0]:.2f} to ${price_stats['max'][0]:.2f}/MWh (avg ${price_stats['avg'][0]:.2f})")
    print()

    # Focus on a specific month - May 2024 (high revenue month per benchmark)
    print("=" * 80)
    print("FOCUS: MAY 2024 ANALYSIS (Industry benchmark: $13.50/kW-month)")
    print("=" * 80)
    print()

    # Filter to May 2024
    df_gen_may = df_gen.filter(
        (pl.col("sced_dt").dt.month() == 5) & (pl.col("sced_dt").dt.year() == 2024)
    )

    df_load_may = df_load.filter(
        (pl.col("sced_dt").dt.month() == 5) & (pl.col("sced_dt").dt.year() == 2024)
    )

    # Convert price_datetime from epoch ms to datetime for filtering
    df_prices_may = df_prices.with_columns([
        pl.from_epoch(pl.col("price_datetime"), time_unit='ms').alias("price_dt")
    ]).filter(
        (pl.col("price_dt").dt.month() == 5) & (pl.col("price_dt").dt.year() == 2024)
    )

    print(f"May 2024 intervals:")
    print(f"   Discharge: {len(df_gen_may):,}")
    print(f"   Charge:    {len(df_load_may):,}")
    print(f"   Prices:    {len(df_prices_may):,}")
    print()

    if len(df_gen_may) > 0:
        # Join with prices
        df_gen_may = df_gen_may.with_columns([
            pl.col("sced_dt").dt.truncate("15m").dt.epoch("ms").alias("rounded_datetime")
        ]).join(
            df_prices_may,
            left_on="rounded_datetime",
            right_on="price_datetime",
            how="left"
        )

        # Show sample rows
        print("Sample Discharge Intervals (first 10):")
        print(df_gen_may.head(10).select([
            "SCEDTimeStamp",
            "discharge_mw",
            "rt_price",
            (pl.col("discharge_mw") * pl.col("rt_price") * 15.0 / 60.0).alias("revenue_$")
        ]))
        print()

        # Calculate May discharge revenue
        df_gen_may_valid = df_gen_may.filter(pl.col("rt_price").is_not_null())
        discharge_revenue_may = df_gen_may_valid.select(
            (pl.col("discharge_mw") * pl.col("rt_price") * 15.0 / 60.0).sum()
        ).item()
        discharge_mwh_may = df_gen_may_valid.select(
            (pl.col("discharge_mw") * 15.0 / 60.0).sum()
        ).item()

        print(f"May Discharge Summary:")
        print(f"   Total MWh:     {discharge_mwh_may:,.1f}")
        print(f"   Total Revenue: ${discharge_revenue_may:,.2f}")
        print(f"   Avg Price:     ${discharge_revenue_may / discharge_mwh_may:.2f}/MWh" if discharge_mwh_may > 0 else "")
        print()

    if len(df_load_may) > 0:
        # Join with prices
        df_load_may = df_load_may.with_columns([
            pl.col("sced_dt").dt.truncate("15m").dt.epoch("ms").alias("rounded_datetime")
        ]).join(
            df_prices_may,
            left_on="rounded_datetime",
            right_on="price_datetime",
            how="left"
        )

        # Show sample rows
        print("Sample Charging Intervals (first 10):")
        print(df_load_may.head(10).select([
            "SCEDTimeStamp",
            "charge_mw",
            "rt_price",
            (pl.col("charge_mw") * pl.col("rt_price") * 15.0 / 60.0).alias("cost_$")
        ]))
        print()

        # Calculate May charge cost
        df_load_may_valid = df_load_may.filter(pl.col("rt_price").is_not_null())
        charge_cost_may = df_load_may_valid.select(
            (pl.col("charge_mw") * pl.col("rt_price") * 15.0 / 60.0).sum()
        ).item()
        charge_mwh_may = df_load_may_valid.select(
            (pl.col("charge_mw") * 15.0 / 60.0).sum()
        ).item()

        print(f"May Charging Summary:")
        print(f"   Total MWh:  {charge_mwh_may:,.1f}")
        print(f"   Total Cost: ${charge_cost_may:,.2f}")
        print(f"   Avg Price:  ${charge_cost_may / charge_mwh_may:.2f}/MWh" if charge_mwh_may > 0 else "")
        print()

    if len(df_gen_may) > 0 and len(df_load_may) > 0:
        rt_net_may = discharge_revenue_may - charge_cost_may
        efficiency_may = discharge_mwh_may / charge_mwh_may if charge_mwh_may > 0 else 0

        print(f"May RT NET:")
        print(f"   Discharge Revenue: ${discharge_revenue_may:,.2f}")
        print(f"   Charge Cost:       ${charge_cost_may:,.2f}")
        print(f"   Net:               ${rt_net_may:,.2f}")
        print(f"   Efficiency:        {efficiency_may * 100:.1f}%")
        print(f"   $/MW-month:        ${rt_net_may / example['capacity']:,.2f}")
        print()
        print(f"   Benchmark comparison: ${13.50 * example['capacity'] * 1000:,.2f} expected")
        print()

print()
print("=" * 100)
print("## 4. KEY QUESTIONS TO EVALUATE")
print("=" * 100)
print()
print("1. Is the RT price data at the correct granularity (15-min)?")
print("2. Are we joining SCED timestamps correctly with RT price timestamps?")
print("3. Should we use BasePoint or some other field from SCED?")
print("4. Are Gen/Load resources correctly mapped?")
print("5. Why are some batteries showing negative RT arbitrage?")
print("6. Why are some showing >100% efficiency?")
print("7. Are we missing any revenue components (deployment, mileage)?")
print()

#!/usr/bin/env python3
"""
Utilities for integrating ERCOT-wide BESS rollup metrics into the master feature set.

This does not modify files directly—use the functions from create_enhanced_master_with_net_load.py
or other pipelines to consume the hourly aggregates and merge them into the master dataframe.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import polars as pl

RT_ROLLUP_PATH = Path("/home/enrico/projects/power_market_pipeline/output/ercot_wide_bess_rt_15min_all_years.parquet")
DAM_ROLLUP_PATH = Path("/home/enrico/projects/power_market_pipeline/output/ercot_wide_bess_dam_hourly_all_years.parquet")


def load_rt_bess_hourly(file: Path = RT_ROLLUP_PATH) -> pl.DataFrame:
    """Load 15-minute BESS RT rollup and aggregate to hourly metrics."""
    if not file.exists():
        raise FileNotFoundError(file)
    lf = pl.scan_parquet(str(file))
    df = (
        lf.group_by(pl.col("timestamp_15min").dt.truncate("1h").alias("timestamp"))
        .agg(
            [
                pl.col("rt_discharge_mw_total").sum().alias("bess_rt_discharge_mw_total"),
                pl.col("rt_charge_mw_total").sum().alias("bess_rt_charge_mw_total"),
                pl.col("rt_discharge_mwh").sum().alias("bess_rt_discharge_mwh"),
                pl.col("rt_charge_mwh").sum().alias("bess_rt_charge_mwh"),
                pl.col("rt_discharge_revenue").sum().alias("bess_rt_discharge_revenue"),
                pl.col("rt_charge_cost").sum().alias("bess_rt_charge_cost"),
                pl.col("rt_net_revenue").sum().alias("bess_rt_net_revenue"),
                pl.col("bess_count_gen_rt").max().alias("bess_rt_count_gen"),
                pl.col("bess_count_load_rt").max().alias("bess_rt_count_load"),
            ]
        )
        .sort("timestamp")
        .with_columns(pl.col("timestamp").dt.replace_time_zone(None))
        .collect()
    )
    return df


def load_dam_bess_hourly(file: Path = DAM_ROLLUP_PATH) -> pl.DataFrame:
    """Load hourly DAM rollup and map to standard column names."""
    if not file.exists():
        raise FileNotFoundError(file)
    df = (
        pl.read_parquet(str(file))
        .select(
            [
                pl.col("timestamp_ct").dt.replace_time_zone(None).alias("timestamp"),
                pl.col("dam_discharge_mwh").alias("bess_dam_discharge_mwh"),
                pl.col("dam_load_regup_mw").alias("bess_dam_load_regup_mw"),
                pl.col("dam_load_rrs_mw").alias("bess_dam_load_rrs_mw"),
                pl.col("dam_load_ecrs_mw").alias("bess_dam_load_ecrs_mw"),
                pl.col("dam_load_nonspin_mw").alias("bess_dam_load_nonspin_mw"),
                pl.col("dam_gen_regup_mw").alias("bess_dam_gen_regup_mw"),
                pl.col("dam_gen_rrs_mw").alias("bess_dam_gen_rrs_mw"),
                pl.col("dam_gen_ecrs_mw").alias("bess_dam_gen_ecrs_mw"),
                pl.col("dam_gen_nonspin_mw").alias("bess_dam_gen_nonspin_mw"),
                pl.col("dam_gen_regup_revenue").alias("bess_dam_gen_regup_revenue"),
                pl.col("dam_gen_rrs_revenue").alias("bess_dam_gen_rrs_revenue"),
                pl.col("dam_gen_ecrs_revenue").alias("bess_dam_gen_ecrs_revenue"),
                pl.col("dam_gen_nonspin_revenue").alias("bess_dam_gen_nonspin_revenue"),
            ]
        )
        .sort("timestamp")
    )
    return df


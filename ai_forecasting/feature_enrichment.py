#!/usr/bin/env python3
"""
Feature enrichment utilities for price-forecast training scripts.

These helpers merge the corrected net-load dataset and the ERCOT-wide BESS rollups
onto the base master feature tables so every model trains on the latest signals.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import polars as pl

MASTER_DEFAULT = Path(
    "/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/"
    "master_enhanced_with_net_load_reserves_2019_2025.parquet"
)

NET_LOAD_FIXED = Path(
    "/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/processed/"
    "net_load_features_2018_2025_FIXED.parquet"
)

BESS_RT_ROLLUP = Path(
    "/home/enrico/projects/power_market_pipeline/output/"
    "ercot_wide_bess_rt_15min_all_years.parquet"
)

NET_LOAD_COLUMNS: Sequence[str] = [
    "actual_system_load_MW",
    "wind_generation_MW",
    "solar_generation_MW",
    "net_load_MW",
    "renewable_penetration_pct",
    "net_load_ramp_1h",
    "net_load_ramp_3h",
    "net_load_roll_24h_mean",
    "net_load_roll_24h_std",
    "net_load_roll_24h_max",
    "net_load_roll_24h_min",
    "houston_pct_of_net_load",
    "north_pct_of_net_load",
    "high_renewable_flag",
    "large_ramp_flag",
    "low_net_load_flag",
]

BESS_RT_COLUMNS: Sequence[str] = [
    "bess_rt_discharge_mw_total",
    "bess_rt_charge_mw_total",
    "bess_rt_discharge_mwh",
    "bess_rt_charge_mwh",
    "bess_rt_discharge_revenue",
    "bess_rt_charge_cost",
    "bess_rt_net_revenue",
    "bess_rt_count_gen",
    "bess_rt_count_load",
]


def _load_master(base_path: Path) -> pd.DataFrame:
    agg = (
        pl.read_parquet(str(base_path), columns=None)
        .with_columns(pl.col("timestamp").cast(pl.Datetime("ns")))
        .group_by("timestamp")
        .mean()
        .sort("timestamp")
    )
    df = agg.to_pandas()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def _load_net_load() -> pd.DataFrame:
    df = (
        pl.read_parquet(str(NET_LOAD_FIXED))
        .with_columns(pl.col("timestamp").cast(pl.Datetime("ns")))
        .sort("timestamp")
        .to_pandas()
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.drop_duplicates("timestamp").reset_index(drop=True)
    return df


def _load_rt_bess_hourly() -> pd.DataFrame:
    if not BESS_RT_ROLLUP.exists():
        raise FileNotFoundError(BESS_RT_ROLLUP)
    lf = pl.scan_parquet(str(BESS_RT_ROLLUP))
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
        .with_columns(pl.col("timestamp").cast(pl.Datetime("ns")))
        .collect()
        .to_pandas()
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.drop_duplicates("timestamp").reset_index(drop=True)
    return df


def load_enriched_master_dataframe(base_path: Path = MASTER_DEFAULT) -> pd.DataFrame:
    """Return a pandas DataFrame with corrected net load and BESS rollup features merged in."""
    base = _load_master(base_path)

    # Merge corrected net load (override existing columns with fixed values)
    net = _load_net_load()
    for col in NET_LOAD_COLUMNS:
        if col in base.columns:
            base.drop(columns=col, inplace=True)
    base = base.merge(net, on="timestamp", how="left")

    # Merge aggregated BESS fleet metrics (fill missing with zeros)
    bess = _load_rt_bess_hourly()
    base = base.merge(bess, on="timestamp", how="left")
    for col in BESS_RT_COLUMNS:
        if col in base.columns:
            base[col] = base[col].fillna(0.0)

    base.sort_values("timestamp", inplace=True)
    base.reset_index(drop=True, inplace=True)
    return base

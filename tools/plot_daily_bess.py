#!/usr/bin/env python3
"""
Plot daily BESS charts from hourly parquet outputs (awards and dispatch).

Usage:
  python tools/plot_daily_bess.py --base-dir /pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data \
      --bess CHISMGRD_BES1 --year 2024 --date 2024-02-20

Outputs two-panel PNG under tools/output/plots/<BESS>_<date>_daily.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt


def load_awards(base: Path, bess: str, year: int) -> pl.DataFrame:
    p = base / "bess_analysis" / "hourly" / "awards" / f"{bess}_{year}_awards.parquet"
    return pl.read_parquet(p) if p.exists() else pl.DataFrame()


def load_dispatch(base: Path, bess: str, year: int) -> pl.DataFrame:
    p = base / "bess_analysis" / "hourly" / "dispatch" / f"{bess}_{year}_dispatch.parquet"
    return pl.read_parquet(p) if p.exists() else pl.DataFrame()


def main(args):
    base = Path(args.base_dir)
    awards = load_awards(base, args.bess, args.year)
    dispatch = load_dispatch(base, args.bess, args.year)
    if len(awards) == 0 or len(dispatch) == 0:
        raise FileNotFoundError("Hourly awards/dispatch parquet not found. Generate with bess_revenue_calculator first.")

    # Filter to date
    target_date = pd.to_datetime(args.date).date()
    aw_d = awards.filter(pl.col("local_date") == target_date).sort("local_hour")
    dp_d = dispatch.filter(pl.col("local_date") == target_date).sort("local_hour")

    if len(aw_d) == 0 and len(dp_d) == 0:
        raise ValueError("No data for the selected date.")

    hours = dp_d["local_hour"].to_list() if len(dp_d) > 0 else aw_d["local_hour"].to_list()

    # Build figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Panel 1: Market Awards stacked bars
    if len(aw_d) > 0:
        df_aw = aw_d.to_pandas()
        h = df_aw["local_hour"].values
        stacks = {
            "NonSpin": (df_aw.get("nonspin_mw", 0.0).values, "#2C3E50"),
            "ECRS": (df_aw.get("ecrs_mw", 0.0).values, "#D7BDE2"),
            "RRS": (df_aw.get("rrs_mw", 0.0).values, "#5DADE2"),
            "RegDown": (df_aw.get("regdown_mw", 0.0).values, "#F4A6A6"),
            "RegUp": (df_aw.get("regup_mw", 0.0).values, "#F4E76E"),
            "DA Energy": (df_aw.get("da_energy_award_mw", 0.0).values, "#8B7BC8"),
        }
        bottom = None
        for label, (vals, color) in reversed(list(stacks.items())):
            if bottom is None:
                ax1.bar(h, vals, color=color, label=label)
                bottom = vals
            else:
                ax1.bar(h, vals, bottom=bottom, color=color, label=label)
                bottom = bottom + vals
        ax1.set_title(f"{target_date:%-m/%-d} Market Awards | {args.bess}")
        ax1.set_xlabel("Hour")
        ax1.set_ylabel("Market Awards (MW)")
        ax1.legend(loc="upper left", fontsize=8)

    # Panel 2: Dispatch Profile + Prices
    if len(dp_d) > 0:
        df_dp = dp_d.to_pandas()
        h = df_dp["local_hour"].values
        # Bars: net actual MWh
        ax2.bar(h, df_dp["net_actual_mwh"].values, color="#5FD4AF", alpha=0.6, label="Output (MWh)")
        # Line: RT basepoint net (bp_gen - bp_load) MWh
        bp_net_mwh = (df_dp["basepoint_gen_mwh"].values - df_dp["basepoint_load_mwh"].values)
        ax2.bar(h, bp_net_mwh, color="#2ECC71", alpha=0.4, label="RT Basepoint (MWh)")
        # Lines: RT and DA energy prices (right axis)
        ax3 = ax2.twinx()
        if 'rt_price_avg' in df_dp.columns:
            ax3.plot(h, df_dp['rt_price_avg'].values, color='#34495E', label='RT Price', linewidth=1.8)
        if 'da_price_hour' in df_dp.columns:
            ax3.plot(h, df_dp['da_price_hour'].values, color='#8B7BC8', label='DA Price', linewidth=1.4)
        # Ancillary MCPCs from awards parquet
        if len(aw_d) > 0:
            df_awp = aw_d.to_pandas()
            if 'ecrs_mcpc' in df_awp.columns:
                ax3.plot(h, df_awp['ecrs_mcpc'].values, color='#1F618D', label='ECRS MCPC', linewidth=1.2)
            if 'rrs_mcpc' in df_awp.columns:
                ax3.plot(h, df_awp['rrs_mcpc'].values, color='#5DADE2', label='RRS MCPC', linewidth=1.0)
            if 'regup_mcpc' in df_awp.columns:
                ax3.plot(h, df_awp['regup_mcpc'].values, color='#F4E76E', label='RegUp MCPC', linewidth=1.0)
            if 'regdown_mcpc' in df_awp.columns:
                ax3.plot(h, df_awp['regdown_mcpc'].values, color='#F4A6A6', label='RegDown MCPC', linewidth=1.0)
            if 'nonspin_mcpc' in df_awp.columns:
                ax3.plot(h, df_awp['nonspin_mcpc'].values, color='#2C3E50', label='NonSpin MCPC', linewidth=1.0)
        ax2.set_title(f"{target_date:%-m/%-d} Dispatch Profile | {args.bess}")
        ax2.set_xlabel("Hour")
        ax2.set_ylabel("MWh")
        ax2.legend(loc="upper left", fontsize=8)
        ax3.set_ylabel("Price ($/MWh)")
        ax3.legend(loc='upper right', fontsize=8)

    out_dir = Path(__file__).parent / "output" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.bess}_{args.date}_daily.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", required=True)
    p.add_argument("--bess", required=True)
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--date", required=True)
    args = p.parse_args()
    main(args)

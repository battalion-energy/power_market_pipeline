#!/usr/bin/env python3
"""
Create annual nodal TB2 charts without rebuilding the whole dataset.

Reads a single TB2 parquet (e.g., tb2_daily_2019_2025.parquet) or a
partitioned directory of daily/yearly TB2 parquets and renders a
bar chart of nodal TB2 for a given year.

Usage examples:
  python create_tb2_annual_chart.py --year 2025
  python create_tb2_annual_chart.py --year 2025 --tb2 tb2_daily_2019_2025.parquet --top 100
  python create_tb2_annual_chart.py --year 2025 --tb2 data/tb2_daily_dataset
"""

from __future__ import annotations

import argparse
from pathlib import Path
import polars as pl
import matplotlib.pyplot as plt
import numpy as np


def load_tb2(tb2_path: Path) -> pl.DataFrame:
    if tb2_path.is_dir():
        # Read a partitioned dataset (any parquet files under the directory)
        lf = pl.scan_parquet(str(tb2_path / "**/*.parquet"), recurse=True)
        return lf.collect()
    else:
        return pl.read_parquet(tb2_path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Annual nodal TB2 chart")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--tb2", default=None, help="TB2 parquet file or directory (default: auto-detect)")
    ap.add_argument("--top", type=int, default=50, help="Top N nodes to plot (by TB2 sum)")
    ap.add_argument("--outdir", default="tb2_charts", help="Output directory for charts & CSV")
    args = ap.parse_args()

    # Pick a TB2 source
    candidates = []
    if args.tb2:
        candidates.append(Path(args.tb2))
    candidates += [
        Path("tb2_daily_2019_2025.parquet"),
        Path("tb2_daily_2019_2024.parquet"),
        Path("tb2_daily.parquet"),
        Path("tb2_daily"),
    ]
    tb2_path = next((p for p in candidates if (p.is_file() or p.is_dir())), None)
    if tb2_path is None:
        raise SystemExit("No TB2 parquet or dataset directory found.")

    df = load_tb2(tb2_path)

    # Filter to year (by DeliveryDate; fallback to 'year' column if present)
    if "DeliveryDate" in df.columns:
        df_year = df.filter(pl.col("DeliveryDate").dt.year() == args.year)
    elif "year" in df.columns:
        df_year = df.filter(pl.col("year") == args.year)
    else:
        raise SystemExit("TB2 parquet missing DeliveryDate/year columns")

    if len(df_year) == 0:
        raise SystemExit(f"No TB2 data found for {args.year}")

    # Aggregate per node for the year
    agg = (
        df_year.group_by("SettlementPoint")
        .agg([
            pl.col("TB2").sum().alias("tb2_sum"),
            pl.col("TB2").mean().alias("tb2_avg"),
            pl.len().alias("days"),
        ])
        .sort("tb2_sum", descending=True)
    )

    # Save CSV and render chart for top N
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f"tb2_nodal_{args.year}.csv"
    agg.write_csv(csv_path)

    top = agg.head(args.top).to_pandas()

    # Plot
    fig, ax = plt.subplots(figsize=(24, 10))
    x = np.arange(len(top))
    ax.bar(x, top["tb2_sum"], width=0.8, color="#5DADE2", label="TB2 sum ($/MWh·day)")
    ax.set_xticks(x)
    ax.set_xticklabels(top["SettlementPoint"], rotation=90, ha="right", fontsize=8)
    ax.set_ylabel("Annual TB2 (sum of daily spreads)")
    ax.set_title(f"{args.year} ERCOT Nodal TB2 — Top {len(top)} Nodes")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(loc="upper right")
    out_png = outdir / f"tb2_nodal_{args.year}_top{len(top)}.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")

    print(f"Saved: {out_png}")
    print(f"Saved: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


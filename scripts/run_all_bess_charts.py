#!/usr/bin/env python3
"""
Run all BESS chart generators in one go:
 - Annual stacked chart: create_bess_revenue_chart.py
 - Quarterly & monthly period charts: create_bess_revenue_charts_quarterly_monthly.py
 - Sampled daily plots: tools/batch_daily_plots.py

Usage examples:
  python scripts/run_all_bess_charts.py \
    --base-dir /pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data \
    --years 2024 2023 2022 \
    --all-bess \
    --per-month 5 --with-15min --with-advanced

  python scripts/run_all_bess_charts.py --skip-daily  # Only annual + period charts
"""

from __future__ import annotations

import argparse
import logging
import shlex
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str] | str) -> int:
    if isinstance(cmd, str):
        pretty = cmd
        real_cmd = cmd
        shell = True
    else:
        pretty = " ".join(shlex.quote(x) for x in cmd)
        real_cmd = cmd
        shell = False
    logging.info("Running: %s", pretty)
    res = subprocess.run(real_cmd, shell=shell)
    if res.returncode != 0:
        logging.warning("Command failed (rc=%s): %s", res.returncode, pretty)
    return res.returncode


def main() -> int:
    p = argparse.ArgumentParser(description="Run annual, quarterly/monthly, and sampled daily BESS charts")
    p.add_argument("--base-dir", help="Base ERCOT data directory (required for daily plots)")
    p.add_argument("--years", nargs="*", type=int, default=[], help="Years to plot for sampled daily (e.g., 2024 2023)")
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--all-bess", action="store_true", help="Plot sampled daily for all BESS with available parquets")
    grp.add_argument("--bess", help="Single BESS gen resource name for sampled daily")
    p.add_argument("--per-month", type=int, default=5, help="Sampled days per month for daily plots")
    p.add_argument("--with-15min", action="store_true", help="Also render 15-minute daily chart variant")
    p.add_argument("--with-advanced", action="store_true", help="Also render advanced 4-panel daily chart")
    p.add_argument("--skip-annual", action="store_true", help="Skip annual stacked chart")
    p.add_argument("--skip-periods", action="store_true", help="Skip quarterly & monthly charts")
    p.add_argument("--skip-daily", action="store_true", help="Skip sampled daily charts")
    # Optional passthrough for period charts
    p.add_argument("--period-years", nargs="*", type=int, default=None, help="Years for quarterly/monthly/YTD charts (e.g., 2025)")
    p.add_argument("--period-base-dir", default=None, help="Override base-dir for period charts (hourly parquet root)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    py = sys.executable
    repo_root = Path(__file__).resolve().parents[1]

    # 1) Annual chart
    if not args.skip_annual:
        rc = run([py, str(repo_root / "create_bess_revenue_chart.py")])
        if rc != 0:
            logging.warning("Annual chart generation returned non-zero status")

    # 2) Quarterly & Monthly period charts
    if not args.skip_periods:
        cmd = [py, str(repo_root / "create_bess_revenue_charts_quarterly_monthly.py")]
        if args.period_years:
            cmd.extend(["--years", *[str(y) for y in args.period_years]])
        if args.period_base_dir:
            cmd.extend(["--base-dir", args.period_base_dir])
        rc = run(cmd)
        if rc != 0:
            logging.warning("Period charts generation returned non-zero status")

    # 3) Sampled Daily (optional)
    if not args.skip_daily:
        if not args.base_dir:
            logging.error("--base-dir is required for sampled daily charts")
            return 2
        if not args.years:
            logging.error("--years is required for sampled daily charts (e.g., --years 2024 2023)")
            return 2

        for y in args.years:
            cmd = [
                py,
                str(repo_root / "tools" / "batch_daily_plots.py"),
                "--base-dir", args.base_dir,
                "--year", str(y),
                "--per-month", str(args.per_month),
            ]
            if args.all_bess:
                cmd.append("--all")
            if args.bess:
                cmd.extend(["--bess", args.bess])
            if args.with_15min:
                cmd.append("--with-15min")
            if args.with_advanced:
                cmd.append("--with-advanced")
            rc = run(cmd)
            if rc != 0:
                logging.warning("Sampled daily generation failed for year %s", y)

    logging.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

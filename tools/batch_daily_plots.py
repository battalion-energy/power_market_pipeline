#!/usr/bin/env python3
"""
Batch-generate daily BESS charts (market awards + dispatch profile).
Picks up to 5 days per month where data exists in the hourly parquets.

Usage:
  python tools/batch_daily_plots.py --base-dir /pool/.../ERCOT_data --year 2024 --bess CHISMGRD_BES1
  python tools/batch_daily_plots.py --base-dir /pool/.../ERCOT_data --year 2024 --all
"""

from __future__ import annotations

import argparse
from pathlib import Path
import polars as pl
import subprocess
import sys


def days_to_plot(dispatch_path: Path, limit_per_month: int = 5) -> list[str]:
    df = pl.read_parquet(dispatch_path)
    if len(df) == 0:
        return []
    df = df.select([pl.col('local_date')]).unique().sort('local_date')
    df = df.with_columns([
        pl.col('local_date').dt.year().alias('year'),
        pl.col('local_date').dt.month().alias('month')
    ])
    picks = []
    for m in df['month'].unique().to_list():
        days = df.filter(pl.col('month') == m)['local_date'].to_list()
        if not days:
            continue
        step = max(1, len(days) // limit_per_month)
        for i in range(0, len(days), step):
            picks.append(str(days[i]))
            if len([d for d in picks if str(d).split('-')[1] == f"{m:02d}"]) >= limit_per_month:
                break
    return picks


def plot_day(base: Path, bess: str, year: int, date: str):
    cmd = [sys.executable, str(Path(__file__).parent / 'plot_daily_bess.py'),
           '--base-dir', str(base), '--bess', bess, '--year', str(year), '--date', date]
    subprocess.run(cmd, check=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base-dir', required=True)
    p.add_argument('--year', type=int, required=True)
    p.add_argument('--bess', default=None)
    p.add_argument('--all', action='store_true')
    p.add_argument('--per-month', type=int, default=5)
    p.add_argument('--with-15min', action='store_true', help='Also render 15-minute daily chart variant')
    p.add_argument('--with-advanced', action='store_true', help='Also render advanced 4-panel daily chart')
    args = p.parse_args()

    base = Path(args.base_dir)
    ddir = base / 'bess_analysis' / 'hourly' / 'dispatch'
    if args.all:
        targets = list(ddir.glob(f'*_{args.year}_dispatch.parquet'))
    else:
        targets = [ddir / f'{args.bess}_{args.year}_dispatch.parquet']

    for dp in targets:
        if not dp.exists():
            continue
        bess = dp.name.replace(f'_{args.year}_dispatch.parquet','')
        dates = days_to_plot(dp, args.per_month)
        for date in dates:
            plot_day(base, bess, args.year, date)
            if args.with_15min:
                # Call 15-min variant
                import subprocess, sys
                subprocess.run([sys.executable, str(Path(__file__).parent / 'plot_daily_bess_15min.py'),
                                '--base-dir', str(base), '--bess', bess, '--year', str(args.year), '--date', date],
                               check=False)
            if args.with_advanced:
                import subprocess, sys
                subprocess.run([sys.executable, str(Path(__file__).parent / 'plot_daily_bess_advanced.py'),
                                '--base-dir', str(base), '--bess', bess, '--year', str(args.year), '--date', date],
                               check=False)


if __name__ == '__main__':
    main()

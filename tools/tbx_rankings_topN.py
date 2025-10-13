#!/usr/bin/env python3
"""
Produce consolidated top-N rankings per year/type and multi-year weighted-average rankings.

Inputs: tbx_points_rollup/annual/year=YYYY.parquet
Outputs:
  - Per year/type (annual): tbx_rankings/year=YYYY/annual/top{N}_tb2_{TYPE}.csv
  - Multi-year range (weighted by days): tbx_rankings/multi_year/Y1_Y2/ranking_tb2_{TYPE}.csv
    and top{N}_tb2_{TYPE}.csv

Weighted average computed as sum(tb2_sum)/sum(days) per SettlementPoint/(type).
Parallel metrics tb4_avg_day and rtb120_avg_day are reported as weighted averages as well.
"""

from __future__ import annotations

import argparse
import os
from dotenv import load_dotenv
from pathlib import Path
import polars as pl


TYPES = ['ALL', 'RN', 'LZ', 'HUB']


def load_annual(points_dir: Path, year: int) -> pl.DataFrame | None:
    p = points_dir / 'annual' / f'year={year}.parquet'
    if not p.exists():
        return None
    return pl.read_parquet(p)


def filter_type(df: pl.DataFrame, t: str) -> pl.DataFrame:
    if t == 'ALL' or 'SettlementPointType' not in df.columns:
        return df
    return df.filter(pl.col('SettlementPointType') == t)


def emit_top_year(df: pl.DataFrame, year: int, out_dir: Path, top: int) -> None:
    outy = out_dir / f'year={year}' / 'annual'
    outy.mkdir(parents=True, exist_ok=True)
    for t in TYPES:
        d = filter_type(df, t)
        if 'tb2_avg_day' not in d.columns:
            # compute avg/day from sums if missing
            denom = 'days' if 'days' in d.columns else 'days_eff'
            d = d.with_columns((pl.col('tb2_sum')/pl.col(denom)).alias('tb2_avg_day'))
        ranked = d.select([
            'SettlementPoint',
            *( ['SettlementPointType'] if 'SettlementPointType' in d.columns else [] ),
            'tb2_avg_day','tb4_avg_day','rtb120_avg_day',
            'tb2_sum','tb4_sum','rtb120_sum',
            'days' if 'days' in d.columns else 'days_eff'
        ]).sort('tb2_avg_day', descending=True, nulls_last=True)
        ranked.head(top).write_csv(outy / f'top{top}_tb2_{t}.csv')


def emit_multi_year(points_dir: Path, start_year: int, end_year: int, out_dir: Path, top: int) -> None:
    dfs = []
    for y in range(start_year, end_year+1):
        df = load_annual(points_dir, y)
        if df is not None:
            dfs.append(df)
    if not dfs:
        return
    all_yr = pl.concat(dfs, how='diagonal_relaxed')
    denom = 'days' if 'days' in all_yr.columns else 'days_eff'
    grouped = (all_yr.group_by(['SettlementPoint','SettlementPointType'] if 'SettlementPointType' in all_yr.columns else ['SettlementPoint'])
                     .agg([
                         pl.col('tb2_sum').sum().alias('tb2_sum'),
                         pl.col('tb4_sum').sum().alias('tb4_sum'),
                         pl.col('rtb120_sum').sum().alias('rtb120_sum'),
                         pl.col(denom).sum().alias('days')
                     ])
                     .with_columns([
                         (pl.col('tb2_sum')/pl.col('days')).alias('tb2_avg_day'),
                         (pl.col('tb4_sum')/pl.col('days')).alias('tb4_avg_day'),
                         (pl.col('rtb120_sum')/pl.col('days')).alias('rtb120_avg_day'),
                     ]))

    outm = out_dir / 'multi_year' / f'{start_year}_{end_year}'
    outm.mkdir(parents=True, exist_ok=True)
    for t in TYPES:
        d = filter_type(grouped, t)
        ranked = d.select([
            'SettlementPoint',
            *( ['SettlementPointType'] if 'SettlementPointType' in d.columns else [] ),
            'tb2_avg_day','tb4_avg_day','rtb120_avg_day',
            'tb2_sum','tb4_sum','rtb120_sum','days'
        ]).sort('tb2_avg_day', descending=True, nulls_last=True)
        ranked.write_csv(outm / f'ranking_tb2_{t}.csv')
        ranked.head(top).write_csv(outm / f'top{top}_tb2_{t}.csv')


def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', nargs='*', type=int, default=[2025])
    ap.add_argument('--start-year', type=int, default=2022)
    ap.add_argument('--end-year', type=int, default=2025)
    default_data = os.getenv('ERCOT_DATA_DIR', '/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data')
    charts_dir = os.getenv('CHARTS_OUTPUT_DIR', 'charts_output')
    ap.add_argument('--points-dir', default=str(Path(default_data) / 'tbx_points_rollup'))
    ap.add_argument('--out-dir', default=str(Path(charts_dir) / 'tbx_rankings'))
    ap.add_argument('--top', type=int, default=50)
    args = ap.parse_args()

    points = Path(args.points_dir)
    out = Path(args.out_dir)

    for y in args.years:
        df = load_annual(points, y)
        if df is not None:
            emit_top_year(df, y, out, args.top)

    emit_multi_year(points, args.start_year, args.end_year, out, args.top)
    print('[topN] wrote consolidated top-N and multi-year rankings')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

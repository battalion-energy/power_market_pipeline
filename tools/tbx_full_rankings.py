#!/usr/bin/env python3
"""
Produce full, ordered rankings of all settlement points by TB2 (and also TB4, RTB120) using
the annual point rollups (tbx_points_rollup/annual/year=YYYY.parquet).

Outputs per year:
  - tbx_rankings/year=YYYY/ranking_tb2_ALL.csv (all types)
  - tbx_rankings/year=YYYY/ranking_tb2_RN.csv  (resource nodes)
  - tbx_rankings/year=YYYY/ranking_tb2_LZ.csv  (load zones)
  - tbx_rankings/year=YYYY/ranking_tb2_HUB.csv (hubs)

Each CSV is sorted descending by tb2_avg_day and includes helpful columns:
  SettlementPoint, SettlementPointType, tb2_avg_day, tb4_avg_day, rtb120_avg_day,
  tb2_sum, tb4_sum, rtb120_sum, days
"""

from __future__ import annotations

import argparse
import os
from dotenv import load_dotenv
from pathlib import Path
import polars as pl


def write_rankings_for_year(year: int, points_dir: Path, out_dir: Path) -> None:
    src = points_dir / 'annual' / f'year={year}.parquet'
    if not src.exists():
        print(f"[rank] missing {src}")
        return
    df = pl.read_parquet(src)
    # Ensure expected columns
    for col in ['tb2_avg_day', 'tb4_avg_day', 'rtb120_avg_day']:
        if col not in df.columns:
            # Backward compatibility: compute from sums/days if needed
            base = col.replace('_avg_day', '')
            if base + '_sum' in df.columns and ('days' in df.columns or 'days_eff' in df.columns):
                denom = 'days' if 'days' in df.columns else 'days_eff'
                df = df.with_columns((pl.col(base + '_sum')/pl.col(denom)).alias(col))
            else:
                df = df.with_columns(pl.lit(None).alias(col))

    # Common projection
    cols = [
        'SettlementPoint','SettlementPointType',
        'tb2_avg_day','tb4_avg_day','rtb120_avg_day',
        'tb2_sum','tb4_sum','rtb120_sum',
        'days' if 'days' in df.columns else 'days_eff'
    ]
    cols = [c for c in cols if c in df.columns]

    outy = out_dir / f'year={year}'
    outy.mkdir(parents=True, exist_ok=True)

    def emit(label: str, filt: pl.Expr | None):
        d = df
        if filt is not None and 'SettlementPointType' in df.columns:
            d = d.filter(filt)
        ranked = d.select(cols).sort('tb2_avg_day', descending=True, nulls_last=True)
        ranked.write_csv(outy / f'ranking_tb2_{label}.csv')

    emit('ALL', None)
    emit('RN', pl.col('SettlementPointType') == 'RN')
    emit('LZ', pl.col('SettlementPointType') == 'LZ')
    emit('HUB', pl.col('SettlementPointType') == 'HUB')
    print(f"[rank] wrote TB2 rankings for {year} -> {outy}")


def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', nargs='*', type=int, default=[2025])
    default_data = os.getenv('ERCOT_DATA_DIR', '/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data')
    charts_dir = os.getenv('CHARTS_OUTPUT_DIR', 'charts_output')
    ap.add_argument('--points-dir', default=str(Path(default_data) / 'tbx_points_rollup'))
    ap.add_argument('--out-dir', default=str(Path(charts_dir) / 'tbx_rankings'))
    args = ap.parse_args()

    points = Path(args.points_dir)
    out = Path(args.out_dir)
    for y in args.years:
        write_rankings_for_year(y, points, out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

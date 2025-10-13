#!/usr/bin/env python3
"""
Rankings by period (monthly/quarterly/annual) and an all-time ranking.

Inputs (point-level rollups):
  - tbx_points_rollup/monthly/year=YYYY/all.parquet
  - tbx_points_rollup/quarterly/year=YYYY/all.parquet
  - tbx_points_rollup/annual/year=YYYY.parquet

All-time ranking aggregates annual files across years weighting by days (sums/days).
Optionally annotates the earliest BESS COD date per settlement point via dispatch files.
"""

from __future__ import annotations

import argparse
import os
from dotenv import load_dotenv
from pathlib import Path
import polars as pl


def _emit_rank(df: pl.DataFrame, out: Path, metric: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    # guard against duplicate metric selection
    base_cols = [
        'SettlementPoint','SettlementPointType', metric,
        'tb2_avg_day','tb4_avg_day','rtb120_avg_day',
        'tb2_sum','tb4_sum','rtb120_sum','days'
    ]
    # keep order and ensure uniqueness, and only keep existing columns
    seen = set()
    proj = []
    for c in base_cols:
        if c in df.columns and c not in seen:
            proj.append(c)
            seen.add(c)
    df = df.select(proj)
    # rename the primary metric to ensure no duplicate name conflicts in sort/select chain
    if metric not in df.columns:
        return
    df.sort(metric, descending=True, nulls_last=True).write_csv(out)


def _cod_by_sp(mapping_csv: Path, base_dir: Path, years: list[int]) -> pl.DataFrame:
    # Map BESS -> SP
    mp = pl.read_csv(mapping_csv).rename({
        'BESS_Gen_Resource': 'bess_name',
        'Settlement_Point': 'SettlementPoint'
    })[['bess_name','SettlementPoint']].unique(maintain_order=True)
    rows = []
    ddir = base_dir / 'bess_analysis' / 'hourly' / 'dispatch'
    for y in years:
        for r in mp.iter_rows(named=True):
            p = ddir / f"{r['bess_name']}_{y}_dispatch.parquet"
            if not p.exists():
                continue
            try:
                df = pl.read_parquet(p, columns=['local_date'])
                first = df.select(pl.col('local_date').min()).item()
                rows.append({'SettlementPoint': r['SettlementPoint'], 'first_bess_cod': first})
            except Exception:
                pass
    if not rows:
        return pl.DataFrame({'SettlementPoint': [], 'first_bess_cod': []}, schema={'SettlementPoint': pl.Utf8, 'first_bess_cod': pl.Date})
    return pl.DataFrame(rows).group_by('SettlementPoint').agg(pl.col('first_bess_cod').min())


def rankings_for_year(year: int, points_dir: Path, out_dir: Path) -> None:
    # Monthly
    mpath = points_dir / 'monthly' / f'year={year}' / 'all.parquet'
    if mpath.exists():
        m = pl.read_parquet(mpath)
        for metric in ['tb2_avg_day']:
            for label, filt in [('ALL', None), ('RN', pl.col('SettlementPointType')=='RN'), ('LZ', pl.col('SettlementPointType')=='LZ'), ('HUB', pl.col('SettlementPointType')=='HUB')]:
                d = m if filt is None else m.filter(filt)
                months = [x for x in d.select('month').unique()['month'].to_list() if x is not None]
                for mo in sorted(months):
                    dm = d.filter(pl.col('month')==mo)
                    _emit_rank(dm, out_dir / f'year={year}' / 'monthly' / f'month={mo:02d}_ranking_{metric}_{label}.csv', metric)
    # Quarterly
    qpath = points_dir / 'quarterly' / f'year={year}' / 'all.parquet'
    if qpath.exists():
        q = pl.read_parquet(qpath)
        for metric in ['tb2_avg_day']:
            for label, filt in [('ALL', None), ('RN', pl.col('SettlementPointType')=='RN'), ('LZ', pl.col('SettlementPointType')=='LZ'), ('HUB', pl.col('SettlementPointType')=='HUB')]:
                d = q if filt is None else q.filter(filt)
                qts = [x for x in d.select('quarter').unique()['quarter'].to_list() if x is not None]
                for qt in sorted(qts):
                    dq = d.filter(pl.col('quarter')==qt)
                    _emit_rank(dq, out_dir / f'year={year}' / 'quarterly' / f'quarter=Q{qt}_ranking_{metric}_{label}.csv', metric)
    # Annual
    apath = points_dir / 'annual' / f'year={year}.parquet'
    if apath.exists():
        a = pl.read_parquet(apath)
        for metric in ['tb2_avg_day']:
            for label, filt in [('ALL', None), ('RN', pl.col('SettlementPointType')=='RN'), ('LZ', pl.col('SettlementPointType')=='LZ'), ('HUB', pl.col('SettlementPointType')=='HUB')]:
                d = a if filt is None else a.filter(filt)
                _emit_rank(d, out_dir / f'year={year}' / 'annual' / f'ranking_{metric}_{label}.csv', metric)


def rankings_all_time(years: list[int], points_dir: Path, out_dir: Path, mapping_csv: Path | None, base_dir: Path | None) -> None:
    # Concatenate annuals and weight by days
    dfs = []
    for y in years:
        p = points_dir / 'annual' / f'year={y}.parquet'
        if p.exists():
            df = pl.read_parquet(p)
            dfs.append(df)
    if not dfs:
        print('[rank] no annual files for all-time aggregation')
        return
    all_yr = pl.concat(dfs, how='diagonal_relaxed')
    denom = 'days' if 'days' in all_yr.columns else 'days_eff'
    agg = (all_yr.group_by(['SettlementPoint','SettlementPointType'] if 'SettlementPointType' in all_yr.columns else ['SettlementPoint'])
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
    # Optional COD annotation: earliest BESS COD per SP
    if mapping_csv and base_dir:
        cod = _cod_by_sp(mapping_csv, base_dir, years)
        agg = agg.join(cod, on='SettlementPoint', how='left')
    # Emit rankings
    out = out_dir / 'all_time'
    out.mkdir(parents=True, exist_ok=True)
    for label, filt in [('ALL', None), ('RN', pl.col('SettlementPointType')=='RN'), ('LZ', pl.col('SettlementPointType')=='LZ'), ('HUB', pl.col('SettlementPointType')=='HUB')]:
        d = agg if (filt is None or 'SettlementPointType' not in agg.columns) else agg.filter(filt)
        _emit_rank(d, out / f'ranking_tb2_{label}.csv', 'tb2_avg_day')
    print('[rank] wrote all-time rankings ->', out)


def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', nargs='*', type=int, default=list(range(2019, 2026)))
    default_data = os.getenv('ERCOT_DATA_DIR', '/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data')
    charts_dir = os.getenv('CHARTS_OUTPUT_DIR', 'charts_output')
    ap.add_argument('--points-dir', default=str(Path(default_data) / 'tbx_points_rollup'))
    ap.add_argument('--out-dir', default=str(Path(charts_dir) / 'tbx_rankings'))
    ap.add_argument('--mapping', default='bess_mapping/BESS_UNIFIED_MAPPING_V3_CLARIFIED.csv')
    ap.add_argument('--base-dir', default=default_data)
    args = ap.parse_args()

    points = Path(args.points_dir)
    out = Path(args.out_dir)
    for y in args.years:
        rankings_for_year(y, points, out)
    rankings_all_time(args.years, points, out, Path(args.mapping), Path(args.base_dir))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

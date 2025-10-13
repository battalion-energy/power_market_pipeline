#!/usr/bin/env python3
"""
Aggregate TB1/TB2/TB4/RTB120 by point and type (RN/LZ/HUB/etc) without COD filtering.
Writes rollups under tbx_points_rollup/{monthly,quarterly,annual}.
"""

from __future__ import annotations

import argparse
import os
from dotenv import load_dotenv
from pathlib import Path
import polars as pl


def scan_tb_daily(root: Path, year: int) -> pl.DataFrame:
    """Return one row per (SettlementPoint, DeliveryDate) with summed daily TB values and a type.

    We perform full joins across TB1/TB2/TB4/RTB120 daily partitions, then collapse any
    duplicates by grouping by (SP, date). Type is sourced from RTB120 (if present) and
    otherwise set to 'UNKNOWN'.
    """
    tb1 = pl.scan_parquet(str(root / 'tb1_daily' / f'year={year}' / 'date=*.parquet'), extra_columns='ignore').select([
        pl.col('SettlementPoint'), pl.col('DeliveryDate'), pl.col('TB1')
    ])
    tb2 = pl.scan_parquet(str(root / 'tb2_daily' / f'year={year}' / 'date=*.parquet'), extra_columns='ignore').select([
        pl.col('SettlementPoint').alias('SP'), pl.col('DeliveryDate').alias('DD'), pl.col('TB2')
    ])
    tb4 = pl.scan_parquet(str(root / 'tb4_daily' / f'year={year}' / 'date=*.parquet'), extra_columns='ignore').select([
        pl.col('SettlementPoint').alias('SP2'), pl.col('DeliveryDate').alias('DD2'), pl.col('TB4')
    ])
    rtb = pl.scan_parquet(str(root / 'rtb120_daily' / f'year={year}' / 'date=*.parquet'), extra_columns='ignore').select([
        pl.col('SettlementPoint').alias('SPR'), pl.col('DeliveryDate').alias('DDR'), pl.col('RTB120'),
        pl.col('SettlementPointType').alias('SPT_R')
    ])
    lf = (
        tb1.join(tb2, left_on=['SettlementPoint','DeliveryDate'], right_on=['SP','DD'], how='full')
           .join(tb4, left_on=['SettlementPoint','DeliveryDate'], right_on=['SP2','DD2'], how='full')
           .join(rtb, left_on=['SettlementPoint','DeliveryDate'], right_on=['SPR','DDR'], how='full')
           .with_columns([
               pl.col('SPT_R').fill_null('UNKNOWN').alias('SettlementPointType')
           ])
           .select([
               'SettlementPoint','SettlementPointType','DeliveryDate','TB1','TB2','TB4','RTB120'
           ])
    )
    df = lf.collect()
    # Collapse to unique daily row per SP
    df_daily = (df.group_by(['SettlementPoint','SettlementPointType','DeliveryDate'])
                  .agg([
                      pl.col('TB1').sum().alias('TB1'),
                      pl.col('TB2').sum().alias('TB2'),
                      pl.col('TB4').sum().alias('TB4'),
                      pl.col('RTB120').sum().alias('RTB120'),
                  ]))
    return df_daily


def rollup(year: int, root: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = scan_tb_daily(root, year)
    if len(df) == 0:
        print(f"[points] no data for {year}")
        return
    if 'SettlementPointType' not in df.columns:
        df = df.with_columns(pl.lit('UNKNOWN').alias('SettlementPointType'))

    # Monthly
    monthly = (df.with_columns(pl.col('DeliveryDate').dt.month().alias('month'))
                .group_by(['SettlementPoint','SettlementPointType','month'])
                .agg([
                    pl.col('TB1').sum().alias('tb1_sum'),
                    pl.col('TB2').sum().alias('tb2_sum'),
                    pl.col('TB4').sum().alias('tb4_sum'),
                    pl.col('RTB120').sum().alias('rtb120_sum'),
                    pl.col('DeliveryDate').n_unique().alias('days')
                ])
                .with_columns([
                    (pl.col('tb1_sum')/pl.col('days')).alias('tb1_avg_day'),
                    (pl.col('tb2_sum')/pl.col('days')).alias('tb2_avg_day'),
                    (pl.col('tb4_sum')/pl.col('days')).alias('tb4_avg_day'),
                    (pl.col('rtb120_sum')/pl.col('days')).alias('rtb120_avg_day'),
                ]))
    mdir = out_dir / 'monthly' / f'year={year}'
    mdir.mkdir(parents=True, exist_ok=True)
    monthly.write_parquet(mdir / 'all.parquet')

    # Quarterly
    def qtr(m: pl.Expr) -> pl.Expr:
        return ((m - 1) // 3 + 1)
    quarterly = (df.with_columns([
                        pl.col('DeliveryDate').dt.month().alias('month'),
                        qtr(pl.col('DeliveryDate').dt.month()).alias('quarter')
                    ])
                    .group_by(['SettlementPoint','SettlementPointType','quarter'])
                    .agg([
                        pl.col('TB1').sum().alias('tb1_sum'),
                        pl.col('TB2').sum().alias('tb2_sum'),
                        pl.col('TB4').sum().alias('tb4_sum'),
                        pl.col('RTB120').sum().alias('rtb120_sum'),
                        pl.col('DeliveryDate').n_unique().alias('days')
                    ])
                    .with_columns([
                        (pl.col('tb1_sum')/pl.col('days')).alias('tb1_avg_day'),
                        (pl.col('tb2_sum')/pl.col('days')).alias('tb2_avg_day'),
                        (pl.col('tb4_sum')/pl.col('days')).alias('tb4_avg_day'),
                        (pl.col('rtb120_sum')/pl.col('days')).alias('rtb120_avg_day'),
                    ]))
    qdir = out_dir / 'quarterly' / f'year={year}'
    qdir.mkdir(parents=True, exist_ok=True)
    quarterly.write_parquet(qdir / 'all.parquet')

    # Annual
    annual = (df.group_by(['SettlementPoint','SettlementPointType'])
                .agg([
                    pl.col('TB1').sum().alias('tb1_sum'),
                    pl.col('TB2').sum().alias('tb2_sum'),
                    pl.col('TB4').sum().alias('tb4_sum'),
                    pl.col('RTB120').sum().alias('rtb120_sum'),
                    pl.col('DeliveryDate').n_unique().alias('days')
                ])
                .with_columns([
                    (pl.col('tb1_sum')/pl.col('days')).alias('tb1_avg_day'),
                    (pl.col('tb2_sum')/pl.col('days')).alias('tb2_avg_day'),
                    (pl.col('tb4_sum')/pl.col('days')).alias('tb4_avg_day'),
                    (pl.col('rtb120_sum')/pl.col('days')).alias('rtb120_avg_day'),
                ]))
    adir = out_dir / 'annual'
    adir.mkdir(parents=True, exist_ok=True)
    annual.write_parquet(adir / f'year={year}.parquet')
    print(f"[points] wrote rollups for {year} -> {out_dir}")


def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', nargs='*', type=int, default=[2025])
    default_data = os.getenv('ERCOT_DATA_DIR', '/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data')
    ap.add_argument('--tb-root', default=str(Path(default_data) / 'tbx'))
    ap.add_argument('--out-dir', default=str(Path(default_data) / 'tbx_points_rollup'))
    args = ap.parse_args()

    root = Path(args.tb_root)
    out = Path(args.out_dir)
    for y in args.years:
        rollup(y, root, out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

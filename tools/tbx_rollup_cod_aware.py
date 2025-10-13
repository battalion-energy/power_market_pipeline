#!/usr/bin/env python3
"""
COD-aware rollups for TB1/TB2/TB4 from partitioned daily datasets.

Inputs (default relative to repo root):
  - tb1_daily/year=YYYY/date=YYYY-MM-DD.parquet
  - tb2_daily/year=YYYY/date=YYYY-MM-DD.parquet
  - tb4_daily/year=YYYY/date=YYYY-MM-DD.parquet
  - bess_mapping/BESS_UNIFIED_MAPPING_V3_CLARIFIED.csv
  - dispatch parquets under <base_dir>/bess_analysis/hourly/dispatch/*_{year}_dispatch.parquet

Outputs under tbx_rollup/ (created if missing):
  - monthly/year=YYYY/month=MM.parquet (bess_name, settlement_point, tb1_sum, tb2_sum, tb4_sum, days_eff)
  - quarterly/year=YYYY/quarter=Qn.parquet
  - annual/year=YYYY.parquet
"""

from __future__ import annotations

import argparse
import os
from dotenv import load_dotenv
from pathlib import Path
import polars as pl


def compute_cod_map(year: int, mapping: pl.DataFrame, base_dir: Path) -> pl.DataFrame:
    ddir = base_dir / 'bess_analysis' / 'hourly' / 'dispatch'
    rows = []
    for r in mapping.iter_rows(named=True):
        bess = r['gen_resource']
        sp = r['settlement_point']
        p = ddir / f"{bess}_{year}_dispatch.parquet"
        if not p.exists():
            continue
        try:
            df = pl.read_parquet(p)
            dfy = df.filter(pl.col('local_date').dt.year() == year)
            if len(dfy) == 0:
                continue
            first = dfy.select(pl.col('local_date').min()).item()
            rows.append({'bess_name': bess, 'settlement_point': sp, 'cod_date': first})
        except Exception:
            pass
    if not rows:
        return pl.DataFrame(schema={'bess_name': pl.Utf8, 'settlement_point': pl.Utf8, 'cod_date': pl.Date})
    return pl.DataFrame(rows).with_columns(pl.col('cod_date').cast(pl.Date, strict=False))


def scan_tb_daily(root: Path, year: int) -> pl.DataFrame:
    # Read daily datasets and join on (SP, date)
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
        pl.col('SettlementPoint').alias('SPR'), pl.col('DeliveryDate').alias('DDR'), pl.col('RTB120')
    ])
    lf = (
        tb1.join(tb2, left_on=['SettlementPoint','DeliveryDate'], right_on=['SP','DD'], how='full')
           .join(tb4, left_on=['SettlementPoint','DeliveryDate'], right_on=['SP2','DD2'], how='full')
           .join(rtb, left_on=['SettlementPoint','DeliveryDate'], right_on=['SPR','DDR'], how='full')
           .select(['SettlementPoint','DeliveryDate','TB1','TB2','TB4','RTB120'])
    )
    return lf.collect()


def rollup_year(year: int, tb_root: Path, base_dir: Path, out_dir: Path, mapping_csv: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping = pl.read_csv(mapping_csv).rename({
        'BESS_Gen_Resource': 'gen_resource',
        'Settlement_Point': 'settlement_point',
        'IQ_Capacity_MW': 'capacity_mw'
    })[['gen_resource', 'settlement_point']]

    df = scan_tb_daily(tb_root, year)
    if len(df) == 0:
        print(f"[rollup] no TB daily rows for {year}")
        return

    # Map nodes to BESS and apply COD filter (per-bess)
    cod = compute_cod_map(year, mapping, base_dir)
    if len(cod) == 0:
        print(f"[warn] no COD derived from dispatch for {year}; results will be empty")
        return

    df = df.join(mapping, left_on='SettlementPoint', right_on='settlement_point', how='inner')
    df = df.rename({'gen_resource': 'bess_name'})
    df = df.join(cod, left_on=['bess_name', 'SettlementPoint'], right_on=['bess_name','settlement_point'], how='inner')
    df = df.filter(pl.col('DeliveryDate') >= pl.col('cod_date'))

    # Monthly
    monthly = (df.with_columns([
                    pl.col('DeliveryDate').dt.month().alias('month'),
                ])
                .group_by(['bess_name', 'SettlementPoint', 'month'])
                .agg([
                    pl.col('TB1').sum().alias('tb1_sum'),
                    pl.col('TB2').sum().alias('tb2_sum'),
                    pl.col('TB4').sum().alias('tb4_sum'),
                    pl.col('RTB120').sum().alias('rtb120_sum'),
                    pl.col('DeliveryDate').n_unique().alias('days_eff')
                ])
                .with_columns([
                    (pl.col('tb1_sum')/pl.col('days_eff')).alias('tb1_avg_day'),
                    (pl.col('tb2_sum')/pl.col('days_eff')).alias('tb2_avg_day'),
                    (pl.col('tb4_sum')/pl.col('days_eff')).alias('tb4_avg_day'),
                    (pl.col('rtb120_sum')/pl.col('days_eff')).alias('rtb120_avg_day'),
                ])
               )
    mdir = out_dir / 'monthly' / f'year={year}'
    mdir.mkdir(parents=True, exist_ok=True)
    for row in monthly.iter_rows(named=True):
        # Write one file per month to keep sizes small
        month = int(row['month'])
        path = mdir / f'month={month:02d}.parquet'
        # Append-like: read old content if any, then rewrite
        try:
            old = pl.read_parquet(path)
            dfw = old.vstack(pl.DataFrame([row]))
        except Exception:
            dfw = pl.DataFrame([row])
        dfw.write_parquet(path)

    # Quarterly
    def qtr(m: pl.Expr) -> pl.Expr:
        return ((m - 1) // 3 + 1)

    quarterly = (df.with_columns([
                        pl.col('DeliveryDate').dt.month().alias('month'),
                        qtr(pl.col('DeliveryDate').dt.month()).alias('quarter')
                    ])
                    .group_by(['bess_name', 'SettlementPoint', 'quarter'])
                    .agg([
                        pl.col('TB1').sum().alias('tb1_sum'),
                        pl.col('TB2').sum().alias('tb2_sum'),
                        pl.col('TB4').sum().alias('tb4_sum'),
                        pl.col('RTB120').sum().alias('rtb120_sum'),
                        pl.col('DeliveryDate').n_unique().alias('days_eff')
                    ])
                    .with_columns([
                        (pl.col('tb1_sum')/pl.col('days_eff')).alias('tb1_avg_day'),
                        (pl.col('tb2_sum')/pl.col('days_eff')).alias('tb2_avg_day'),
                        (pl.col('tb4_sum')/pl.col('days_eff')).alias('tb4_avg_day'),
                        (pl.col('rtb120_sum')/pl.col('days_eff')).alias('rtb120_avg_day'),
                    ])
               )
    qdir = out_dir / 'quarterly' / f'year={year}'
    qdir.mkdir(parents=True, exist_ok=True)
    for row in quarterly.iter_rows(named=True):
        q = int(row['quarter'])
        path = qdir / f'quarter=Q{q}.parquet'
        try:
            old = pl.read_parquet(path)
            dfw = old.vstack(pl.DataFrame([row]))
        except Exception:
            dfw = pl.DataFrame([row])
        dfw.write_parquet(path)

    # Annual
    annual = (df.group_by(['bess_name', 'SettlementPoint'])
                .agg([
                    pl.col('TB1').sum().alias('tb1_sum'),
                    pl.col('TB2').sum().alias('tb2_sum'),
                    pl.col('TB4').sum().alias('tb4_sum'),
                    pl.col('RTB120').sum().alias('rtb120_sum'),
                    pl.col('DeliveryDate').n_unique().alias('days_eff')
                ])
                .with_columns([
                    (pl.col('tb1_sum')/pl.col('days_eff')).alias('tb1_avg_day'),
                    (pl.col('tb2_sum')/pl.col('days_eff')).alias('tb2_avg_day'),
                    (pl.col('tb4_sum')/pl.col('days_eff')).alias('tb4_avg_day'),
                    (pl.col('rtb120_sum')/pl.col('days_eff')).alias('rtb120_avg_day'),
                ]))
    adir = out_dir / 'annual'
    adir.mkdir(parents=True, exist_ok=True)
    apath = adir / f'year={year}.parquet'
    annual.write_parquet(apath)
    print(f"[rollup] wrote monthly, quarterly, annual for {year} -> {out_dir}")


def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', nargs='*', type=int, default=[2025])
    default_data = os.getenv('ERCOT_DATA_DIR', '/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data')
    ap.add_argument('--tb-root', default=str(Path(default_data) / 'tbx'))
    ap.add_argument('--base-dir', default=default_data)
    ap.add_argument('--mapping', default='bess_mapping/BESS_UNIFIED_MAPPING_V3_CLARIFIED.csv')
    ap.add_argument('--out-dir', default=str(Path(default_data) / 'tbx_rollup'))
    args = ap.parse_args()

    root = Path(args.tb_root)
    base = Path(args.base_dir)
    out = Path(args.out_dir)
    mapping_csv = Path(args.mapping)

    for y in args.years:
        rollup_year(y, root, base, out, mapping_csv)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

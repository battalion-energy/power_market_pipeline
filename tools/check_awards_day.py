#!/usr/bin/env python3
import argparse
from pathlib import Path
import polars as pl

parser = argparse.ArgumentParser()
parser.add_argument('--base-dir', required=True)
parser.add_argument('--bess', required=True)
parser.add_argument('--year', type=int, required=True)
parser.add_argument('--date', required=True)
args = parser.parse_args()

base = Path(args.base_dir)
aw = base / 'bess_analysis' / 'hourly' / 'awards' / f'{args.bess}_{args.year}_awards.parquet'
if not aw.exists():
    print('awards parquet missing:', aw)
    raise SystemExit(1)

df = pl.read_parquet(aw)
dfd = df.filter(pl.col('local_date') == pl.date(args.date))
print('rows:', len(dfd))
if len(dfd) > 0:
    sums = dfd.select([
        pl.col('da_energy_award_mw').sum().alias('da_energy_mw_sum'),
        pl.col('regup_mw').sum().alias('regup_mw_sum'),
        pl.col('regdown_mw').sum().alias('regdown_mw_sum'),
        pl.col('rrs_mw').sum().alias('rrs_mw_sum'),
        pl.col('ecrs_mw').sum().alias('ecrs_mw_sum'),
        pl.col('nonspin_mw').sum().alias('nonspin_mw_sum'),
    ])
    print(sums)
    print(dfd.head(8))


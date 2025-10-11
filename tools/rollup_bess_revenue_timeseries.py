#!/usr/bin/env python3
"""
Roll up BESS revenues monthly, quarterly, annually, and all-years
using only the hourly/15-minute parquet outputs produced by
bess_revenue_calculator.py.

Inputs (defaults target pool layout):
  --base-dir   /pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data
  --years      2020,2021,2022,2023,2024 (comma separated)
  --mapping    bess_mapping/BESS_UNIFIED_MAPPING_V3_CLARIFIED.csv  (for capacity)
  --out-dir    <base-dir>/bess_analysis/rollups

Outputs:
  monthly.parquet/csv, quarterly.parquet/csv, annual.parquet/csv, all_years.parquet/csv

Notes:
  - Energy components come from hourly dispatch parquet:
      da_energy_revenue_hour, rt_net_revenue_hour, rt_gross_revenue_hour, da_spread_revenue_hour
    plus rt_price_avg, da_price_hour for reference.
  - Ancillary revenue computed from hourly awards parquet by Σ(award_mw × mcpc).
  - Capacity comes from the mapping file for $/kW metrics.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import polars as pl


def load_capacity_map(mapping_path: Path) -> pl.DataFrame:
    df = pl.read_csv(mapping_path)
    # Standardize cols
    rename = {}
    if 'BESS_Gen_Resource' in df.columns:
        rename['BESS_Gen_Resource'] = 'bess_name'
    if 'IQ_Capacity_MW' in df.columns:
        rename['IQ_Capacity_MW'] = 'capacity_mw'
    if 'Settlement_Point' in df.columns:
        rename['Settlement_Point'] = 'settlement_point'
    df = df.rename(rename)
    return df.select(['bess_name','capacity_mw','settlement_point'])


def discover_pairs(base: Path, years: list[int]) -> list[tuple[str,int,Path,Path]]:
    dispatch_dir = base / 'bess_analysis' / 'hourly' / 'dispatch'
    awards_dir = base / 'bess_analysis' / 'hourly' / 'awards'
    pairs = []
    if not dispatch_dir.exists() or not awards_dir.exists():
        return pairs
    for y in years:
        for p in dispatch_dir.glob(f'*_{y}_dispatch.parquet'):
            bess = p.name.replace(f'_{y}_dispatch.parquet','')
            aw = awards_dir / f'{bess}_{y}_awards.parquet'
            if aw.exists():
                pairs.append((bess,y,p,aw))
    return pairs


def hourly_to_periods(bess: str, year: int, dispatch_path: Path, awards_path: Path) -> pl.DataFrame:
    dp = pl.read_parquet(dispatch_path)
    aw = pl.read_parquet(awards_path)

    # Add period columns
    dp = dp.with_columns([
        pl.col('local_date').dt.year().alias('year'),
        pl.col('local_date').dt.month().alias('month'),
        ((pl.col('local_date').dt.month() - 1) // 3 + 1).alias('quarter')
    ])
    aw = aw.with_columns([
        pl.col('local_date').dt.year().alias('year'),
        pl.col('local_date').dt.month().alias('month'),
        ((pl.col('local_date').dt.month() - 1) // 3 + 1).alias('quarter')
    ])

    # Energy components from dispatch parquet (exact hourly sums)
    energy = dp.group_by(['year','quarter','month']).agg([
        pl.col('da_energy_revenue_hour').sum().alias('da_energy'),
        pl.col('rt_net_revenue_hour').sum().alias('rt_net'),
        pl.col('rt_gross_revenue_hour').sum().alias('rt_gross'),
        pl.col('da_spread_revenue_hour').sum().alias('da_spread')
    ])

    # Ancillary revenues: Σ(award_mw × mcpc) hourly
    # Create rev columns on the fly where both exist
    for c_mw,c_p in [('regup_mw','regup_mcpc'),('regdown_mw','regdown_mcpc'),('rrs_mw','rrs_mcpc'),('ecrs_mw','ecrs_mcpc'),('nonspin_mw','nonspin_mcpc')]:
        if c_mw in aw.columns and c_p in aw.columns:
            aw = aw.with_columns((pl.col(c_mw).fill_null(0.0) * pl.col(c_p).fill_null(0.0)).alias(f'{c_mw}_rev'))

    as_cols = [c for c in ['regup_mw_rev','regdown_mw_rev','rrs_mw_rev','ecrs_mw_rev','nonspin_mw_rev'] if c in aw.columns]
    if as_cols:
        as_rev = aw.group_by(['year','quarter','month']).agg([
            *[pl.col(c).sum().alias(c) for c in as_cols],
        ])
        # Sum AS rev
        as_rev = as_rev.with_columns(pl.sum_horizontal([pl.col(c).fill_null(0.0) for c in as_cols]).alias('as_total'))
    else:
        as_rev = pl.DataFrame({'year':[], 'quarter':[], 'month':[], 'as_total':[]})

    # Merge energy + AS
    period = energy.join(as_rev, on=['year','quarter','month'], how='left').with_columns(
        pl.col('as_total').fill_null(0.0)
    )
    period = period.with_columns((pl.col('da_energy') + pl.col('rt_net') + pl.col('as_total')).alias('total'))
    return period.with_columns([
        pl.lit(bess).alias('bess_name'),
        pl.lit(year).alias('file_year')
    ])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base-dir', default='/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data')
    p.add_argument('--years', default='2020,2021,2022,2023,2024,2025')
    p.add_argument('--mapping', default='bess_mapping/BESS_UNIFIED_MAPPING_V3_CLARIFIED.csv')
    p.add_argument('--out-dir', default=None)
    args = p.parse_args()

    base = Path(args.base_dir)
    # Default to repo-local output unless --out-dir is given to avoid permission issues
    out_dir = Path(args.out_dir) if args.out_dir else Path('tools/output/rollups')
    out_dir.mkdir(parents=True, exist_ok=True)

    years = [int(x) for x in args.years.split(',')]
    cap_map = load_capacity_map(Path(args.mapping))

    # Build period rows
    rows = []
    for bess, year, dp, aw in discover_pairs(base, years):
        try:
            period = hourly_to_periods(bess, year, dp, aw)
            rows.append(period)
        except Exception:
            continue

    if not rows:
        print('No hourly pairs discovered. Aborting.')
        return

    periods = pl.concat(rows, how='diagonal_relaxed')

    # Join capacity for $/kW metrics
    periods = periods.join(cap_map, left_on='bess_name', right_on='bess_name', how='left')
    periods = periods.with_columns([
        (pl.col('total') / pl.col('capacity_mw')).alias('total_per_mw'),
        (pl.col('da_energy') / pl.col('capacity_mw')).alias('da_per_mw'),
        (pl.col('rt_net') / pl.col('capacity_mw')).alias('rt_per_mw'),
        (pl.col('as_total') / pl.col('capacity_mw')).alias('as_per_mw')
    ])

    # Monthly
    monthly = periods.select(['bess_name','file_year','year','month','da_energy','rt_net','rt_gross','da_spread','as_total','total','capacity_mw','total_per_mw','da_per_mw','rt_per_mw','as_per_mw'])
    monthly.write_parquet(out_dir / 'monthly.parquet')
    monthly.write_csv(out_dir / 'monthly.csv')

    # Quarterly
    quarterly = periods.group_by(['bess_name','file_year','year','quarter']).agg([
        pl.col('da_energy').sum(),
        pl.col('rt_net').sum(),
        pl.col('rt_gross').sum(),
        pl.col('da_spread').sum(),
        pl.col('as_total').sum(),
        pl.col('total').sum(),
        pl.col('capacity_mw').first()
    ]).with_columns([
        (pl.col('total') / pl.col('capacity_mw')).alias('total_per_mw')
    ])
    quarterly.write_parquet(out_dir / 'quarterly.parquet')
    quarterly.write_csv(out_dir / 'quarterly.csv')

    # Annual
    annual = periods.group_by(['bess_name','file_year']).agg([
        pl.col('da_energy').sum(), pl.col('rt_net').sum(), pl.col('rt_gross').sum(), pl.col('da_spread').sum(), pl.col('as_total').sum(), pl.col('total').sum(), pl.col('capacity_mw').first()
    ]).with_columns([(pl.col('total') / pl.col('capacity_mw')).alias('total_per_mw')])
    annual.write_parquet(out_dir / 'annual.parquet')
    annual.write_csv(out_dir / 'annual.csv')

    # All years aggregated
    all_years = annual.group_by(['bess_name']).agg([
        pl.col('da_energy').sum(), pl.col('rt_net').sum(), pl.col('rt_gross').sum(), pl.col('da_spread').sum(), pl.col('as_total').sum(), pl.col('total').sum(), pl.col('capacity_mw').first()
    ]).with_columns([(pl.col('total') / pl.col('capacity_mw')).alias('total_per_mw')])
    all_years.write_parquet(out_dir / 'all_years.parquet')
    all_years.write_csv(out_dir / 'all_years.csv')

    print(f"Saved rollups in {out_dir}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Daily 15-minute dispatch + price chart for a single BESS.

Reads the 15-minute settlement parquet produced by bess_revenue_calculator.py
and renders two panels:
  - Left: 15-min net actual MWh and 15-min net basepoint MWh (bars)
  - Right: 15-min RT price (and DA price if present) as lines

Usage:
  python tools/plot_daily_bess_15min.py \
      --base-dir /pool/.../ERCOT_data --bess CHISMGRD_BES1 --year 2024 --date 2024-02-20
"""

from __future__ import annotations

import argparse
from pathlib import Path
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base-dir', required=True)
    p.add_argument('--bess', required=True)
    p.add_argument('--year', type=int, required=True)
    p.add_argument('--date', required=True)
    args = p.parse_args()

    base = Path(args.base_dir)
    p15 = base / 'bess_analysis' / 'settlement_15min' / f'{args.bess}_{args.year}_settlement_15min.parquet'
    if not p15.exists():
        raise FileNotFoundError(f"15-min parquet not found: {p15}")

    df = pl.read_parquet(p15)
    day = pd.to_datetime(args.date).date()
    df = df.filter(pl.col('local_date') == day).sort('ts_utc')
    if len(df) == 0:
        raise ValueError('No 15-min rows for this date')

    # Compute MWh per 15-min
    df = df.with_columns([
        (pl.col('actual_mw') * 0.25).alias('net_actual_mwh_15'),
        ((pl.col('bp_gen_mw') - pl.col('bp_load_mw')) * 0.25).alias('net_bp_mwh_15')
    ])

    pdf = df.to_pandas()
    # Fallback: if DA price missing or all-NaN in 15-min parquet, use hourly dispatch DA price
    if ('da_price' not in pdf.columns) or (pd.isna(pdf.get('da_price', pd.Series())).all()):
        try:
            phd = base / 'bess_analysis' / 'hourly' / 'dispatch' / f'{args.bess}_{args.year}_dispatch.parquet'
            if phd.exists():
                dph = pl.read_parquet(phd).filter(pl.col('local_date') == day).sort('local_hour').to_pandas()
                if 'da_price_hour' in dph.columns and len(dph) > 0:
                    # map hour -> DA price across 15-min bins
                    hour_to_da = {int(h): v for h, v in zip(dph['local_hour'].values, dph['da_price_hour'].values)}
                    da_series = [hour_to_da.get(int(hh), np.nan) for hh in pdf['local_hour'].values]
                    pdf['da_price'] = da_series
        except Exception:
            pass
    # X-axis in hour.fraction for 15-min resolution
    x = pdf['local_hour'].astype(float).values + (pd.to_datetime(pdf['ts_utc']).dt.minute.values / 60.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # Panel 1: 15-min bars for actual and basepoint (distinct colors + slight offset)
    OUT_COLOR = '#17BECF'    # teal (distinct)
    BP_COLOR  = '#FF7F0E'    # orange (distinct)
    width = 0.10
    ax1.bar(x - width/2, pdf['net_actual_mwh_15'].values, width=width, color=OUT_COLOR, alpha=0.85, label='Output (MWh/15m)', linewidth=0)
    ax1.bar(x + width/2, pdf['net_bp_mwh_15'].values, width=width, color=BP_COLOR, alpha=0.65, label='RT Basepoint (MWh/15m)', linewidth=0)
    ax1.set_title(f"{day.strftime('%-m/%-d')} 15-min Dispatch | {args.bess}")
    ax1.set_xlabel('Hour')
    ax1.set_ylabel('MWh / 15m')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.legend(loc='upper left', fontsize=8)

    # Panel 2: RT and DA prices (15-min)
    ax2.plot(x, pdf['rt_price'].values, color='#2E4053', label='RT Price', linewidth=1.6)
    if 'da_price' in pdf.columns:
        ax2.plot(x, pdf['da_price'].values, color='#1f77b4', label='DA Price', linewidth=1.2)
    ax2.set_title(f"{day.strftime('%-m/%-d')} 15-min Prices | {args.bess}")
    ax2.set_xlabel('Hour')
    ax2.set_ylabel('Price ($/MWh)')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.legend(loc='upper left', fontsize=8)

    out_dir = Path(__file__).parent / 'output' / 'plots_15min'
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.bess}_{args.date}_15min.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()

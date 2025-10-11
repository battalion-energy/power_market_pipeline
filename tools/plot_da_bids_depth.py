#!/usr/bin/env python3
"""
DA bids/offers depth heatmap with DA hourly energy price overlaid.

Searches for DAM *Energy Only Offers* in rollup files and builds an hourly
time–price depth view using price blocks (EnergyOnlyOfferPriceN / MWN). The
overlay shows DA price at the resource node (or hub) for the day.

Usage:
  python tools/plot_da_bids_depth.py \
    --base-dir /pool/.../ERCOT_data \
    --bess CROSSETT_BES1 --year 2024 --date 2024-02-11 \
    --mapping bess_mapping/BESS_UNIFIED_MAPPING_V3_CLARIFIED.csv \
    --pmin -250 --pmax 250 --pstep 5 --out-dir tools/output/da_bids_depth
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap


def load_mapping(mp: Path) -> pd.DataFrame:
    df = pd.read_csv(mp)
    cols = {"BESS_Gen_Resource":"bess","Settlement_Point":"sp"}
    for k,v in cols.items():
        if k in df.columns:
            df = df.rename(columns={k:v})
    return df


def find_offers_file(base: Path, year: int) -> Path | None:
    cand_dirs = [
        base / 'rollup_files' / 'DAM_Energy_Only_Offers',
        base / 'rollup_files' / 'DAM_Gen_Resources',
    ]
    for d in cand_dirs:
        p = d / f'{year}.parquet'
        if p.exists():
            return p
    return None


def blue_orange():
    neg = ['#cfe7ff','#8fc2ff','#4f9bff','#1f70b4']    # light->deep blue
    pos = ['#ffe1c2','#f7b267','#e98b2a','#c55400']    # light->deep orange
    colors = neg[::-1] + ['#f2e1c8'] + pos
    return LinearSegmentedColormap.from_list('blueorange', colors, N=256)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-dir', required=True)
    ap.add_argument('--bess', required=True)
    ap.add_argument('--year', type=int, required=True)
    ap.add_argument('--date', required=True)
    ap.add_argument('--mapping', required=True)
    ap.add_argument('--pmin', type=float, default=-250)
    ap.add_argument('--pmax', type=float, default=250)
    ap.add_argument('--pstep', type=float, default=5.0)
    ap.add_argument('--out-dir', default='tools/output/da_bids_depth')
    args = ap.parse_args()

    base = Path(args.base_dir)
    mp = load_mapping(Path(args.mapping))
    row = mp[mp['bess'] == args.bess].head(1)
    sp = row['sp'].values[0] if not row.empty else None

    offers_p = find_offers_file(base, args.year)
    if not offers_p:
        raise FileNotFoundError('No DAM offers parquet found')
    df = pl.read_parquet(offers_p)

    # Try various column names for price/mw blocks
    cols = df.columns
    price_cols = [c for c in cols if 'Price' in c and ('Offer' in c or 'EnergyOnly' in c)]
    mw_cols    = [c for c in cols if (c.endswith('MW') or 'MW' in c) and ('Offer' in c or 'EnergyOnly' in c)]
    price_cols = sorted(price_cols)
    mw_cols = sorted(mw_cols)

    # Filter by resource name or settlement point
    if 'ResourceName' in cols:
        dff = df.filter(pl.col('ResourceName') == args.bess)
    elif 'SettlementPoint' in cols and sp is not None:
        dff = df.filter(pl.col('SettlementPoint') == sp)
    else:
        dff = df.head(0)
    if len(dff) == 0:
        raise ValueError('No DAM offer rows matched resource or node')

    # Date/hour selection
    if 'DeliveryDate' in dff.columns:
        # ensure DeliveryDate is a date and filter by literal date
        dff = dff.with_columns(pl.col('DeliveryDate').cast(pl.Date))
        from datetime import date as _date
        tdate = _date.fromisoformat(str(args.date))
        dff = dff.filter(pl.col('DeliveryDate') == pl.lit(tdate))
        hour_col = 'hour' if 'hour' in dff.columns else 'HourEnding'
        if hour_col == 'HourEnding':
            dff = dff.with_columns(pl.col('HourEnding').cast(pl.Utf8).str.slice(0,2).cast(pl.Int32).alias('hour'))
        dff = dff.sort('hour')
    else:
        raise ValueError('DAM offers missing DeliveryDate/hour meta')

    hours = dff['hour'].to_list()

    bins = np.arange(args.pmin, args.pmax+args.pstep/2, args.pstep)
    Z = np.zeros((len(bins), len(hours)))

    # Build depth for each hour using cumulative logic (sell above 0, buy below 0)
    pdff = dff.to_pandas()
    for j,h in enumerate(hours):
        row = pdff.iloc[j]
        pairs = []
        for pc,mc in zip(price_cols, mw_cols):
            pv = row.get(pc); mv = row.get(mc)
            if pd.notna(pv) and pd.notna(mv):
                pairs.append((float(pv), float(mv)))
        pairs.sort(key=lambda x: x[0])
        if not pairs:
            continue
        p = np.array([x[0] for x in pairs]); mw = np.array([x[1] for x in pairs])
        sell_depth = np.array([mw[p <= y].sum() for y in bins])
        buy_depth  = np.array([mw[p >= y].sum() for y in bins])
        Z[:,j] = np.where(bins>=0, sell_depth, 0.0) + np.where(bins<=0, -buy_depth, 0.0)

    # Load DA price overlay for the node (or HB_BUSAVG if node not available in the file)
    da_p = base / 'rollup_files' / 'DA_prices' / f'{args.year}.parquet'
    da_overlay = None
    if da_p.exists():
        dpr = pl.read_parquet(da_p)
        if 'datetime' in dpr.columns:
            dp = dpr
        else:
            dp = dpr.rename({'DeliveryDate':'datetime'})
        # Wide format likely; find column for node or fallback
        if sp and sp in dp.columns:
            dp = dp.select(['datetime', sp]).rename({sp:'da_price'})
        elif 'HB_BUSAVG' in dp.columns:
            dp = dp.select(['datetime','HB_BUSAVG']).rename({'HB_BUSAVG':'da_price'})
        else:
            dp = None
        if dp is not None:
            dp = dp.with_columns(pl.col('datetime').cast(pl.Datetime)).filter(pl.col('datetime').dt.date()==pl.date(args.date))
            # take hour values
            dp = dp.with_columns(pl.col('datetime').dt.hour().alias('hour')).sort('hour')
            da_overlay = {int(h): float(v) for h,v in zip(dp['hour'].to_list(), dp['da_price'].to_list())}

    # Plot
    fig, ax = plt.subplots(figsize=(12,6))
    vmax = max(1.0, np.nanmax(np.abs(Z)))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = blue_orange()
    X, Y = np.meshgrid(range(len(hours)), bins)
    im = ax.pcolormesh(X, Y, Z, cmap=cmap, norm=norm, shading='auto')
    ax.set_ylabel('Price ($/MWh)'); ax.set_xlabel('Hour')
    ax.set_xticks(range(len(hours)))
    ax.set_xticklabels([f"{h:02d}:00" for h in hours], rotation=0)
    fig.colorbar(im, ax=ax, label='Power (MW)')
    if da_overlay:
        xs = list(range(len(hours)))
        ys = [da_overlay.get(h, np.nan) for h in hours]
        ax.plot(xs, ys, color='#1f77b4', linewidth=1.6, label='DA Price')
        ax.legend(loc='upper right', fontsize=8)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / f"{args.bess}_{args.date}_da_bids.png"
    fig.tight_layout(); fig.savefig(out_path, dpi=180, bbox_inches='tight')
    print('Saved:', out_path)


if __name__ == '__main__':
    main()

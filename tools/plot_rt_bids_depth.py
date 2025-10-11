#!/usr/bin/env python3
"""
RT bids/offers depth heatmap with realized RT/DA prices overlaid.

Builds a time–price grid using SCED offer curve segments (SCED1/SCED2) for a
single BESS and day. For each 15-minute bin, it takes the last SCED run and
computes cumulative sell depth (SCED1; discharge) and buy depth (SCED2; charge)
across price bins. Positive depth is plotted in warm colors, negative in greys.

Usage:
  python tools/plot_rt_bids_depth.py \
    --base-dir /pool/.../ERCOT_data \
    --bess CROSSETT_BES1 --year 2024 --date 2024-02-11 \
    --pmin -250 --pmax 250 --pstep 2

Output:
  tools/output/rt_bids_depth/<BESS>_<date>_rt_bids.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime, date
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap


def last_sced_per_15(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.sort('sced_dt')
          .group_by('t15')
          .agg([pl.all().last()])
          .sort('t15')
    )


def extract_curve_pairs(df_row: pd.Series, prefix: str) -> list[tuple[float,float]]:
    # prefix 'SCED1' or 'SCED2'
    prices = []
    mws = []
    for i in range(1, 40):  # enough headroom
        pcol = f'{prefix} Curve-Price{i}'
        mcol = f'{prefix} Curve-MW{i}'
        if pcol in df_row and mcol in df_row:
            pv = df_row[pcol]
            mv = df_row[mcol]
            if pd.notna(pv) and pd.notna(mv):
                prices.append(float(pv))
                mws.append(float(mv))
    pairs = [(p,m) for p,m in zip(prices,mws) if np.isfinite(p) and np.isfinite(m)]
    # Sort by price ascending for cumulative logic
    pairs.sort(key=lambda x: x[0])
    return pairs


def curve_depth_at_bins(pairs: list[tuple[float,float]], bins: np.ndarray, side: str) -> np.ndarray:
    # side == 'sell' (SCED1): depth(y) = sum(mw for price <= y)
    # side == 'buy'  (SCED2): depth(y) = sum(mw for price >= y)
    if not pairs:
        return np.zeros_like(bins, dtype=float)
    p = np.array([pr for pr,_ in pairs])
    mw = np.array([mw for _,mw in pairs])
    depth = np.zeros_like(bins, dtype=float)
    if side == 'sell':
        # cumulative when bin price passes segment price
        # vectorized: for each bin y, include segments with p <= y
        for i,y in enumerate(bins):
            depth[i] = mw[p <= y].sum()
    else:
        for i,y in enumerate(bins):
            depth[i] = mw[p >= y].sum()
    return depth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-dir', required=True)
    ap.add_argument('--bess', required=True)
    ap.add_argument('--year', type=int, required=True)
    ap.add_argument('--date', required=True)
    ap.add_argument('--pmin', type=float, default=-250)
    ap.add_argument('--pmax', type=float, default=250)
    ap.add_argument('--pstep', type=float, default=2.0)
    ap.add_argument('--out-dir', type=str, default=None, help='Directory to save image (defaults to tools/output/rt_bids_depth)')
    args = ap.parse_args()

    base = Path(args.base_dir)
    sced_p = base / 'rollup_files' / 'SCED_Gen_Resources' / f'{args.year}.parquet'
    if not sced_p.exists():
        raise FileNotFoundError(f'Missing SCED file: {sced_p}')

    # Load SCED for resource and date
    sced = pl.read_parquet(sced_p).filter(pl.col('ResourceName') == args.bess)
    if len(sced) == 0:
        raise ValueError('No SCED rows for resource')
    sced = sced.with_columns([
        pl.col('SCEDTimeStamp').str.strptime(pl.Datetime, '%m/%d/%Y %H:%M:%S')
            .dt.replace_time_zone('America/Chicago').dt.convert_time_zone('UTC').alias('sced_dt'),
        ])
    target = pd.to_datetime(args.date).date()
    sced = sced.filter(sced['sced_dt'].dt.date() == target)
    if len(sced) == 0:
        raise ValueError('No SCED rows for date')
    sced = sced.with_columns([pl.col('sced_dt').dt.truncate('15m').alias('t15')])
    sced15 = last_sced_per_15(sced)

    # Settlement 15m prices for overlay
    rt15_p = base / 'bess_analysis' / 'settlement_15min' / f'{args.bess}_{args.year}_settlement_15min.parquet'
    prices15 = None
    if rt15_p.exists():
        prices15 = pl.read_parquet(rt15_p).filter(pl.col('local_date')==pl.lit(target)).sort('ts_utc')

    # Build price bins and grids
    bins = np.arange(args.pmin, args.pmax+args.pstep/2, args.pstep)
    times = sced15['t15'].to_list()
    Z = np.zeros((len(bins), len(times)), dtype=float)

    # Compute depth for each 15m from SCED1/SCED2 curves
    cols = sced15.columns
    mw1 = sorted([c for c in cols if c.startswith('SCED1 Curve-MW')])
    pr1 = sorted([c for c in cols if c.startswith('SCED1 Curve-Price')])
    mw2 = sorted([c for c in cols if c.startswith('SCED2 Curve-MW')])
    pr2 = sorted([c for c in cols if c.startswith('SCED2 Curve-Price')])

    import warnings
    df15 = sced15.to_pandas()
    for j in range(len(times)):
        row = df15.iloc[j]
        # Collect pairs
        def pairs_from(prefix, mwcols, prcols):
            pairs = []
            for mc,pc in zip(mwcols, prcols):
                mv = row.get(mc); pv = row.get(pc)
                if pd.notna(mv) and pd.notna(pv):
                    pairs.append((float(pv), float(mv)))
            pairs.sort(key=lambda x: x[0])
            return pairs

        sell_pairs = pairs_from('SCED1', mw1, pr1)
        buy_pairs  = pairs_from('SCED2', mw2, pr2)
        sell_depth = curve_depth_at_bins(sell_pairs, bins, 'sell')
        buy_depth  = curve_depth_at_bins(buy_pairs, bins, 'buy')
        # Only show buy on y <= 0 and sell on y >= 0
        Z[:,j] = np.where(bins>=0, sell_depth, 0.0) + np.where(bins<=0, -buy_depth, 0.0)

    # Plot
    fig, ax = plt.subplots(figsize=(12,6))
    # Create diverging normalization centered on zero
    vmax = max(1.0, np.nanmax(np.abs(Z)))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    # Build a color-blind friendly diverging colormap (blue/orange) with color near zero
    def blue_orange():
        neg = ['#cfe7ff','#8fc2ff','#4f9bff','#1f70b4']    # light->deep blue
        pos = ['#ffe1c2','#f7b267','#e98b2a','#c55400']    # light->deep orange
        colors = neg[::-1] + ['#f2e1c8'] + pos
        return LinearSegmentedColormap.from_list('blueorange', colors, N=256)
    cmap = blue_orange()
    t_axis = [pd.to_datetime(t).tz_convert('America/Chicago') for t in times]
    X, Y = np.meshgrid(range(len(t_axis)), bins)
    im = ax.pcolormesh(X, Y, Z, cmap=cmap, norm=norm, shading='auto')
    ax.set_ylabel('Price ($/MWh)'); ax.set_xlabel('Time')
    ax.set_xticks(range(0,len(t_axis), max(1,len(t_axis)//8)))
    ax.set_xticklabels([t.strftime('%H:%M') for i,t in enumerate(t_axis) if i in ax.get_xticks()])
    cbar = fig.colorbar(im, ax=ax, label='Power (MW)')

    # Overlay RT/DA prices (15-min)
    if prices15 is not None and len(prices15)>0:
        p15 = prices15.to_pandas().sort_values('ts_utc')
        # map to the same t15 bins
        # find index in times for each ts_utc floor 15
        idxmap = {pd.to_datetime(t): i for i,t in enumerate(times)}
        xs=[]; rt=[]; da=[]
        for ts,rtp, dap in zip(p15['ts_utc'], p15['rt_price'], p15.get('da_price', pd.Series([np.nan]*len(p15)))):
            key = pd.to_datetime(ts)
            i = idxmap.get(key, None)
            if i is not None:
                xs.append(i)
                rt.append(rtp)
                da.append(np.nan if dap is None else dap)
        ax.plot(xs, rt, color='#2E4053', linewidth=1.6, label='RT Price')
        # Guard against empty/None DA values without emitting warnings
        mean_da = np.nan
        if len(da):
            with np.errstate(all='ignore'):
                mean_da = np.nanmean(np.array(da, dtype=float))
        if np.isfinite(mean_da):
            ax.plot(xs, da, color='#1f77b4', linewidth=1.2, label='DA Price')
        ax.legend(loc='upper right', fontsize=8)

    # Choose output folder
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path(__file__).parent / 'output' / 'rt_bids_depth'
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.bess}_{args.date}_rt_bids.png"
    fig.tight_layout(); fig.savefig(out, dpi=180, bbox_inches='tight')
    print('Saved:', out)


if __name__ == '__main__':
    main()

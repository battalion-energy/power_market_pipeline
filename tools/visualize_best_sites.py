#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from dotenv import load_dotenv
from pathlib import Path
import math
import polars as pl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load_annual(points_dir: Path, year: int) -> pl.DataFrame:
    p = points_dir / 'annual' / f'year={year}.parquet'
    if not p.exists():
        raise SystemExit(f"Missing annual rollup: {p}")
    return pl.read_parquet(p)


def load_monthly(points_dir: Path, year: int) -> pl.DataFrame:
    p = points_dir / 'monthly' / f'year={year}' / 'all.parquet'
    if not p.exists():
        raise SystemExit(f"Missing monthly rollup: {p}")
    return pl.read_parquet(p)


def pick_top_sites(df_annual: pl.DataFrame, sp_type: str | None, top: int) -> pl.DataFrame:
    d = df_annual
    if sp_type and 'SettlementPointType' in d.columns:
        d = d.filter(pl.col('SettlementPointType') == sp_type)
    # ensure tb2_avg_day available
    if 'tb2_avg_day' not in d.columns:
        denom = 'days' if 'days' in d.columns else 'days_eff'
        d = d.with_columns((pl.col('tb2_sum')/pl.col(denom)).alias('tb2_avg_day'))
    return d.sort('tb2_avg_day', descending=True, nulls_last=True).head(top)


def bar_tb2(df_top: pl.DataFrame, out_path: Path, title: str):
    d = df_top.to_pandas()
    names = d['SettlementPoint']
    tb2 = d['tb2_avg_day']
    tb4 = d['tb4_avg_day'] if 'tb4_avg_day' in d.columns else None
    rt = d['rtb120_avg_day'] if 'rtb120_avg_day' in d.columns else None

    fig, ax = plt.subplots(figsize=(20, 8))
    x = np.arange(len(names))
    ax.bar(x, tb2, color='#5DADE2', label='TB2 avg/day ($/MW-day)')
    # Overlay TB4 and RTB120 as markers
    if tb4 is not None:
        ax.plot(x, tb4, color='#8B7BC8', marker='o', linewidth=2, label='TB4 avg/day')
    if rt is not None:
        ax.plot(x, rt, color='#F39C12', marker='^', linewidth=2, label='RTB120 avg/day')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90, ha='right', fontsize=8)
    ax.set_ylabel('$/MW-day')
    ax.set_title(title)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend(loc='upper right')
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def scatter_tb2_rtb(df_top: pl.DataFrame, out_path: Path, title: str):
    d = df_top.to_pandas()
    if 'rtb120_avg_day' not in d.columns:
        return
    x = d['tb2_avg_day']
    y = d['rtb120_avg_day']
    if 'days' in d.columns:
        sz = d['days']
    elif 'days_eff' in d.columns:
        sz = d['days_eff']
    else:
        sz = np.full(len(d), 180)
    # scale sizes
    sizes = 30 + 70 * (sz / sz.max())
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(x, y, s=sizes, c='#5FD4AF', alpha=0.8, edgecolor='k', linewidth=0.3)
    # annotate top 8 by TB2
    ord_idx = np.argsort(-x)[:8]
    for i in ord_idx:
        ax.annotate(d['SettlementPoint'].iloc[i], (x.iloc[i], y.iloc[i]), xytext=(5,5), textcoords='offset points', fontsize=8)
    ax.set_xlabel('TB2 avg/day ($/MW-day)')
    ax.set_ylabel('RTB120 avg/day ($/MW-day)')
    ax.set_title(title)
    ax.grid(alpha=0.3, linestyle='--')
    # trend line
    if len(d) >= 3:
        m, b = np.polyfit(x, y, 1)
        xx = np.linspace(x.min(), x.max(), 50)
        ax.plot(xx, m*xx + b, color='gray', linestyle='--', linewidth=1)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def heatmap_monthly(df_monthly: pl.DataFrame, df_top: pl.DataFrame, out_path: Path, title: str):
    # build matrix SP x month of tb2_avg_day
    sps = df_top['SettlementPoint'].to_list()
    d = df_monthly.filter(pl.col('SettlementPoint').is_in(sps))
    if 'tb2_avg_day' not in d.columns:
        d = d.with_columns((pl.col('tb2_sum')/pl.col('days')).alias('tb2_avg_day'))
    mat = d.select(['SettlementPoint','month','tb2_avg_day'])
    # pivot
    piv = mat.to_pandas().pivot_table(index='SettlementPoint', columns='month', values='tb2_avg_day', aggfunc='first')
    # order rows by annual avg from df_top
    order = [sp for sp in df_top['SettlementPoint'].to_list() if sp in piv.index]
    piv = piv.loc[order]
    # plot
    fig, ax = plt.subplots(figsize=(12, max(6, len(order)*0.3)))
    im = ax.imshow(piv.values, aspect='auto', cmap='viridis')
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(order, fontsize=8)
    months = piv.columns.to_list()
    ax.set_xticks(np.arange(len(months)))
    ax.set_xticklabels([f'{int(m):02d}' for m in months], fontsize=8)
    ax.set_xlabel('Month')
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('TB2 avg/day ($/MW-day)')
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser(description='Visualize top sites by TB2')
    ap.add_argument('--year', type=int, required=True)
    ap.add_argument('--type', default='RN', choices=['ALL','RN','LZ','HUB'])
    ap.add_argument('--top', type=int, default=25)
    default_data = os.getenv('ERCOT_DATA_DIR', '/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data')
    charts_dir = os.getenv('CHARTS_OUTPUT_DIR', 'charts_output')
    ap.add_argument('--points-dir', default=str(Path(default_data) / 'tbx_points_rollup'))
    ap.add_argument('--out-dir', default=str(Path(charts_dir) / 'tbx_charts'))
    args = ap.parse_args()

    points = Path(args.points_dir)
    out = Path(args.out_dir)

    ann = load_annual(points, args.year)
    top_df = pick_top_sites(ann, None if args.type=='ALL' else args.type, args.top)
    # Charts
    label = f"{args.year} Top {len(top_df)} {args.type} by TB2 avg/day"
    bar_tb2(top_df, out / f'year={args.year}' / f'bar_tb2_{args.type}.png', label)
    scatter_tb2_rtb(top_df, out / f'year={args.year}' / f'scatter_tb2_rtb120_{args.type}.png', label + ' — TB2 vs RTB120')
    # Heatmap monthly
    mon = load_monthly(points, args.year)
    heatmap_monthly(mon, top_df, out / f'year={args.year}' / f'heatmap_monthly_tb2_{args.type}.png', f'{args.year} Monthly TB2 — Top {len(top_df)} {args.type}')
    print('wrote charts to', out / f'year={args.year}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

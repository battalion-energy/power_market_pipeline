#!/usr/bin/env python3
"""
Advanced daily BESS chart with 4 panels:
 1) RT bids/offers vs RT price (15-min)
 2) SOC (estimated) over the day
 3) Hourly revenue breakdown by stream + cumulative total
 4) Dispatch physical/financial decomposition (DA energy, RT deviations, AS capacity revenue)

Data sources: settlement_15min parquet, hourly dispatch parquet, hourly awards parquet, and SCED_Gen_Resources (optional) for offer curves.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_parquets(base: Path, bess: str, year: int):
    p15 = base / 'bess_analysis' / 'settlement_15min' / f'{bess}_{year}_settlement_15min.parquet'
    phd = base / 'bess_analysis' / 'hourly' / 'dispatch' / f'{bess}_{year}_dispatch.parquet'
    pha = base / 'bess_analysis' / 'hourly' / 'awards' / f'{bess}_{year}_awards.parquet'
    return p15, phd, pha


def load_mapping(mapping_path: Path) -> pd.DataFrame:
    df = pd.read_csv(mapping_path)
    cols = {"BESS_Gen_Resource":"bess_name","IQ_Capacity_MW":"capacity_mw","Energy_MWh":"energy_mwh"}
    for k,v in cols.items():
        if k in df.columns:
            df = df.rename(columns={k:v})
    return df


def compute_soc(df15: pd.DataFrame, capacity_mw: float, energy_mwh: float | None, default_hours: float = 2.0) -> pd.Series:
    # SOC estimation integrates net_actual_mwh_15 and clamps to [0, E_mwh]
    if not energy_mwh or energy_mwh <= 0:
        energy_mwh = capacity_mw * default_hours
    mwh15 = df15['net_actual_mwh_15'].fillna(0.0).values
    soc = np.cumsum(mwh15)
    # Normalize to start at 50% if negative at start
    soc = soc - soc.min() + 0.5 * energy_mwh
    soc = np.clip(soc, 0.0, energy_mwh)
    return pd.Series(soc, index=df15.index)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base-dir', required=True)
    p.add_argument('--bess', required=True)
    p.add_argument('--year', type=int, required=True)
    p.add_argument('--date', required=True)
    p.add_argument('--mapping', default='bess_mapping/BESS_UNIFIED_MAPPING_V3_CLARIFIED.csv')
    p.add_argument('--default-hours', type=float, default=2.0)
    args = p.parse_args()

    base = Path(args.base_dir)
    date = pd.to_datetime(args.date).date()
    p15, phd, pha = load_parquets(base, args.bess, args.year)
    if not (p15.exists() and phd.exists() and pha.exists()):
        raise FileNotFoundError('One or more time-series parquet files are missing.')

    df15 = pl.read_parquet(p15).filter(pl.col('local_date') == date).sort('ts_utc')
    if len(df15) == 0:
        raise ValueError('No 15-min rows for selected date.')
    dfh = pl.read_parquet(phd).filter(pl.col('local_date') == date).sort('local_hour')
    dfa = pl.read_parquet(pha).filter(pl.col('local_date') == date).sort('local_hour')

    # Prepare 15-min
    df15 = df15.with_columns([
        (pl.col('actual_mw') * 0.25).alias('net_actual_mwh_15'),
        ((pl.col('bp_gen_mw') - pl.col('bp_load_mw')) * 0.25).alias('net_bp_mwh_15')
    ])
    p15d = df15.to_pandas()
    x15 = p15d['local_hour'].astype(float).values + (pd.to_datetime(p15d['ts_utc']).dt.minute.values / 60.0)

    # Capacity/SOC
    mapdf = load_mapping(Path(args.mapping))
    row = mapdf[mapdf['bess_name'] == args.bess].head(1)
    capacity_mw = float(row['capacity_mw'].values[0]) if not row.empty else 100.0
    energy_mwh = float(row['energy_mwh'].values[0]) if ('energy_mwh' in row and not row.empty and not pd.isna(row['energy_mwh'].values[0])) else None
    soc_series = compute_soc(p15d, capacity_mw, energy_mwh, default_hours=args.default_hours)

    # Hourly revenues + cumulative (align by hour to handle DST/missing hours)
    dph = dfh.to_pandas()
    dfa_h = dfa.fill_null(0.0).to_pandas()
    def align_to_hours(hours, sh, sv):
        m = {int(h): v for h, v in zip(sh, sv)}
        import numpy as np
        return np.array([m.get(int(h), 0.0) for h in hours], dtype=float)
    # Compute AS hourly revenue (award*mcpc) per product
    def rev(df, mw, price):
        return (df.get(mw, 0.0) * df.get(price, 0.0)).fillna(0.0)
    hours = dph['local_hour'].values if 'local_hour' in dph.columns else np.arange(len(dph))
    as_regup = align_to_hours(hours, dfa_h.get('local_hour', []), rev(dfa_h, 'regup_mw', 'regup_mcpc'))
    as_regdown = align_to_hours(hours, dfa_h.get('local_hour', []), rev(dfa_h, 'regdown_mw', 'regdown_mcpc'))
    as_rrs = align_to_hours(hours, dfa_h.get('local_hour', []), rev(dfa_h, 'rrs_mw', 'rrs_mcpc'))
    as_ecrs = align_to_hours(hours, dfa_h.get('local_hour', []), rev(dfa_h, 'ecrs_mw', 'ecrs_mcpc'))
    as_nonspin = align_to_hours(hours, dfa_h.get('local_hour', []), rev(dfa_h, 'nonspin_mw', 'nonspin_mcpc'))
    as_total = as_regup + as_regdown + as_rrs + as_ecrs + as_nonspin
    # Energy components from dispatch parquet
    da_energy = dph.get('da_energy_revenue_hour', pd.Series([0.0]*len(dph))).fillna(0.0).values
    rt_net = dph.get('rt_net_revenue_hour', pd.Series([0.0]*len(dph))).fillna(0.0).values
    total_hour = da_energy + rt_net + as_total
    cum_total = total_hour.cumsum()

    # Create 4-panel figure
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[1,1.1])
    ax_rt = fig.add_subplot(gs[0,0])
    ax_soc = fig.add_subplot(gs[0,1])
    ax_rev = fig.add_subplot(gs[1,0])
    ax_disp = fig.add_subplot(gs[1,1])

    # (1) RT bids/offers vs RT price (from SCED offer curves when available)
    RT_COLOR = '#2E4053'
    DA_COLOR = '#1f77b4'
    ax_rt.plot(x15, p15d['rt_price'].values, color=RT_COLOR, label='RT Price', linewidth=1.6)
    if 'da_price' in p15d.columns:
        ax_rt.plot(x15, p15d['da_price'].values, color=DA_COLOR, label='DA Price', linewidth=1.2, alpha=0.8)

    # Try to reconstruct cleared MW at RT price from SCED offer curves in SCED_Gen_Resources/<year>.parquet
    try:
        sced = pl.read_parquet(base / 'rollup_files' / 'SCED_Gen_Resources' / f'{args.year}.parquet')
        # Filter to this resource and day
        sced = sced.filter(pl.col('ResourceName') == args.bess)
        # parse SCED timestamp to UTC then to date/hour/min
        sced = sced.with_columns([
            pl.col('SCEDTimeStamp').str.strptime(pl.Datetime, '%m/%d/%Y %H:%M:%S').dt.replace_time_zone('America/Chicago').dt.convert_time_zone('UTC').alias('sced_dt')
        ])
        sced = sced.filter(sced['sced_dt'].dt.date() == pd.to_datetime(args.date).date())
        if len(sced) > 0:
            # Map to 15-min bins
            sced = sced.with_columns([
                pl.col('sced_dt').dt.truncate('15m').alias('t15'),
                pl.col('sced_dt').dt.convert_time_zone('America/Chicago').dt.hour().alias('hh'),
                pl.col('sced_dt').dt.minute().alias('mm')
            ])
            # build cleared sell/buy MW at RT price for each 15-min
            cleared_sell = []
            cleared_buy = []
            t_bins = []
            # Pre-extract price/mw columns
            cols = sced.columns
            mw1 = sorted([c for c in cols if c.startswith('SCED1 Curve-MW')])
            pr1 = sorted([c for c in cols if c.startswith('SCED1 Curve-Price')])
            mw2 = sorted([c for c in cols if c.startswith('SCED2 Curve-MW')])
            pr2 = sorted([c for c in cols if c.startswith('SCED2 Curve-Price')])
            # group by 15-min bins (take last SCED run within the 15-min)
            for t, grp in sced.groupby('t15'):
                row = grp.sort('sced_dt')[-1]
                rt_slice = p15d[(pd.to_datetime(p15d['ts_utc']).dt.floor('15min') == t)]
                if rt_slice.empty:
                    continue
                rt_price = float(rt_slice['rt_price'].values[0])
                # SCED1 assumed sell offers (positive MW)
                sell_mw = 0.0
                for mcol,pcol in zip(mw1, pr1):
                    mv = row[mcol]
                    pv = row[pcol]
                    if mv is not None and pv is not None and pv <= rt_price:
                        sell_mw += float(mv)
                # SCED2 assumed buy bids (charge), accumulate MW where bid price >= RT price
                buy_mw = 0.0
                for mcol,pcol in zip(mw2, pr2):
                    mv = row[mcol]
                    pv = row[pcol]
                    if mv is not None and pv is not None and pv >= rt_price:
                        buy_mw += float(mv)
                t_bins.append(float(rt_slice['local_hour'].values[0]) + pd.to_datetime(rt_slice['ts_utc']).dt.minute.values[0]/60.0)
                cleared_sell.append(sell_mw)
                cleared_buy.append(-buy_mw)
            if t_bins:
                ax_rt.bar(t_bins, cleared_sell, width=0.20, color='salmon', alpha=0.35, label='Cleared Sell MW (SCED1)')
                ax_rt.bar(t_bins, cleared_buy, width=0.20, color='lightgray', alpha=0.35, label='Cleared Buy MW (SCED2)')
    except Exception:
        pass
    ax_rt.set_title(f"{date.strftime('%-m/%-d')} RT (proxy bids) | {args.bess}")
    ax_rt.set_ylabel('$/MWh'); ax_rt.set_xlabel('Hour')
    ax_rt.legend(loc='upper left', fontsize=8)
    ax_rt.grid(axis='y', alpha=0.3, linestyle='--')

    # (2) SOC
    ax_soc.plot(x15, soc_series.values, color='#7D3C98', alpha=0.7)
    ax_soc.set_title('SOC (estimated)')
    ax_soc.set_ylabel('MWh'); ax_soc.set_xlabel('Hour')
    ax_soc.grid(axis='y', alpha=0.3, linestyle='--')

    # (3) Hourly revenue breakdown by stream (stacked) + cumulative
    hours = dph['local_hour'].values if 'local_hour' in dph.columns else np.arange(len(total_hour))
    width = 0.8
    DA_COLOR = '#8B7BC8'
    RT_COLOR = '#5FD4AF'
    REGUP_COLOR = '#F4E76E'
    REGDOWN_COLOR = '#F4A6A6'
    RRS_COLOR = '#5DADE2'
    ECRS_COLOR = '#D7BDE2'
    NSPIN_COLOR = '#2C3E50'

    pos_bottom = np.zeros_like(hours, dtype=float)
    # DA energy
    ax_rev.bar(hours, da_energy, width, color=DA_COLOR, alpha=0.85, label='DA Energy')
    pos_bottom = pos_bottom + da_energy
    # RT net
    ax_rev.bar(hours, rt_net, width, bottom=pos_bottom, color=RT_COLOR, alpha=0.85, label='RT Net')
    pos_bottom = pos_bottom + rt_net
    # AS components stacked separately
    ax_rev.bar(hours, as_regup, width, bottom=pos_bottom, color=REGUP_COLOR, alpha=0.9, label='RegUp')
    pos_bottom = pos_bottom + as_regup
    ax_rev.bar(hours, as_regdown, width, bottom=pos_bottom, color=REGDOWN_COLOR, alpha=0.9, label='RegDown')
    pos_bottom = pos_bottom + as_regdown
    ax_rev.bar(hours, as_rrs, width, bottom=pos_bottom, color=RRS_COLOR, alpha=0.9, label='RRS')
    pos_bottom = pos_bottom + as_rrs
    ax_rev.bar(hours, as_ecrs, width, bottom=pos_bottom, color=ECRS_COLOR, alpha=0.9, label='ECRS')
    pos_bottom = pos_bottom + as_ecrs
    ax_rev.bar(hours, as_nonspin, width, bottom=pos_bottom, color=NSPIN_COLOR, alpha=0.9, label='NonSpin')
    pos_bottom = pos_bottom + as_nonspin
    ax2 = ax_rev.twinx()
    ax2.plot(hours, cum_total, color='black', linewidth=1.8, label='Cumulative Total')
    ax_rev.set_title('Revenue Summary (hourly)')
    ax_rev.set_ylabel('$'); ax_rev.set_xlabel('Hour')
    ax_rev.grid(axis='y', alpha=0.3, linestyle='--')
    h1,l1 = ax_rev.get_legend_handles_labels(); h2,l2 = ax2.get_legend_handles_labels(); ax_rev.legend(h1+h2,l1+l2,loc='upper left',fontsize=8)
    ax2.set_ylabel('$ cumulative')

    # (4) Dispatch Physical and Financial
    # Physical MWh bars (actual and basepoint) with financial overlays via hatching for DA/RT components
    ax_disp.bar(hours, dph.get('net_actual_mwh', pd.Series([0.0]*len(hours))), width, color='#5FD4AF', alpha=0.6, label='Actual (MWh)')
    ax_disp.bar(hours, (dph.get('basepoint_gen_mwh', pd.Series(0.0)) - dph.get('basepoint_load_mwh', pd.Series(0.0))), width, color='#2ECC71', alpha=0.4, label='RT Basepoint (MWh)')
    # Financial overlays: DA energy as hatched, RT net as opposite hatch
    ax_disp.bar(hours, da_energy, width, color='none', edgecolor='#8B7BC8', hatch='/', label='DA Energy $')
    ax_disp.bar(hours, rt_net, width, color='none', edgecolor='#1ABC9C', hatch='\\\\', label='RT Net $')
    ax_disp.set_title('Dispatch (Physical & Financial)')
    ax_disp.set_ylabel('MWh / $'); ax_disp.set_xlabel('Hour'); ax_disp.grid(axis='y', alpha=0.3, linestyle='--')
    ax_disp.legend(loc='upper left', fontsize=8)

    out_dir = Path(__file__).parent / 'output' / 'plots_advanced'
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.bess}_{args.date}_advanced.png"
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches='tight')
    print('Saved:', out)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Build partitioned RTB120 (real-time 120-minute top-bottom arbitrage) daily dataset.

Reads ERCOT real-time prices per year and writes per-day Parquet files:
  - rtb120_daily/year=YYYY/date=YYYY-MM-DD.parquet

Also writes a convenience per-year file rtb120_daily_YYYY.parquet.

RTB120 definition: pick the cheapest and most expensive set of intervals that
add up to 120 minutes each side, apply round-trip efficiency (default 0.90),
and compute net $/MW-day revenue. This adapts to 5-minute (12 intervals/hour)
or 15-minute (4 intervals/hour) data automatically.
"""

from __future__ import annotations

import argparse
import os
from dotenv import load_dotenv
from pathlib import Path
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _calc_rtb120_daily(prices: np.ndarray, intervals_per_hour: int, eta: float = 0.90) -> float:
    total_intervals = intervals_per_hour * 24
    if prices.shape[0] != total_intervals:
        return 0.0
    # intervals needed to cover 120 minutes
    minutes_per_interval = 60 / intervals_per_hour
    k = int(round(120 / minutes_per_interval))
    # filter out NaNs; require enough intervals
    p = prices[~np.isnan(prices)]
    if p.size < 2 * k:
        return 0.0
    idx = np.argsort(p)
    charge = p[idx[:k]]
    discharge = p[idx[-k:]]
    hours_per_interval = 1.0 / intervals_per_hour  # 5-min => 1/12, 15-min => 1/4
    charge_cost = charge.sum() * hours_per_interval / eta
    discharge_rev = discharge.sum() * hours_per_interval * eta
    return float(discharge_rev - charge_cost)


def process_year(rt_file: Path, out_root: Path, year: int, efficiency: float, force: bool=False) -> None:
    if not rt_file.exists():
        print(f"[skip] RT prices missing for {year}: {rt_file}")
        return
    print(f"[load-rt] {rt_file}")
    df = pd.read_parquet(rt_file)

    # Normalize date and 5-min index
    # DeliveryDate appears as 'MM/DD/YYYY'
    df['DeliveryDate'] = pd.to_datetime(df['DeliveryDate'], format='%m/%d/%Y', errors='coerce').dt.date
    # Determine intervals per hour (5-min => 12, 15-min => 4)
    if 'DeliveryHour' in df.columns and 'DeliveryInterval' in df.columns:
        intervals_per_hour = int(pd.to_numeric(df['DeliveryInterval'], errors='coerce').max())
        if intervals_per_hour not in (12, 4):
            raise SystemExit(f"Unexpected DeliveryInterval max={intervals_per_hour}; expected 12 (5-min) or 4 (15-min)")
        df['interval_index'] = (df['DeliveryHour'].astype(int) - 1) * intervals_per_hour + df['DeliveryInterval'].astype(int)
    else:
        raise SystemExit("RT parquet missing DeliveryHour/DeliveryInterval columns")

    for day, g in df.groupby('DeliveryDate'):
        part_dir = out_root / 'rtb120_daily' / f"year={year}"
        _ensure_dir(part_dir)
        out_path = part_dir / f"date={day}.parquet"
        if out_path.exists() and not force:
            continue

        # Keep type per settlement point
        if 'SettlementPointType' in g.columns:
            tm = g[['SettlementPointName', 'SettlementPointType']].drop_duplicates()
            type_map = dict(zip(tm['SettlementPointName'], tm['SettlementPointType']))
        else:
            type_map = None

        piv = g.pivot_table(index='SettlementPointName', columns='interval_index', values='SettlementPointPrice', aggfunc='first')
        total_intervals = intervals_per_hour * 24
        # Ensure all intervals exist
        for i in range(1, total_intervals + 1):
            if i not in piv.columns:
                piv[i] = np.nan
        piv = piv[[i for i in range(1, total_intervals + 1)]]

        vals = piv.values
        rtb = np.apply_along_axis(_calc_rtb120_daily, 1, vals, intervals_per_hour, efficiency)
        sps = piv.index.to_numpy()
        if type_map is not None:
            sptypes = np.array([type_map.get(sp, 'UNKNOWN') for sp in sps])
        else:
            sptypes = np.array(['UNKNOWN'] * len(sps))
        date_col = np.full(sps.shape[0], day, dtype='datetime64[D]')
        tbl = pa.table({
            'SettlementPoint': pa.array(sps, type=pa.string()),
            'SettlementPointType': pa.array(sptypes, type=pa.string()),
            'DeliveryDate': pa.array(date_col, type=pa.date32()),
            'RTB120': pa.array(rtb, type=pa.float64())
        })
        pq.write_table(tbl, out_path)

    # Per-year convenience file
    ydir = out_root / 'rtb120_daily' / f"year={year}"
    if ydir.exists():
        ds = pa.dataset.dataset(str(ydir), format='parquet')
        pq.write_table(ds.to_table(), out_root / f"rtb120_daily_{year}.parquet")
        print(f"[year] wrote rtb120_daily_{year}.parquet")


def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', nargs='*', type=int, default=[2025])
    default_data = os.getenv('ERCOT_DATA_DIR', '/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data')
    ap.add_argument('--rt-dir', default=str(Path(default_data) / 'rollup_files/RT_prices'))
    ap.add_argument('--out-dir', default=str(Path(default_data) / 'tbx'))
    ap.add_argument('--efficiency', type=float, default=0.90)
    ap.add_argument('--force', action='store_true', help='Overwrite existing per-day files')
    args = ap.parse_args()

    rt_dir = Path(args.rt_dir)
    out_root = Path(args.out_dir)
    for y in args.years:
        process_year(rt_dir / f"{y}.parquet", out_root, y, args.efficiency, args.force)
    print("done rtb120.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

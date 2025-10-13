#!/usr/bin/env python3
"""
Build partitioned TB1/TB2/TB4 daily datasets from ERCOT DA prices.

Writes per-day Parquet files under:
  - tb1_daily/year=YYYY/date=YYYY-MM-DD.parquet
  - tb2_daily/year=YYYY/date=YYYY-MM-DD.parquet
  - tb4_daily/year=YYYY/date=YYYY-MM-DD.parquet

Also emits optional per-year convenience files (tb?_daily_YYYY.parquet)
containing all daily rows for that year.

This script is idempotent and incremental: it skips days whose output files
already exist, so you can run it daily after appending new DA prices.
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


def _calc_tbk_daily(prices_24: np.ndarray, hours: int, eta: float = 0.90) -> float:
    """Heuristic perfect-foresight arbitrage for k-hour system for a single day.

    - Choose 'hours' cheapest hours to charge and 'hours' most expensive to discharge.
    - Apply round-trip efficiency split evenly to charge/discharge sides.
    Returns $/MW-day revenue.
    """
    if prices_24.shape[0] != 24 or np.any(np.isnan(prices_24)):
        return 0.0
    idx = np.argsort(prices_24)
    charge = idx[:hours]
    discharge = idx[-hours:]
    charge_cost = prices_24[charge].sum() / eta
    discharge_rev = prices_24[discharge].sum() * eta
    return float(discharge_rev - charge_cost)


def process_year(da_file: Path, out_root: Path, year: int, efficiency: float) -> None:
    if not da_file.exists():
        print(f"[skip] DA prices missing for {year}: {da_file}")
        return
    print(f"[load] {da_file}")
    df = pd.read_parquet(da_file)

    # Normalize columns
    if 'hour' in df.columns:
        # hour may be '01:00' or numeric
        if df['hour'].dtype == object:
            hr = pd.to_datetime(df['hour'].str.replace('24:00', '00:00'), format='%H:%M').dt.hour
            hr = hr.replace(0, 24)
        else:
            hr = df['hour'].astype(int)
    elif 'HourEnding' in df.columns:
        he = df['HourEnding']
        if he.dtype == object:
            hr = pd.to_datetime(he.str.replace('24:00', '00:00'), format='%H:%M').dt.hour
            hr = hr.replace(0, 24)
        else:
            hr = he.astype(int)
    else:
        raise SystemExit("DA parquet must contain 'hour' or 'HourEnding'")
    df = df.assign(hour=hr,
                   DeliveryDate=pd.to_datetime(df['DeliveryDate']).dt.date)

    # Daily loop for incremental writes
    for day, g in df.groupby('DeliveryDate'):
        # Partition path
        part_dir1 = out_root / 'tb1_daily' / f"year={year}"
        part_dir2 = out_root / 'tb2_daily' / f"year={year}"
        part_dir4 = out_root / 'tb4_daily' / f"year={year}"
        _ensure_dir(part_dir1)
        _ensure_dir(part_dir2)
        _ensure_dir(part_dir4)

        f1 = part_dir1 / f"date={day}.parquet"
        f2 = part_dir2 / f"date={day}.parquet"
        f4 = part_dir4 / f"date={day}.parquet"
        if f1.exists() and f2.exists() and f4.exists():
            continue

        # Pivot to 24-hour vectors per node
        # Keep type for each settlement point
        if 'SettlementPointType' in g.columns:
            type_map = g[['SettlementPoint', 'SettlementPointType']].drop_duplicates().set_index('SettlementPoint')['SettlementPointType']
        else:
            # default unknown type
            type_map = None

        piv = g.pivot_table(index='SettlementPoint', columns='hour', values='SettlementPointPrice', aggfunc='first')
        # Ensure all 24 hours
        for h in range(1, 25):
            if h not in piv.columns:
                piv[h] = np.nan
        piv = piv[[h for h in range(1, 25)]]

        # Compute TB1/TB2/TB4
        pvals = piv.values
        tb1 = np.apply_along_axis(_calc_tbk_daily, 1, pvals, 1, efficiency)
        tb2 = np.apply_along_axis(_calc_tbk_daily, 1, pvals, 2, efficiency)
        tb4 = np.apply_along_axis(_calc_tbk_daily, 1, pvals, 4, efficiency)

        # Build Arrow tables and write
        sps = piv.index.to_numpy()
        if type_map is not None:
            sptypes = type_map.reindex(piv.index).fillna('UNKNOWN').to_numpy()
        else:
            sptypes = np.array(['UNKNOWN'] * len(sps))
        date_col = np.full(sps.shape[0], day, dtype='datetime64[D]')
        def write_file(path: Path, values: np.ndarray, colname: str):
            tbl = pa.table({
                'SettlementPoint': pa.array(sps, type=pa.string()),
                'SettlementPointType': pa.array(sptypes, type=pa.string()),
                'DeliveryDate': pa.array(date_col, type=pa.date32()),
                colname: pa.array(values, type=pa.float64())
            })
            pq.write_table(tbl, path)

        if not f1.exists():
            write_file(f1, tb1, 'TB1')
        if not f2.exists():
            write_file(f2, tb2, 'TB2')
        if not f4.exists():
            write_file(f4, tb4, 'TB4')

    # Optional per-year combined files for convenience
    for name in ('tb1_daily', 'tb2_daily', 'tb4_daily'):
        ydir = out_root / name / f"year={year}"
        out_year = out_root / f"{name}_{year}.parquet"
        lf = pa.dataset.dataset(str(ydir), format="parquet") if ydir.exists() else None
        if lf:
            # Materialize concat to a single file for the year
            pq.write_table(lf.to_table(), out_year)
            print(f"[year] wrote {out_year}")


def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', nargs='*', type=int, default=list(range(2019, 2026)))
    default_data = os.getenv('ERCOT_DATA_DIR', '/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data')
    ap.add_argument('--da-dir', default=str(Path(default_data) / 'rollup_files/DA_prices'))
    ap.add_argument('--out-dir', default=str(Path(default_data) / 'tbx'))
    ap.add_argument('--efficiency', type=float, default=0.90)
    args = ap.parse_args()

    da_dir = Path(args.da_dir)
    out_root = Path(args.out_dir)
    for y in args.years:
        process_year(da_dir / f"{y}.parquet", out_root, y, args.efficiency)
    print("done.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

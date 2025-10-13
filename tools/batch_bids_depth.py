#!/usr/bin/env python3
"""
Batch-generate RT and DA bid-depth heatmaps for many BESS days.

Strategy
- Use existing hourly dispatch files to pick dates that likely have curve data.
- For each selected (bess, date), call the RT and/or DA plotting scripts.
- Skip gracefully on any failure (missing data, etc.).

Defaults generate 50 RT and 50 DA charts into tools/output/{rt,da}_bids_depth.

Examples
  python tools/batch_bids_depth.py \
    --base-dir /pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data \
    --year 2024 --count-rt 50 --count-da 50
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import subprocess
import sys
import polars as pl


def pick_dates_from_dispatch(dispatch_path: Path, max_days: int = 20) -> list[str]:
    try:
        df = pl.read_parquet(dispatch_path)
    except Exception:
        return []
    if len(df) == 0 or 'local_date' not in df.columns:
        return []
    dates = (
        df.select(pl.col('local_date'))
          .unique()
          .sort('local_date')
          .to_series()
          .dt.strftime('%Y-%m-%d')
          .to_list()
    )
    if not dates:
        return []
    # spread selections across the year
    step = max(1, len(dates) // max(1, max_days))
    return [dates[i] for i in range(0, len(dates), step)][:max_days]


def gather_targets(base: Path, year: int, need: int) -> list[tuple[str, str]]:
    """Return up to `need` (bess, date) pairs by scanning dispatch files."""
    ddir = base / 'bess_analysis' / 'hourly' / 'dispatch'
    pairs: list[tuple[str, str]] = []
    for dp in sorted(ddir.glob(f'*_{year}_dispatch.parquet')):
        bess = dp.name.replace(f'_{year}_dispatch.parquet', '')
        for day in pick_dates_from_dispatch(dp, max_days=6):
            pairs.append((bess, day))
            if len(pairs) >= need:
                return pairs
    return pairs


def run_rt(base: Path, bess: str, year: int, date: str, out_dir: Path | None) -> bool:
    cmd = [
        sys.executable, str(Path(__file__).parent / 'plot_rt_bids_depth.py'),
        '--base-dir', str(base), '--bess', bess, '--year', str(year), '--date', date,
        '--kink-mw', '1', '--fast-frac', '0.9'
    ]
    if out_dir is not None:
        cmd.extend(['--out-dir', str(out_dir)])
    subprocess.run(cmd, check=False)
    # Determine expected output path and report success
    dest_dir = out_dir if out_dir is not None else (Path(__file__).parent / 'output' / 'rt_bids_depth')
    return (dest_dir / f"{bess}_{date}_rt_bids.png").exists()


def run_da(base: Path, mapping: Path, bess: str, year: int, date: str, out_dir: Path | None) -> bool:
    cmd = [
        sys.executable, str(Path(__file__).parent / 'plot_da_bids_depth.py'),
        '--base-dir', str(base), '--bess', bess, '--year', str(year), '--date', date,
        '--mapping', str(mapping), '--kink-mw', '1', '--fast-frac', '0.9'
    ]
    if out_dir is not None:
        cmd.extend(['--out-dir', str(out_dir)])
    subprocess.run(cmd, check=False)
    dest_dir = out_dir if out_dir is not None else (Path(__file__).parent / 'output' / 'da_bids_depth')
    return (dest_dir / f"{bess}_{date}_da_bids.png").exists()


def main() -> int:
    ap = argparse.ArgumentParser(description='Batch-generate RT/DA bid-depth charts')
    ap.add_argument('--base-dir', required=True)
    ap.add_argument('--mapping', default=None, help='Path to BESS mapping CSV (for DA). If omitted, inferred from base-dir/bess_mapping/*.csv')
    ap.add_argument('--year', type=int, default=2024)
    ap.add_argument('--count-rt', type=int, default=50)
    ap.add_argument('--count-da', type=int, default=50)
    ap.add_argument('--seed', type=int, default=42)
    # None means let child scripts use their own defaults (relative to their repo)
    ap.add_argument('--out-rt', dest='out_rt', default=None)
    ap.add_argument('--out-da', dest='out_da', default=None)
    args = ap.parse_args()

    random.seed(args.seed)
    base = Path(args.base_dir)

    # Resolve mapping for DA
    mapping_path: Path | None = Path(args.mapping) if args.mapping else None
    if mapping_path is None:
        bm = base / 'bess_mapping'
        cands = sorted(bm.glob('BESS_UNIFIED_MAPPING_*.csv'))
        mapping_path = cands[-1] if cands else None

    # Pick candidate (bess, date) pairs
    needed = max(args.count_rt, args.count_da)
    pairs = gather_targets(base, args.year, need=needed * 6)  # over-sample for failures
    random.shuffle(pairs)

    rt_done = 0
    da_done = 0
    for bess, day in pairs:
        if rt_done < args.count_rt:
            ok = run_rt(base, bess, args.year, day, Path(args.out_rt) if args.out_rt else None)
            if ok:
                rt_done += 1
        if da_done < args.count_da and mapping_path is not None:
            ok = run_da(base, mapping_path, bess, args.year, day, Path(args.out_da) if args.out_da else None)
            if ok:
                da_done += 1
        if rt_done >= args.count_rt and da_done >= args.count_da:
            break

    print(f'Generated ~{rt_done} RT and ~{da_done} DA charts.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

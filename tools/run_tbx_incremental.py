#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from datetime import date
from dotenv import load_dotenv


def run(cmd: list[str]):
    print('>', ' '.join(cmd))
    subprocess.run(cmd, check=False)


def main() -> int:
    load_dotenv()
    ercot = os.getenv('ERCOT_DATA_DIR', '/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data')
    charts = os.getenv('CHARTS_OUTPUT_DIR', 'charts_output')
    ap = argparse.ArgumentParser(description='Incremental TBX build (idempotent)')
    ap.add_argument('--start-year', type=int, default=2019)
    ap.add_argument('--end-year', type=int, default=date.today().year)
    ap.add_argument('--with-charts', action='store_true', help='Rebuild reports/rankings and chart packs')
    args = ap.parse_args()

    years = [str(y) for y in range(args.start_year, args.end_year + 1)]

    # 1) Build/refresh DA TBX daily
    run(['python', 'tools/tbx_build_partitioned.py', '--years', *years])

    # 2) Build/refresh RTB120 daily
    run(['python', 'tools/rtb120_build_partitioned.py', '--years', *years])

    # 3) Rollups (points + COD-aware for recent years)
    run(['python', 'tools/tbx_points_rollup.py', '--years', *years])
    recent = [str(y) for y in range(max(args.start_year, args.end_year - 3), args.end_year + 1)]
    run(['python', 'tools/tbx_rollup_cod_aware.py', '--years', *recent])

    if args.with_charts:
        # 4) Reports + rankings (top sites, periodic, full, topN)
        run(['python', 'tools/tbx_top_sites_report.py', '--years', *years, '--top', '50'])
        run(['python', 'tools/tbx_full_rankings.py', '--years', *years])
        run(['python', 'tools/tbx_rankings_periodic.py', '--years', *years])
        run(['python', 'tools/tbx_rankings_topN.py', '--years', *years, '--start-year', '2022', '--end-year', str(args.end_year), '--top', '50'])
        # 5) Chart packs (latest two years)
        run(['python', 'tools/build_chart_packs.py', '--years', str(args.end_year-1), str(args.end_year), '--types', 'RN', 'HUB', 'LZ', '--top', '30'])

    print('Incremental TBX build completed.')
    print('ERCOT_DATA_DIR:', ercot)
    print('CHARTS_OUTPUT_DIR:', charts)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv


def main() -> int:
    load_dotenv()
    ercot = os.getenv('ERCOT_DATA_DIR', '/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data')
    charts = os.getenv('CHARTS_OUTPUT_DIR', 'charts_output')
    tbx = Path(ercot) / 'tbx'
    points = Path(ercot) / 'tbx_points_rollup'
    cod = Path(ercot) / 'tbx_rollup'
    chart_imgs = Path(charts) / 'tbx_charts'
    chart_packs = Path(charts) / 'tbx_chart_packs'
    reports = Path(charts) / 'tbx_reports'
    rankings = Path(charts) / 'tbx_rankings'
    period_charts = Path(charts) / 'bess_revenue_charts_periods'

    print('TBX ENV SUMMARY')
    print('  ERCOT_DATA_DIR     =', ercot)
    print('  CHARTS_OUTPUT_DIR  =', charts)
    print('\nDATA INPUTS / OUTPUTS (under ERCOT_DATA_DIR)')
    print('  TBX daily          =', tbx)
    print('  Points rollup      =', points)
    print('  COD-aware rollup   =', cod)
    print('\nCHARTS / REPORTS (under CHARTS_OUTPUT_DIR)')
    print('  Visual charts      =', chart_imgs)
    print('  PDF chart packs    =', chart_packs)
    print('  Top-site reports   =', reports)
    print('  Rankings           =', rankings)
    print('  BESS period charts =', period_charts)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


#!/usr/bin/env python3
"""
Compose a single daily dashboard image by tiling all per-day charts we generate:
 - Hourly awards + dispatch (tools/plot_daily_bess.py)
 - 15-min dispatch + prices (tools/plot_daily_bess_15min.py)
 - Advanced 4-panel (tools/plot_daily_bess_advanced.py)
 - RT bids depth (tools/plot_rt_bids_depth.py)
 - DA bids depth (tools/plot_da_bids_depth.py)

Usage:
  python tools/compose_daily_dashboard.py \
    --bess CROSSETT_BES1 --date 2024-02-11

Expects charts in the default output folders created by each tool:
  tools/output/plots/<BESS>_<date>_daily.png
  tools/output/plots_15min/<BESS>_<date>_15min.png
  tools/output/plots_advanced/<BESS>_<date>_advanced.png
  tools/output/rt_bids_depth/<BESS>_<date>_rt_bids.png
  tools/output/da_bids_depth/<BESS>_<date>_da_bids.png

Saves:
  tools/output/daily_dashboards/<BESS>_<date>_dashboard.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess, sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def load_img(path: Path):
    if path.exists():
        try:
            return mpimg.imread(path)
        except Exception:
            return None
    return None


def gen(cmd: list[str]):
    try:
        subprocess.run(cmd, check=False)
    except Exception:
        pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bess', required=True)
    p.add_argument('--date', required=True)
    p.add_argument('--out', default='tools/output/daily_dashboards')
    # Optional auto-generate of missing sources
    p.add_argument('--gen-missing', action='store_true', help='Generate missing panels automatically')
    p.add_argument('--base-dir', default='/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data')
    p.add_argument('--year', type=int, default=2024)
    p.add_argument('--mapping', default='bess_mapping/BESS_UNIFIED_MAPPING_V3_CLARIFIED.csv')
    args = p.parse_args()

    bess = args.bess
    date = args.date
    base = Path('tools/output')

    # Expected inputs
    charts = [
        ('Hourly Awards & Dispatch', base / 'plots' / f'{bess}_{date}_daily.png'),
        ('15-min Dispatch & Prices', base / 'plots_15min' / f'{bess}_{date}_15min.png'),
        ('Advanced (4-panel)', base / 'plots_advanced' / f'{bess}_{date}_advanced.png'),
        ('RT Bids Depth', base / 'rt_bids_depth' / f'{bess}_{date}_rt_bids.png'),
        ('DA Bids Depth', base / 'da_bids_depth' / f'{bess}_{date}_da_bids.png'),
    ]

    # Layout: 2 rows x 3 columns (last cell for notes if image missing)
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axlist = axes.flatten()
    used = 0
    for title, path in charts:
        img = load_img(path)
        ax = axlist[used]
        if img is None and args.gen_missing:
            # Attempt to generate the missing chart on the fly
            if 'Hourly Awards' in title:
                gen([sys.executable, 'tools/plot_daily_bess.py', '--base-dir', args.base_dir, '--bess', bess, '--year', str(args.year), '--date', date])
            elif '15-min Dispatch' in title:
                gen([sys.executable, 'tools/plot_daily_bess_15min.py', '--base-dir', args.base_dir, '--bess', bess, '--year', str(args.year), '--date', date])
            elif 'Advanced' in title:
                gen([sys.executable, 'tools/plot_daily_bess_advanced.py', '--base-dir', args.base_dir, '--bess', bess, '--year', str(args.year), '--date', date, '--mapping', args.mapping])
            elif 'RT Bids Depth' in title:
                gen([sys.executable, 'tools/plot_rt_bids_depth.py', '--base-dir', args.base_dir, '--bess', bess, '--year', str(args.year), '--date', date, '--pmin', '-250', '--pmax', '250', '--pstep', '5', '--out-dir', str(base / 'rt_bids_depth')])
            elif 'DA Bids Depth' in title:
                gen([sys.executable, 'tools/plot_da_bids_depth.py', '--base-dir', args.base_dir, '--bess', bess, '--year', str(args.year), '--date', date, '--mapping', args.mapping, '--pmin', '-250', '--pmax', '250', '--pstep', '5', '--out-dir', str(base / 'da_bids_depth')])
            # Reload if generated
            img = load_img(path)

        if img is not None:
            ax.imshow(img)
            ax.set_title(title, fontsize=12)
            ax.axis('off')
        else:
            ax.text(0.5, 0.5, f'Missing:\n{path}', ha='center', va='center', fontsize=11)
            ax.set_title(title, fontsize=12)
            ax.axis('off')
        used += 1
        if used >= len(axlist):
            break

    # If any empty cells remain, turn off
    for i in range(used, len(axlist)):
        axlist[i].axis('off')

    fig.suptitle(f'{bess} — {date}', fontsize=16)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{bess}_{date}_dashboard.png'
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(out_path, dpi=150)
    print('Saved:', out_path)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
import shutil
from dotenv import load_dotenv


def safe_move(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        try:
            dst.unlink()
        except Exception:
            pass
    shutil.move(str(src), str(dst))


def main() -> int:
    repo = Path.cwd()
    load_dotenv(dotenv_path=repo / '.env')
    ercot = Path(os.getenv('ERCOT_DATA_DIR', '/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data'))
    charts = Path(os.getenv('CHARTS_OUTPUT_DIR', 'charts_output'))

    moved = 0
    deleted = 0

    # Move bess revenue CSVs to ERCOT_DATA_DIR/bess_revenue
    rev_dir = ercot / 'bess_revenue'
    for p in repo.glob('bess_revenue_*.csv'):
        safe_move(p, rev_dir / p.name)
        moved += 1

    # Move stray TBX daily parquets into ERCOT_DATA_DIR/tbx
    tbx_dir = ercot / 'tbx'
    for p in repo.glob('tb?_daily_*.parquet'):
        safe_move(p, tbx_dir / p.name)
        moved += 1

    # Delete leftover TBX/TB120 folders at repo root if any (they should be empty after prior moves)
    for name in ['tb1_daily','tb2_daily','tb4_daily','rtb120_daily',
                 'tbx_points_rollup','tbx_rollup','tbx_charts','tbx_chart_packs',
                 'tbx_reports','tbx_rankings','bess_revenue_charts_periods']:
        path = repo / name
        if path.exists():
            for f in path.rglob('*'):
                if f.is_file():
                    try:
                        f.unlink()
                        deleted += 1
                    except Exception:
                        pass
            for d in sorted(path.glob('**/*'), reverse=True):
                if d.is_dir():
                    try:
                        d.rmdir()
                    except Exception:
                        pass
            try:
                path.rmdir()
            except Exception:
                pass

    print(f"Moved {moved} files; deleted {deleted} files from repo root.")
    print("ERCOT_DATA_DIR:", ercot)
    print("CHARTS_OUTPUT_DIR:", charts)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


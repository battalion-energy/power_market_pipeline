#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from dotenv import load_dotenv
from pathlib import Path
import subprocess
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def run_vis(year: int, sp_type: str, top: int, points_dir: Path, out_dir: Path) -> list[Path]:
    cmd = [
        "python", str(Path(__file__).parent / "visualize_best_sites.py"),
        "--year", str(year), "--type", sp_type, "--top", str(top),
        "--points-dir", str(points_dir), "--out-dir", str(out_dir),
    ]
    subprocess.run(cmd, check=False)
    ydir = out_dir / f"year={year}"
    # expected files
    files = [
        ydir / f"bar_tb2_{sp_type}.png",
        ydir / f"scatter_tb2_rtb120_{sp_type}.png",
        ydir / f"heatmap_monthly_tb2_{sp_type}.png",
    ]
    return [p for p in files if p.exists()]


def make_pdf_pack(year: int, sp_type: str, image_paths: list[Path], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(dest) as pdf:
        for img_path in image_paths:
            try:
                fig = plt.figure(figsize=(12, 8))
                ax = fig.add_subplot(111)
                ax.axis('off')
                img = mpimg.imread(str(img_path))
                ax.imshow(img)
                ax.set_title(f"{year} {sp_type}: {img_path.name}")
                pdf.savefig(fig)
                plt.close(fig)
            except Exception:
                continue


def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser(description="Build chart packs (PDF) for best sites")
    ap.add_argument('--years', nargs='*', type=int, default=[2022, 2023, 2024, 2025])
    ap.add_argument('--types', nargs='*', default=['RN','HUB','LZ'])
    ap.add_argument('--top', type=int, default=30)
    default_data = os.getenv('ERCOT_DATA_DIR', '/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data')
    charts_dir = os.getenv('CHARTS_OUTPUT_DIR', 'charts_output')
    ap.add_argument('--points-dir', default=str(Path(default_data) / 'tbx_points_rollup'))
    ap.add_argument('--out-dir', default=str(Path(charts_dir) / 'tbx_charts'))
    ap.add_argument('--packs-dir', default=str(Path(charts_dir) / 'tbx_chart_packs'))
    args = ap.parse_args()

    points_dir = Path(args.points_dir)
    out_dir = Path(args.out_dir)
    packs_dir = Path(args.packs_dir)

    for y in args.years:
        for t in args.types:
            imgs = run_vis(y, t, args.top, points_dir, out_dir)
            if imgs:
                make_pdf_pack(y, t, imgs, packs_dir / f"{y}_best_sites_{t}.pdf")
    print("Chart packs written to", packs_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

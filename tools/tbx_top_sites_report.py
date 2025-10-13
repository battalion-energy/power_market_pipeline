#!/usr/bin/env python3
"""
Top sites report: rank RN/LZ/HUB/SP by tb2_avg_day, tb4_avg_day, rtb120_avg_day.

Reads annual point rollups and writes per-metric CSVs + a small Markdown summary.
"""

from __future__ import annotations

import argparse
import os
from dotenv import load_dotenv
from pathlib import Path
import polars as pl


def _normalize_types(df: pl.DataFrame) -> pl.DataFrame:
    if 'SettlementPointType' not in df.columns:
        return df
    t = pl.col('SettlementPointType')
    typ = (
        pl.when(t.is_in(['HU','AH','SH']))
          .then(pl.lit('HUB'))
          .when(t.is_in(['LZ','LZ_DC','LZ_DCEW','LZEW']))
          .then(pl.lit('LZ'))
          .when(t.is_in(['RN','PCCRN','LCCRN']))
          .then(pl.lit('RN'))
          .otherwise(t)
          .alias('TypeNorm')
    )
    return df.with_columns(typ)


def top_n(df: pl.DataFrame, metric: str, sp_type: str | None, n: int) -> pl.DataFrame:
    d = df
    if sp_type and 'TypeNorm' in d.columns:
        d = d.filter(pl.col('TypeNorm') == sp_type)
    cols = ['SettlementPoint'] + (["TypeNorm"] if 'TypeNorm' in d.columns else []) + [metric]
    if metric not in d.columns:
        return pl.DataFrame({c: [] for c in cols})
    out = d.select(cols).sort(metric, descending=True).head(n)
    if 'TypeNorm' in out.columns:
        out = out.rename({'TypeNorm':'SettlementPointType'})
    return out


def run_for_year(year: int, points_dir: Path, out_dir: Path, top: int) -> None:
    ann_path = points_dir / 'annual' / f'year={year}.parquet'
    if not ann_path.exists():
        print(f"[report] missing {ann_path}")
        return
    df = pl.read_parquet(ann_path)
    df = _normalize_types(df)
    outy = out_dir / f'year={year}'
    outy.mkdir(parents=True, exist_ok=True)

    metrics = ['tb2_avg_day', 'tb4_avg_day', 'rtb120_avg_day']
    # discover types present to avoid empty files
    types_present = []
    if 'TypeNorm' in df.columns:
        types_present = [t for t in df.select('TypeNorm').unique()['TypeNorm'].to_list() if t]
    # use canonical list but only those present
    canonical = ['RN','LZ','HUB','PUN']
    types = [t for t in canonical if t in types_present]
    labels = types + ['ALL', 'SPP']

    md = []
    md.append(f"# Top Sites — {year}\n")
    for metric in metrics:
        md.append(f"## {metric}\n")
        for label in labels:
            tp = None if label in ('ALL','SPP') else label
            tdf = top_n(df, metric, tp, top)
            if tdf.height == 0:
                continue
            # for SPP emit alias file equal to ALL
            out_name = f"top_{metric}_{label}.csv"
            csvp = outy / out_name
            tdf.write_csv(csvp)
            md.append(f"- {metric} {label}: {csvp.name}")
        md.append("")

    (outy / 'README.md').write_text("\n".join(md))
    print(f"[report] wrote top-site CSVs for {year} -> {outy}")


def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', nargs='*', type=int, default=[2025])
    default_data = os.getenv('ERCOT_DATA_DIR', '/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data')
    charts_dir = os.getenv('CHARTS_OUTPUT_DIR', 'charts_output')
    ap.add_argument('--points-dir', default=str(Path(default_data) / 'tbx_points_rollup'))
    ap.add_argument('--out-dir', default=str(Path(charts_dir) / 'tbx_reports'))
    ap.add_argument('--top', type=int, default=25)
    args = ap.parse_args()

    points = Path(args.points_dir)
    out = Path(args.out_dir)
    for y in args.years:
        run_for_year(y, points, out, args.top)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

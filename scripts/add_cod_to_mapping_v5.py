"""
Add COD dates to BESS_UNIFIED_MAPPING_V5.csv by scanning ERCOT SCED parquet data.

Behavior:
- If V5 does not exist, copy from V4 and add a new column `COD_Date` (YYYY-MM-DD).
- If V5 exists but lacks `COD_Date`, add it. Use `--overwrite` to recompute all.
- Activity definition: BasePoint non-null and non-zero (ignore telemetry).
- Year search: scans ALL available years within optional bounds and returns the
  earliest BasePoint timestamp for the resource (Gen and Load independently),
  then takes the earliest across Gen/Load.

Data assumptions:
- SCED Gen parquet path: <rollup_dir>/SCED_Gen_Resources/YYYY.parquet
- SCED Load parquet path: <rollup_dir>/SCED_Load_Resources/YYYY.parquet
- Columns used: ResourceName, BasePoint, datetime (preferred) or SCEDTimeStamp.

CLI:
  python scripts/add_cod_to_mapping_v5.py \
      --v4 bess_mapping/BESS_UNIFIED_MAPPING_V4.csv \
      --v5 bess_mapping/BESS_UNIFIED_MAPPING_V5.csv \
      --rollup /pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/rollup_files [--overwrite]

This script is idempotent by default and only fills missing COD_Date values unless --overwrite is provided.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import polars as pl
import pandas as pd


def detect_years(dir_path: Path) -> list[int]:
    """Return sorted list of years available as <year>.parquet under dir_path."""
    years = []
    if not dir_path.exists():
        return years
    for p in dir_path.glob("*.parquet"):
        try:
            y = int(p.stem)
            years.append(y)
        except ValueError:
            continue
    return sorted(set(years))


ACCEPTABLE_GEN_TYPES = {"PWRSTR", "BATTERY", "BESS", "STORAGE"}


def first_activity_datetime_in_file(parquet_file: Path, resource: str, file_year: Optional[int] = None, require_battery: bool = False) -> Optional[pd.Timestamp]:
    """Return earliest activity timestamp for a single parquet file (or None).

    Activity is defined as any row for the ResourceName where at least one of:
    - BasePoint is not null
    - TelemeteredNetOutput is not null (gen only, if present)

    Uses `datetime` column when present; else falls back to `SCEDTimeStamp`.
    """
    # Build the minimal projection; some columns may not exist in all years
    base_cols = ["ResourceName", "datetime", "SCEDTimeStamp", "ResourceType"]
    cond_cols = ["BasePoint"]
    cols = list(dict.fromkeys(base_cols + cond_cols))

    try:
        lf = pl.scan_parquet(str(parquet_file)).select([c for c in cols if c in lf.collect_schema().names()])
    except Exception:
        # If schema inference fails, attempt simple scan then intersect columns
        lf = pl.scan_parquet(str(parquet_file))
        lf = lf.select([c for c in cols if c in lf.collect_schema().names()])

    names = lf.collect_schema().names()
    if "ResourceName" not in names:
        return None

    cond = pl.col("ResourceName") == resource
    if require_battery and "ResourceType" in names:
        cond = cond & pl.col("ResourceType").is_in(list(ACCEPTABLE_GEN_TYPES))

    has_basepoint = "BasePoint" in names
    if has_basepoint:
        bp = pl.col("BasePoint").cast(pl.Float64, strict=False)
        activity = bp.is_not_null() & (bp != 0)
        filt = cond & activity
    else:
        # If BasePoint column is missing, there is no valid activity per spec
        return None

    ts_col = "datetime" if "datetime" in names else ("SCEDTimeStamp" if "SCEDTimeStamp" in names else None)
    if ts_col is None:
        return None

    # Collect the timestamp column and parse via pandas for robustness across formats
    try:
        out_pl = lf.filter(filt).select(pl.col(ts_col).alias("ts")).collect()
    except Exception:
        return None
    if out_pl.height == 0:
        return None
    try:
        s = out_pl.to_pandas()["ts"]
        # Pandas now uses strict format inference by default; no need for infer_datetime_format
        ts = pd.to_datetime(s, errors="coerce")
        ts = ts.dropna()
        if file_year is not None:
            ts = ts[ts.dt.year == file_year]
        if ts.empty:
            return None
        return ts.min()
    except Exception:
        return None


def earliest_activity_for_resource_all_years(resource: str, dir_years: list[tuple[Path, list[int]]], require_battery: bool = False) -> Optional[pd.Timestamp]:
    """Return the earliest BasePoint timestamp across ALL available years.

    Only considers timestamps within each file's own year to avoid cross-year bleed.
    """
    if not dir_years:
        return None
    earliest: Optional[pd.Timestamp] = None
    for dir_path, years in dir_years:
        for y in sorted(set(years)):
            p = dir_path / f"{y}.parquet"
            if not p.exists():
                continue
            ts = first_activity_datetime_in_file(p, resource, file_year=y, require_battery=require_battery)
            if ts is None:
                continue
            if earliest is None or ts < earliest:
                earliest = ts
    return earliest


def resolve_cod_for_row(row: pd.Series, gen_dirs: list[tuple[Path, list[int]]], load_dirs: list[tuple[Path, list[int]]]) -> Optional[str]:
    """Compute COD date (YYYY-MM-DD) for a single mapping row."""
    gen_name = str(row.get("BESS_Gen_Resource") or "").strip()
    load_name = str(row.get("BESS_Load_Resource") or "").strip()

    candidates: list[pd.Timestamp] = []

    if gen_name:
        g = earliest_activity_for_resource_all_years(gen_name, gen_dirs, require_battery=True)
        if g is not None:
            candidates.append(g)

    if load_name:
        l = earliest_activity_for_resource_all_years(load_name, load_dirs)
        if l is not None:
            candidates.append(l)

    if not candidates:
        return None
    cod = min(candidates).normalize()  # midnight UTC
    return cod.strftime("%Y-%m-%d")


def main():
    ap = argparse.ArgumentParser(description="Add COD_Date to BESS_UNIFIED_MAPPING_V5.csv using SCED parquet")
    ap.add_argument("--v4", default="bess_mapping/BESS_UNIFIED_MAPPING_V4.csv", help="Path to V4 mapping CSV")
    ap.add_argument("--v5", default="bess_mapping/BESS_UNIFIED_MAPPING_V5.csv", help="Output V5 mapping CSV")
    ap.add_argument(
        "--rollup",
        default="/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/rollup_files",
        help="Root directory of rollup parquet files",
    )
    ap.add_argument("--start-year", type=int, default=None, help="Optional min year to include in search (filters availability)")
    ap.add_argument("--end-year", type=int, default=None, help="Optional max year to include in search (filters availability)")
    ap.add_argument("--overwrite", action="store_true", help="Recompute COD_Date for all rows (overwrite existing values)")
    args = ap.parse_args()

    v4p = Path(args.v4)
    v5p = Path(args.v5)
    rollup = Path(args.rollup)

    if not v4p.exists():
        raise SystemExit(f"V4 mapping not found: {v4p}")

    # Load or create V5 from V4
    if v5p.exists():
        df = pd.read_csv(v5p)
    else:
        df = pd.read_csv(v4p)
        df.to_csv(v5p, index=False)

    # Ensure COD_Date column exists with a string dtype (to avoid dtype warnings)
    if "COD_Date" not in df.columns:
        df["COD_Date"] = pd.Series([pd.NA] * len(df), dtype="string")
    else:
        # Normalize dtype to string to allow assigning date strings
        try:
            df["COD_Date"] = df["COD_Date"].astype("string")
        except Exception:
            df["COD_Date"] = df["COD_Date"].astype(object)

    # Determine available years for gen/load datasets
    gen_dir = rollup / "SCED_Gen_Resources"
    load_dir = rollup / "SCED_Load_Resources"

    gen_years = detect_years(gen_dir)
    load_years = detect_years(load_dir)

    if args.start_year is not None:
        gen_years = [y for y in gen_years if y >= args.start_year]
        load_years = [y for y in load_years if y >= args.start_year]
    if args.end_year is not None:
        gen_years = [y for y in gen_years if y <= args.end_year]
        load_years = [y for y in load_years if y <= args.end_year]

    if not gen_years and not load_years:
        raise SystemExit(f"No SCED parquet years found under {rollup}")

    gen_dirs = [(gen_dir, gen_years)] if gen_years else []
    load_dirs = [(load_dir, load_years)] if load_years else []

    # Iterate rows with missing COD_Date and fill
    if args.overwrite:
        missing_mask = pd.Series([True] * len(df), index=df.index)
    else:
        missing_mask = df["COD_Date"].isna() | (df["COD_Date"].astype(str).str.strip() == "")
    if missing_mask.sum() == 0:
        print("No missing COD_Date values; nothing to do.")
        return

    print(f"Filling COD_Date for {int(missing_mask.sum())} BESS entries...")
    for idx in df[missing_mask].index:
        row = df.loc[idx]
        cod = resolve_cod_for_row(row, gen_dirs, load_dirs)
        if cod:
            df.at[idx, "COD_Date"] = cod
            print(f"  ✓ {row.get('BESS_Gen_Resource') or row.get('BESS_Load_Resource')}: {cod}")
        else:
            print(f"  - {row.get('BESS_Gen_Resource') or row.get('BESS_Load_Resource')}: no activity found")

    df.to_csv(v5p, index=False)
    print(f"Saved updated mapping with COD_Date → {v5p}")


if __name__ == "__main__":
    main()

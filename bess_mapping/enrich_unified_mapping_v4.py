#!/usr/bin/env python3
"""
Enrich BESS_UNIFIED_MAPPING_V3_CLARIFIED.csv with Latitude, Longitude, and County.

Strategy:
- Primary join: map final `Latitude`/`Longitude` from
  `BESS_COMPREHENSIVE_WITH_COORDINATES_V2.csv` by `BESS_Gen_Resource`.
- County: prefer `IQ_County` if present, else fallback to `EIA_County`.
- Fallback for coords: if still missing and `EIA_Generator_ID` exists,
  try to fetch coordinates/county from the EIA generators Excel provided by user.

Outputs `BESS_UNIFIED_MAPPING_V4.csv` alongside the inputs.
Also attempts to add a `QSE` column using local ERCOT disclosure files if present.
"""

from __future__ import annotations

import sys
from pathlib import Path
import glob

import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

V3_PATH = HERE / "BESS_UNIFIED_MAPPING_V3_CLARIFIED.csv"
COMP_COORDS_PATH = HERE / "BESS_COMPREHENSIVE_WITH_COORDINATES_V2.csv"
# External EIA Excel path provided in the prompt
EIA_EXCEL_PATH = Path("/home/enrico/projects/battalion-platform/data/EIA/generators/EIA_generators_latest.xlsx")

OUTPUT_PATH = HERE / "BESS_UNIFIED_MAPPING_V4.csv"
ERCOT_DATA_DIR = Path("/home/enrico/data/ERCOT_data")


def load_v3() -> pd.DataFrame:
    df = pd.read_csv(V3_PATH)
    # Normalize key columns
    if "BESS_Gen_Resource" not in df.columns:
        raise SystemExit("BESS_Gen_Resource column missing in V3 mapping")
    return df


def load_comprehensive_coords() -> pd.DataFrame:
    comp = pd.read_csv(COMP_COORDS_PATH)
    cols = [
        "BESS_Gen_Resource",
        "Latitude",
        "Longitude",
        # Some rows may also have EIA_Latitude/EIA_Longitude
        # but the final columns above are already curated.
    ]
    missing = [c for c in cols if c not in comp.columns]
    if missing:
        raise SystemExit(f"Comprehensive file missing columns: {missing}")
    return comp[cols]


def load_eia_excel_if_available() -> pd.DataFrame | None:
    if not EIA_EXCEL_PATH.exists():
        return None
    # Attempt to find reasonable columns for generator-level lat/long/county
    # We read all sheets and concat; some EIA exports have separate tabs.
    xls = pd.ExcelFile(EIA_EXCEL_PATH)
    dfs = []
    for sheet in xls.sheet_names:
        try:
            s = xls.parse(sheet)
            # Harmonize likely column names
            s_cols = {c.lower().strip(): c for c in s.columns}

            # Possible variants seen across EIA releases
            gen_id_col = (
                s_cols.get("generator id")
                or s_cols.get("generator_id")
                or s_cols.get("generatorid")
                or s_cols.get("eia_generator_id")
                or s_cols.get("generator id (unit code)")
            )
            lat_col = s_cols.get("latitude")
            lon_col = s_cols.get("longitude")
            county_col = (
                s_cols.get("county")
                or s_cols.get("county name")
                or s_cols.get("county_name")
            )

            if not gen_id_col or (not lat_col and not lon_col and not county_col):
                continue

            use_cols = {"EIA_Generator_ID": gen_id_col}
            if lat_col:
                use_cols["Latitude_eia"] = lat_col
            if lon_col:
                use_cols["Longitude_eia"] = lon_col
            if county_col:
                use_cols["EIA_County_eia"] = county_col

            sub = s[list(use_cols.values())].rename(columns={v: k for k, v in use_cols.items()})
            # Normalize generator id to string for safe joins
            sub["EIA_Generator_ID"] = sub["EIA_Generator_ID"].astype(str).str.strip()
            dfs.append(sub)
        except Exception:
            # Tolerate odd sheets
            continue

    if not dfs:
        return None
    merged = pd.concat(dfs, ignore_index=True)
    # Drop duplicates, keep first non-null coordinate occurrence
    merged = (
        merged.sort_values(by=["EIA_Generator_ID"])
        .drop_duplicates(subset=["EIA_Generator_ID"], keep="first")
        .reset_index(drop=True)
    )
    return merged


def build_qse_mapping() -> dict[str, str]:
    """Build a best-effort mapping of Resource Name -> QSE using local ERCOT files.

    Looks for CSVs under `/home/enrico/data/ERCOT_data` in these patterns:
      - 60d_DAM_Gen_Resource_Data-*.csv (columns: Resource Name, QSE or QSE Name)
      - 60d_SCED_Gen_Resource_Data-*.csv
      - 60d_DAM_Load_Resource_Data-*.csv (columns: Load Resource Name, QSE or QSE Name)
      - 60d_Load_Resource_Data_in_SCED-*.csv

    Returns an empty dict if no files are present.
    """
    base = ERCOT_DATA_DIR
    if not base.exists():
        return {}

    patterns = [
        str(base / "60-Day_DAM_Disclosure_Reports" / "csv" / "60d_DAM_Gen_Resource_Data-*.csv"),
        str(base / "60-Day_SCED_Disclosure_Reports" / "csv" / "60d_SCED_Gen_Resource_Data-*.csv"),
        str(base / "60-Day_DAM_Disclosure_Reports" / "csv" / "60d_DAM_Load_Resource_Data-*.csv"),
        str(base / "60-Day_SCED_Disclosure_Reports" / "csv" / "60d_Load_Resource_Data_in_SCED-*.csv"),
    ]

    files: list[str] = []
    for pat in patterns:
        files.extend(glob.glob(pat))

    if not files:
        return {}

    qse_by_res: dict[str, str] = {}

    def norm_col_map(cols: list[str]) -> tuple[str | None, str | None]:
        lc = {c.lower(): c for c in cols}
        # Identify resource name column variant
        res_col = lc.get("resource name") or lc.get("resource_name") or lc.get("load resource name")
        # Identify QSE column variant
        qse_col = lc.get("qse") or lc.get("qse name") or lc.get("qse_name")
        return res_col, qse_col

    # Process a subset of recent files to keep runtime quick
    files = sorted(files)[-12:]  # last dozen files

    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        res_col, qse_col = norm_col_map(list(df.columns))
        if not res_col or not qse_col:
            continue
        sub = df[[res_col, qse_col]].dropna()
        sub[res_col] = sub[res_col].astype(str)
        sub[qse_col] = sub[qse_col].astype(str)
        for r, q in zip(sub[res_col], sub[qse_col]):
            # Keep first seen QSE for a resource
            qse_by_res.setdefault(r, q)

    return qse_by_res


def build_qse_mapping() -> dict[str, str]:
    """Build a best-effort mapping of Resource Name -> QSE using local ERCOT files.

    Scans both local and shared ERCOT data directories if present:
      - /home/enrico/data/ERCOT_data
      - /pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data

    Looks for CSVs in patterns:
      - 60d_DAM_Gen_Resource_Data-*.csv (columns: Resource Name, QSE or QSE Name)
      - 60d_SCED_Gen_Resource_Data-*.csv
      - 60d_DAM_Load_Resource_Data-*.csv (columns: Load Resource Name, QSE or QSE Name)
      - 60d_Load_Resource_Data_in_SCED-*.csv

    Returns empty dict if no files are present.
    """
    bases = [
        ERCOT_DATA_DIR,
        Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data"),
    ]

    files: list[str] = []
    for base in bases:
        if not base.exists():
            continue
        patterns = [
            str(base / "60-Day_DAM_Disclosure_Reports" / "csv" / "60d_DAM_Gen_Resource_Data-*.csv"),
            str(base / "60-Day_SCED_Disclosure_Reports" / "csv" / "60d_SCED_Gen_Resource_Data-*.csv"),
            str(base / "60-Day_DAM_Disclosure_Reports" / "csv" / "60d_DAM_Load_Resource_Data-*.csv"),
            str(base / "60-Day_SCED_Disclosure_Reports" / "csv" / "60d_Load_Resource_Data_in_SCED-*.csv"),
        ]
        for pat in patterns:
            matched = sorted(glob.glob(pat))
            if not matched:
                continue
            # Use only the most recent file to keep runtime fast
            files.append(matched[-1])

    if not files:
        return {}

    qse_by_res: dict[str, str] = {}

    def norm_col_map(cols: list[str]) -> tuple[str | None, str | None]:
        lc = {c.lower().strip('"'): c for c in cols}
        # Identify resource name column variant
        res_col = (
            lc.get("resource name")
            or lc.get("resource_name")
            or lc.get("load resource name")
        )
        # Identify QSE column variant
        qse_col = lc.get("qse") or lc.get("qse name") or lc.get("qse_name")
        return res_col, qse_col

    # Keep unique list
    files = sorted(set(files))

    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        res_col, qse_col = norm_col_map(list(df.columns))
        if not res_col or not qse_col or res_col not in df.columns or qse_col not in df.columns:
            continue
        sub = df[[res_col, qse_col]].dropna()
        sub[res_col] = sub[res_col].astype(str)
        sub[qse_col] = sub[qse_col].astype(str)
        for r, q in zip(sub[res_col], sub[qse_col]):
            qse_by_res.setdefault(r, q)

    return qse_by_res


def main() -> None:
    v3 = load_v3()
    comp_coords = load_comprehensive_coords()

    # Merge coordinates by BESS_Gen_Resource
    v4 = v3.merge(
        comp_coords,
        how="left",
        on="BESS_Gen_Resource",
        suffixes=("", "_comp"),
    )

    # County resolution: prefer IQ_County else EIA_County
    county = v4.get("IQ_County")
    if county is None:
        county = pd.Series([None] * len(v4))
    eia_county = v4.get("EIA_County")
    v4["County"] = county.fillna(eia_county)

    # Fallback coordinates from EIA Excel if missing
    need_coords = v4["Latitude"].isna() | v4["Longitude"].isna()
    if need_coords.any():
        eia_df = load_eia_excel_if_available()
        if eia_df is not None and "EIA_Generator_ID" in v4.columns:
            # Normalize ID for join
            v4["EIA_Generator_ID"] = v4["EIA_Generator_ID"].astype(str).str.strip()
            v4 = v4.merge(eia_df, how="left", on="EIA_Generator_ID")
            # Fill missing lat/lon from EIA
            if "Latitude_eia" in v4.columns:
                v4["Latitude"] = v4["Latitude"].fillna(v4["Latitude_eia"])
            if "Longitude_eia" in v4.columns:
                v4["Longitude"] = v4["Longitude"].fillna(v4["Longitude_eia"])
            # Optionally fill county when still missing
            if "EIA_County_eia" in v4.columns:
                v4["County"] = v4["County"].fillna(v4["EIA_County_eia"])

            # Drop helper columns
            drop_helpers = [c for c in ("Latitude_eia", "Longitude_eia", "EIA_County_eia") if c in v4.columns]
            if drop_helpers:
                v4.drop(columns=drop_helpers, inplace=True)

    # Reorder to keep original columns first, then the new ones
    orig_cols = list(v3.columns)
    new_cols = [c for c in ["Latitude", "Longitude", "County"] if c not in orig_cols]
    v4 = v4[orig_cols + new_cols]

    # Attempt to enrich with QSE if ERCOT disclosure data is available
    qse_map = build_qse_mapping()
    if qse_map:
        # Derive QSE preferring Gen Resource mapping, then Load Resource
        qses: list[str | None] = []
        gen_col = "BESS_Gen_Resource"
        load_col = "BESS_Load_Resource"
        for _, row in v4.iterrows():
            qse_val = None
            gen_res = row.get(gen_col)
            load_res = row.get(load_col)
            if isinstance(gen_res, str) and gen_res in qse_map:
                qse_val = qse_map.get(gen_res)
            if not qse_val and isinstance(load_res, str) and load_res in qse_map:
                qse_val = qse_map.get(load_res)
            qses.append(qse_val)
        v4["QSE"] = qses
        # Move QSE to the end for consistency
        if "QSE" not in new_cols:
            v4 = v4[orig_cols + new_cols + ["QSE"]]

    # Save
    v4.to_csv(OUTPUT_PATH, index=False)

    # Basic summary to stdout
    total = len(v4)
    with_coords = v4["Latitude"].notna() & v4["Longitude"].notna()
    with_county = v4["County"].notna()
    print(
        "Enrichment complete:\n"
        f"  Rows: {total}\n"
        f"  With coordinates: {with_coords.sum()} ({with_coords.mean():.1%})\n"
        f"  With county: {with_county.sum()} ({with_county.mean():.1%})\n"
        f"  Output: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    pd.options.display.width = 200
    try:
        main()
    except FileNotFoundError as e:
        sys.stderr.write(f"Missing input file: {e}\n")
        sys.exit(2)
    except SystemExit as e:
        # Pass through our explicit exits
        raise
    except Exception as e:
        sys.stderr.write(f"Unexpected error: {e}\n")
        sys.exit(1)

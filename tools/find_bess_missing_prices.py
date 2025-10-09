#!/usr/bin/env python3
"""
Find BESS resources whose resource nodes lack DA and/or RT nodal prices and
propose connected node candidates (same substation) that do have prices.

Inputs (defaults assume your pool layout):
  --base-dir  /pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data
  --year      2024
  --mapping   bess_mapping/BESS_UNIFIED_MAPPING_V3_CLARIFIED.csv

Output:
  tools/output/bess_missing_prices_<year>.csv with columns:
    bess_name, resource_node, substation, has_da, has_rt,
    da_candidate_node, rt_candidate_node
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def load_price_nodes_da(path: Path) -> set[str]:
    # Try long format first
    p_long = path / "rollup_files" / "DA_prices" / f"{args.year}.parquet"
    if p_long.exists():
        df = pd.read_parquet(p_long, columns=None)
        cols = set(df.columns.str.lower())
        if "settlementpoint" in cols:
            col = "SettlementPoint"
        elif "settlement_point" in cols:
            col = "settlement_point"
        elif "settlementpointname" in cols:
            col = "SettlementPointName"
        else:
            col = None
        if col:
            return set(df[col].astype(str).unique())

    # Fallback: flattened wide file
    p_flat = path / "rollup_files" / "flattened" / f"DA_prices_{args.year}.parquet"
    if p_flat.exists():
        df = pd.read_parquet(p_flat)
        return set(c for c in df.columns if c not in ("datetime", "datetime_ts"))

    return set()


def load_price_nodes_rt(path: Path) -> set[str]:
    p = path / "rollup_files" / "RT_prices" / f"{args.year}.parquet"
    if p.exists():
        df = pd.read_parquet(p, columns=["SettlementPointName"])  # long format
        return set(df["SettlementPointName"].astype(str).unique())
    # Fallback: flattened wide
    p_flat = path / "rollup_files" / "flattened" / f"RT_prices_flat_{args.year}.parquet"
    if p_flat.exists():
        df = pd.read_parquet(p_flat)
        return set(c for c in df.columns if c not in ("datetime", "datetime_ts"))
    return set()


def _to_dt_utc(s: pd.Series) -> pd.Series:
    import pandas as pd
    if pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s):
        return pd.to_datetime(s, unit="ms", utc=True, errors="coerce")
    return pd.to_datetime(s, utc=True, errors="coerce")


def load_rt_price_coverage(path: Path, year: int) -> tuple[pd.DataFrame, int]:
    """Return RT coverage per node and total timestamp count in the year."""
    p = path / "rollup_files" / "RT_prices" / f"{year}.parquet"
    if not p.exists():
        return pd.DataFrame(), 0
    df = pd.read_parquet(p, columns=["SettlementPointName", "datetime"])
    df["datetime"] = _to_dt_utc(df["datetime"]) 
    total_ts = df["datetime"].nunique()
    per = df.groupby("SettlementPointName")["datetime"].nunique().reset_index(name="rt_intervals")
    per["rt_coverage_pct"] = per["rt_intervals"] / total_ts * 100.0
    # also capture first/last timestamp observed per node
    agg = df.groupby("SettlementPointName")["datetime"].agg(["min", "max"]).reset_index()
    cov = per.merge(agg, on="SettlementPointName", how="left")
    return cov, int(total_ts)


def load_da_price_coverage(path: Path, year: int) -> tuple[pd.DataFrame, int]:
    p = path / "rollup_files" / "DA_prices" / f"{year}.parquet"
    if not p.exists():
        return pd.DataFrame(), 0
    df = pd.read_parquet(p)
    # Column normalization
    if "datetime" not in df.columns:
        # Some DA long files use DeliveryDate + hour; fall back to counting by (date,hour)
        if {"DeliveryDate", "hour"}.issubset(df.columns):
            total_ts = df.drop_duplicates(["DeliveryDate", "hour"]).shape[0]
            sp_col = "SettlementPoint" if "SettlementPoint" in df.columns else (
                "settlement_point" if "settlement_point" in df.columns else "SettlementPointName")
            grp = df.groupby(sp_col)[["DeliveryDate", "hour"]].apply(lambda x: len(x.drop_duplicates())).reset_index(name="da_hours")
            grp["da_coverage_pct"] = grp["da_hours"] / total_ts * 100.0
            grp = grp.rename(columns={sp_col: "SettlementPointName"})
            return grp, int(total_ts)
        else:
            return pd.DataFrame(), 0
    df["datetime"] = _to_dt_utc(df["datetime"]) 
    total_ts = df["datetime"].nunique()
    # wide format means many columns; convert to long for counting
    sp_cols = [c for c in df.columns if c not in ("datetime", "datetime_ts")]
    long = df.melt(id_vars=["datetime"], value_vars=sp_cols, var_name="SettlementPointName", value_name="price")
    # consider present even if price is NaN? safer to require non-null
    long = long[long["price"].notna()]
    per = long.groupby("SettlementPointName")["datetime"].nunique().reset_index(name="da_hours")
    per["da_coverage_pct"] = per["da_hours"] / total_ts * 100.0
    return per, int(total_ts)


def latest_settlement_points_csv(base: Path) -> Path | None:
    cand = sorted((base / "Settlement_Points_List_and_Electrical_Buses_Mapping" / "csv").glob("Settlement_Points_*.csv"))
    return cand[-1] if cand else None


def propose_candidate(*args, **kwargs):
    # Candidate node search is disabled by design. Always return None.
    return None


def main(args):
    base = Path(args.base_dir)
    mapping = Path(args.mapping)

    # Load BESS mapping (Gen resource + settlement point)
    df_map = pd.read_csv(mapping)
    # Tolerate slight column name variance
    rn_col = None
    for c in ("Settlement_Point", "SettlementPoint", "Resource_Node", "RESOURCE_NODE"):
        if c in df_map.columns:
            rn_col = c
            break
    if rn_col is None:
        raise ValueError("Cannot find settlement point column in mapping file")
    name_col = "BESS_Gen_Resource" if "BESS_Gen_Resource" in df_map.columns else "Gen_Resource"

    df_map = df_map[[name_col, rn_col]].rename(columns={name_col: "bess_name", rn_col: "resource_node"})
    df_map["resource_node"] = df_map["resource_node"].astype(str)

    # Load availability + coverage
    da_nodes = load_price_nodes_da(base)
    rt_nodes = load_price_nodes_rt(base)
    rt_cov, rt_total = load_rt_price_coverage(base, args.year)
    da_cov, da_total = load_da_price_coverage(base, args.year)

    # Load settlement points table for substation mapping
    sp_csv = latest_settlement_points_csv(base)
    if not sp_csv:
        raise FileNotFoundError("Settlement_Points_*.csv not found under mapping directory")
    df_sp = pd.read_csv(sp_csv)

    rows = []
    for _, r in df_map.iterrows():
        rn = r["resource_node"]
        has_da = rn in da_nodes
        has_rt = rn in rt_nodes
        # coverage lookups
        da_row = da_cov[da_cov["SettlementPointName"] == rn]
        rt_row = rt_cov[rt_cov["SettlementPointName"] == rn]
        da_pct = float(da_row["da_coverage_pct"].iloc[0]) if not da_row.empty else 0.0
        rt_pct = float(rt_row["rt_coverage_pct"].iloc[0]) if not rt_row.empty else 0.0
        rt_first = pd.to_datetime(rt_row["min"].iloc[0]) if not rt_row.empty and "min" in rt_row.columns else pd.NaT
        rt_last = pd.to_datetime(rt_row["max"].iloc[0]) if not rt_row.empty and "max" in rt_row.columns else pd.NaT
        da_cand = None
        rt_cand = None
        if not has_da:
            da_cand = propose_candidate(rn, df_sp, da_nodes)
        if not has_rt:
            rt_cand = propose_candidate(rn, df_sp, rt_nodes)
        if (not has_da) or (not has_rt) or (rt_pct < args.rt_threshold) or (da_pct < args.da_threshold):
            # look up substation for reference
            sub = None
            hit = df_sp[(df_sp["RESOURCE_NODE"] == rn) | (df_sp["NODE_NAME"] == rn)]
            if not hit.empty:
                sub = str(hit["SUBSTATION"].iloc[0])
            rows.append({
                "bess_name": r["bess_name"],
                "resource_node": rn,
                "substation": sub,
                "has_da": has_da,
                "has_rt": has_rt,
                "da_coverage_pct": round(da_pct, 2),
                "rt_coverage_pct": round(rt_pct, 2),
                "rt_first_ts": rt_first,
                "rt_last_ts": rt_last,
                "da_candidate_node": da_cand,
                "rt_candidate_node": rt_cand,
            })

    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"bess_missing_prices_{args.year}.csv"
    pd.DataFrame(rows).to_csv(out_file, index=False)
    print(f"Saved: {out_file}  (rows={len(rows)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--mapping", default="bess_mapping/BESS_UNIFIED_MAPPING_V3_CLARIFIED.csv")
    parser.add_argument("--rt-threshold", type=float, default=95.0, help="Flag nodes with RT coverage below this percent")
    parser.add_argument("--da-threshold", type=float, default=95.0, help="Flag nodes with DA coverage below this percent")
    args = parser.parse_args()
    main(args)

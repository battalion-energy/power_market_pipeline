#!/usr/bin/env python3
"""
ERCOT Time Alignment QC

Checks that spike times on a target date align across:
- /api/prices (as served by ercot_price_service) for Ancillary Services
- BESS awards parquet hourly MCPC (local_date/local_hour in CT)
- DAM Clearing Prices for Capacity (DAMCPC) CSVs

Usage:
  python scripts/qc/ercot_time_alignment_check.py --date 2024-01-16 \
         --bess GAMBIT_BESS1 --hub HB_NORTH \
         [--service http://localhost:8090] \
         [--base /pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import pyarrow.parquet as pq
import requests


def to_ct(ts_utc: str) -> dt.datetime:
    # ts_utc is ISO8601 from API (UTC). Return aware CT time.
    d = dt.datetime.fromisoformat(ts_utc.replace("Z", "+00:00"))
    try:
        import zoneinfo
        tz = zoneinfo.ZoneInfo("America/Chicago")
        return d.astimezone(tz)
    except Exception:
        # Fallback without zoneinfo: assume no DST shift needed (rough)
        return d - dt.timedelta(hours=6)


def hours_from_api(ercot_service: str, date: dt.date) -> Dict[str, List[int]]:
    start = dt.datetime.combine(date, dt.time.min).isoformat() + "Z"
    end = (dt.datetime.combine(date, dt.time.min) + dt.timedelta(days=1)).isoformat() + "Z"
    hubs = "ECRS,REGUP,REGDN,NSPIN,RRS"
    url = f"{ercot_service}/api/prices?" + (
        f"start_date={start}&end_date={end}&hubs={hubs}&price_type=ancillary_services"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    ts = [to_ct(t) for t in data.get("timestamps", [])]
    out: Dict[str, List[int]] = {"NSPIN": [], "ECRS": [], "RRS": [], "REGUP": [], "REGDN": []}
    # Build map hub->prices
    series = {e["hub"]: e["prices"] for e in data.get("data", [])}
    for hub, arr in series.items():
        if hub not in out:
            continue
        for t, v in zip(ts, arr):
            if v is None:
                continue
            if t.date() == date and v > 1000:
                out[hub].append(t.hour)
    return out


def hours_from_awards(base: Path, bess: str, date: dt.date) -> Dict[str, List[int]]:
    fp = base / "bess_analysis" / "hourly" / "awards" / f"{bess}_{date.year}_awards.parquet"
    tbl = pq.read_table(fp)
    df = tbl.to_pandas()
    df["local_date"] = pd.to_datetime(df["local_date"]).dt.date
    day = df[df["local_date"] == date]
    out = {}
    for k, col in {
        "NSPIN": "nonspin_mcpc",
        "ECRS": "ecrs_mcpc",
        "RRS": "rrs_mcpc",
        "REGUP": "regup_mcpc",
        "REGDN": "regdown_mcpc",
    }.items():
        if col in day.columns:
            hrs = day.loc[day[col] > 1000, "local_hour"].astype(int).tolist()
            out[k] = hrs
    return out


def hours_from_damcpc_csv(base: Path, date: dt.date) -> Dict[str, List[int]]:
    # Scan DAMCPC CSV directory for lines with the target date
    csv_dir = base / "DAM_Clearing_Prices_for_Capacity" / "csv"
    out = {"NSPIN": [], "ECRS": [], "RRS": [], "REGUP": [], "REGDN": []}
    if not csv_dir.exists():
        return out
    target = date.strftime("%m/%d/%Y")
    for fp in csv_dir.glob("*.csv"):
        try:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if not line.startswith(target):
                        continue
                    # Example: 01/16/2024,07:00,NSPIN,550.67,N
                    parts = [p.strip() for p in line.strip().split(",")]
                    if len(parts) < 4:
                        continue
                    _, hhmm, svc, val = parts[:4]
                    try:
                        hour = int(hhmm.split(":")[0])
                        price = float(val)
                    except Exception:
                        continue
                    key = {
                        "NSPIN": "NSPIN",
                        "ECRS": "ECRS",
                        "RRS": "RRS",
                        "REGUP": "REGUP",
                        "REGDN": "REGDN",
                    }.get(svc)
                    if key and price > 1000:
                        out[key].append(hour)
        except Exception:
            continue
    # Deduplicate and sort
    for k in out:
        out[k] = sorted(set(out[k]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--bess", default="GAMBIT_BESS1")
    ap.add_argument("--hub", default="HB_NORTH")
    ap.add_argument("--service", default="http://localhost:8090")
    ap.add_argument(
        "--base",
        default="/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data",
        help="Base path to ERCOT_data",
    )
    args = ap.parse_args()

    date = dt.date.fromisoformat(args.date)
    base = Path(args.base)

    api = hours_from_api(args.service, date)
    awards = hours_from_awards(base, args.bess, date)
    damcpc = hours_from_damcpc_csv(base, date)

    def ok(x, y):
        return sorted(set(x)) == sorted(set(y))

    report = {
        "date": args.date,
        "bess": args.bess,
        "hub": args.hub,
        "api_hours_gt_1000": api,
        "awards_hours_gt_1000": awards,
        "damcpc_hours_gt_1000": damcpc,
        "match_api_vs_awards": {k: ok(api.get(k, []), awards.get(k, [])) for k in awards.keys()},
        "match_awards_vs_damcpc": {k: ok(awards.get(k, []), damcpc.get(k, [])) for k in awards.keys()},
    }

    print(json.dumps(report, indent=2, sort_keys=True))
    # Simple exit code when any mismatch occurs
    mismatches = [
        v is False for v in list(report["match_api_vs_awards"].values()) + list(report["match_awards_vs_damcpc"].values())
    ]
    if any(mismatches):
        raise SystemExit(2)


if __name__ == "__main__":
    main()


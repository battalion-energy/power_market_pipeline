#!/usr/bin/env python3
"""
ERCOT Data Status Checker

Provides comprehensive status report on all ERCOT datasets:
- File counts and date ranges
- Gap detection (note: ERCOT uses date-range files, not daily files)
- Data quality indicators
- Recommended next actions

Usage:
    python check_ercot_data_status.py
    python check_ercot_data_status.py --verbose    # Show detailed info
    python check_ercot_data_status.py --json       # Output as JSON
"""

import os
import sys
import argparse
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# Data directory
ERCOT_DATA_DIR = Path(os.getenv('ERCOT_DATA_DIR', '/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data'))

# Dataset configurations
DATASETS = {
    "dam_settlement_prices": {
        "dir": ERCOT_DATA_DIR / "DAM_Settlement_Point_Prices",
        "pattern": "*.csv",
        "description": "Day-Ahead Market Settlement Point Prices",
        "min_file_size_mb": 1,
    },
    "rtm_settlement_prices": {
        "dir": ERCOT_DATA_DIR / "RTM_Settlement_Point_Prices",
        "pattern": "*.csv",
        "description": "Real-Time Market Settlement Point Prices",
        "min_file_size_mb": 1,
    },
    "as_prices": {
        "dir": ERCOT_DATA_DIR / "AS_Prices",
        "pattern": "*.csv",
        "description": "Ancillary Services Prices",
        "min_file_size_kb": 50,
    },
    "fuel_mix": {
        "dir": ERCOT_DATA_DIR / "Fuel_Mix",
        "pattern": "*.csv",
        "description": "Fuel Mix",
        "min_file_size_kb": 10,
    },
    "system_load_forecast_zone": {
        "dir": ERCOT_DATA_DIR / "Actual_System_Load_By_Forecast_Zone",
        "pattern": "*.csv",
        "description": "Actual System Load by Forecast Zone",
        "min_file_size_kb": 10,
    },
    "system_load_weather_zone": {
        "dir": ERCOT_DATA_DIR / "Actual_System_Load_By_Weather_Zone",
        "pattern": "*.csv",
        "description": "Actual System Load by Weather Zone",
        "min_file_size_kb": 10,
    },
    "wind_power": {
        "dir": ERCOT_DATA_DIR / "Wind_Power_Production",
        "pattern": "*.csv",
        "description": "Wind Power Production",
        "min_file_size_kb": 10,
    },
    "solar_power": {
        "dir": ERCOT_DATA_DIR / "Solar_Power_Production",
        "pattern": "*.csv",
        "description": "Solar Power Production",
        "min_file_size_kb": 10,
    },
}


def extract_date_range_from_filename(filename: str) -> Optional[Tuple[datetime, datetime]]:
    """
    Extract date range from ERCOT filename format.
    Examples: dam_prices_20240101_20240131.csv, rtm_prices_20251010_20251011.csv
    """
    # Look for two dates in YYYYMMDD format
    date_matches = re.findall(r'(\d{8})', filename)

    if len(date_matches) >= 2:
        try:
            start_date = datetime.strptime(date_matches[0], '%Y%m%d')
            end_date = datetime.strptime(date_matches[1], '%Y%m%d')
            return (start_date, end_date)
        except ValueError:
            pass

    # Try single date
    if len(date_matches) == 1:
        try:
            date = datetime.strptime(date_matches[0], '%Y%m%d')
            return (date, date)
        except ValueError:
            pass

    return None


def get_file_info(data_dir: Path, pattern: str) -> List[Dict]:
    """Get list of file information including date ranges."""
    files = []

    if not data_dir.exists():
        return files

    for csv_file in data_dir.glob(pattern):
        date_range = extract_date_range_from_filename(csv_file.name)
        if date_range:
            size_mb = csv_file.stat().st_size / (1024 * 1024)
            start_date, end_date = date_range
            days_covered = (end_date - start_date).days + 1
            files.append({
                "path": csv_file,
                "start_date": start_date,
                "end_date": end_date,
                "days_covered": days_covered,
                "size_mb": size_mb
            })

    return sorted(files, key=lambda x: x["start_date"])


def get_dataset_status(dataset_name: str, config: Dict) -> Dict:
    """Get comprehensive status for a dataset."""
    data_dir = config["dir"]
    pattern = config["pattern"]

    file_info = get_file_info(data_dir, pattern)

    if not file_info:
        return {
            "dataset": dataset_name,
            "description": config["description"],
            "status": "NO_DATA",
            "file_count": 0,
            "date_range": None,
            "total_days_covered": 0,
            "total_size_gb": 0,
        }

    # Calculate overall date range and coverage
    overall_start = min(f["start_date"] for f in file_info)
    overall_end = max(f["end_date"] for f in file_info)
    total_days_possible = (overall_end - overall_start).days + 1

    # Count unique days covered (handling overlaps)
    covered_dates = set()
    for f in file_info:
        current = f["start_date"]
        while current <= f["end_date"]:
            covered_dates.add(current)
            current += timedelta(days=1)

    total_days_covered = len(covered_dates)
    total_size_gb = sum(f["size_mb"] for f in file_info) / 1024
    completeness_pct = (total_days_covered / total_days_possible * 100) if total_days_possible > 0 else 0

    days_behind = (datetime.now() - overall_end).days

    # Check for small files
    min_size_mb = config.get("min_file_size_mb", config.get("min_file_size_kb", 0) / 1024)
    small_files = [f for f in file_info if f["size_mb"] < min_size_mb]

    if days_behind <= 2 and completeness_pct >= 95 and len(small_files) == 0:
        status = "GOOD"
    elif days_behind <= 7 and completeness_pct >= 90:
        status = "OK"
    elif days_behind > 30 or completeness_pct < 50:
        status = "CRITICAL"
    else:
        status = "NEEDS_UPDATE"

    return {
        "dataset": dataset_name,
        "description": config["description"],
        "status": status,
        "file_count": len(file_info),
        "date_range": {
            "start": overall_start.strftime('%Y-%m-%d'),
            "end": overall_end.strftime('%Y-%m-%d'),
            "total_days_possible": total_days_possible,
        },
        "total_days_covered": total_days_covered,
        "completeness_pct": round(completeness_pct, 1),
        "days_behind": days_behind,
        "small_files_count": len(small_files),
        "total_size_gb": round(total_size_gb, 2),
    }


def print_status_report(statuses: List[Dict], verbose: bool = False):
    """Print human-readable status report."""
    print("=" * 80)
    print("ERCOT DATA STATUS REPORT")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()

    status_icons = {
        "GOOD": "✓",
        "OK": "⚠",
        "NEEDS_UPDATE": "⚠",
        "CRITICAL": "✗",
        "NO_DATA": "✗"
    }

    for status in statuses:
        icon = status_icons.get(status["status"], "?")
        print(f"{icon} {status['description']}")
        print(f"  Status: {status['status']}")

        if status["status"] == "NO_DATA":
            print(f"  No data found in {DATASETS[status['dataset']]['dir']}")
            print()
            continue

        date_range = status["date_range"]
        print(f"  Files: {status['file_count']} ({status['total_size_gb']} GB)")
        print(f"  Date Range: {date_range['start']} → {date_range['end']}")
        print(f"  Coverage: {status['total_days_covered']}/{date_range['total_days_possible']} days ({status['completeness_pct']}%)")
        print(f"  Days Behind: {status['days_behind']}")

        if status['small_files_count'] > 0:
            print(f"  Small/Incomplete Files: {status['small_files_count']}")

        print()

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    status_counts = defaultdict(int)
    for status in statuses:
        status_counts[status["status"]] += 1

    total_files = sum(s["file_count"] for s in statuses)
    total_size = sum(s["total_size_gb"] for s in statuses)

    print(f"Total Files: {total_files:,}")
    print(f"Total Size: {total_size:.2f} GB")
    print()

    for status_type in ["GOOD", "OK", "NEEDS_UPDATE", "CRITICAL", "NO_DATA"]:
        if status_counts[status_type] > 0:
            icon = status_icons[status_type]
            print(f"{icon} {status_type}: {status_counts[status_type]} dataset(s)")

    print()
    print("Note: ERCOT uses date-range files, not daily files.")
    print("Coverage percentages account for overlapping date ranges.")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Check status of ERCOT datasets"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed information"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()),
        help="Check specific dataset only"
    )

    args = parser.parse_args()

    datasets_to_check = [args.dataset] if args.dataset else list(DATASETS.keys())
    statuses = []

    for dataset_name in datasets_to_check:
        config = DATASETS[dataset_name]
        status = get_dataset_status(dataset_name, config)
        statuses.append(status)

    if args.json:
        print(json.dumps(statuses, indent=2))
    else:
        print_status_report(statuses, verbose=args.verbose)


if __name__ == "__main__":
    main()

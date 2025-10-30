#!/usr/bin/env python3
"""
ENTSO-E Data Status Checker

Provides comprehensive status report on all ENTSO-E datasets:
- File counts and date ranges
- Gap detection
- Data quality indicators
- Handles both daily files and consolidated date-range files

Usage:
    python check_entsoe_data_status.py
    python check_entsoe_data_status.py --verbose    # Show detailed info
    python check_entsoe_data_status.py --json       # Output as JSON
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
ENTSOE_DATA_DIR = Path(os.getenv('ENTSO_E_DATA_DIR', '/pool/ssd8tb/data/iso/ENTSO_E'))

# Dataset configurations
DATASETS = {
    "da_prices": {
        "dir": ENTSOE_DATA_DIR / "csv_files/da_prices",
        "pattern": "*.csv",
        "description": "Day-Ahead Prices",
        "type": "consolidated",  # Expects files with date ranges
    },
    "rebap_imbalance": {
        "dir": ENTSOE_DATA_DIR / "csv_files/rebap",
        "pattern": "rebap_*.csv",
        "description": "reBAP Imbalance Prices",
        "type": "consolidated",
    },
    "fcr_capacity": {
        "dir": ENTSOE_DATA_DIR / "csv_files/de_ancillary_services",
        "pattern": "fcr_capacity_*.csv",
        "description": "FCR Primary Reserve Capacity",
        "type": "mixed",  # Can have both consolidated and daily files
    },
    "afrr_capacity": {
        "dir": ENTSOE_DATA_DIR / "csv_files/de_ancillary_services",
        "pattern": "afrr_capacity_*.csv",
        "description": "aFRR Secondary Reserve Capacity",
        "type": "mixed",
    },
    "afrr_energy": {
        "dir": ENTSOE_DATA_DIR / "csv_files/de_ancillary_services",
        "pattern": "afrr_energy_*.csv",
        "description": "aFRR Energy Activation",
        "type": "mixed",
    },
    "mfrr_capacity": {
        "dir": ENTSOE_DATA_DIR / "csv_files/de_ancillary_services",
        "pattern": "mfrr_capacity_*.csv",
        "description": "mFRR Minute Reserve Capacity",
        "type": "mixed",
    },
    "mfrr_energy": {
        "dir": ENTSOE_DATA_DIR / "csv_files/de_ancillary_services",
        "pattern": "mfrr_energy_*.csv",
        "description": "mFRR Energy Activation",
        "type": "mixed",
    },
}


def extract_date_range_from_filename(filename: str) -> Optional[Tuple[datetime, datetime]]:
    """
    Extract date range from filename.
    Supports patterns like:
    - data_2020-01-01_2025-10-27.csv (start and end date)
    - data_2020-01-01.csv (single date, start=end)
    """
    # Try date range pattern: YYYY-MM-DD_YYYY-MM-DD
    range_match = re.search(r'(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})', filename)
    if range_match:
        try:
            start_date = datetime.strptime(range_match.group(1), '%Y-%m-%d')
            end_date = datetime.strptime(range_match.group(2), '%Y-%m-%d')
            return (start_date, end_date)
        except ValueError:
            pass

    # Try single date pattern: YYYY-MM-DD
    single_match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
    if single_match:
        try:
            date = datetime.strptime(single_match.group(1), '%Y-%m-%d')
            return (date, date)
        except ValueError:
            pass

    # Try YYYYMMDD format
    yyyymmdd_match = re.search(r'(\d{8})', filename)
    if yyyymmdd_match:
        try:
            date = datetime.strptime(yyyymmdd_match.group(1), '%Y%m%d')
            return (date, date)
        except ValueError:
            pass

    return None


def get_file_date_ranges(data_dir: Path, pattern: str) -> List[Tuple[datetime, datetime, Path, float]]:
    """Get list of (start_date, end_date, filepath, size_mb) tuples from CSV files."""
    files = []

    if not data_dir.exists():
        return files

    for csv_file in data_dir.glob(pattern):
        # Skip documentation files
        if csv_file.suffix != '.csv' or 'README' in csv_file.name or 'SUMMARY' in csv_file.name:
            continue

        date_range = extract_date_range_from_filename(csv_file.name)
        if date_range:
            start_date, end_date = date_range
            size_mb = csv_file.stat().st_size / (1024 * 1024)
            files.append((start_date, end_date, csv_file, size_mb))

    return sorted(files, key=lambda x: x[0])


def find_coverage_gaps(file_ranges: List[Tuple[datetime, datetime, Path, float]]) -> Tuple[set, List[Tuple[datetime, datetime]]]:
    """
    Find gaps in date coverage from consolidated files.
    Returns: (covered_dates_set, gaps_list)
    """
    if not file_ranges:
        return (set(), [])

    # Build set of all covered dates
    covered_dates = set()
    for start_date, end_date, _, _ in file_ranges:
        current = start_date
        while current <= end_date:
            covered_dates.add(current)
            current += timedelta(days=1)

    if not covered_dates:
        return (set(), [])

    # Find gaps
    min_date = min(covered_dates)
    max_date = max(covered_dates)

    gaps = []
    gap_start = None
    current_date = min_date

    while current_date <= max_date:
        if current_date not in covered_dates:
            if gap_start is None:
                gap_start = current_date
        else:
            if gap_start is not None:
                gaps.append((gap_start, current_date - timedelta(days=1)))
                gap_start = None
        current_date += timedelta(days=1)

    if gap_start is not None:
        gaps.append((gap_start, max_date))

    return (covered_dates, gaps)


def get_dataset_status(dataset_name: str, config: Dict) -> Dict:
    """Get comprehensive status for a dataset."""
    data_dir = config["dir"]
    pattern = config["pattern"]

    file_ranges = get_file_date_ranges(data_dir, pattern)

    if not file_ranges:
        return {
            "dataset": dataset_name,
            "description": config["description"],
            "status": "NO_DATA",
            "file_count": 0,
            "date_range": None,
            "coverage_days": 0,
            "gaps": [],
            "total_size_gb": 0,
        }

    # Calculate coverage
    covered_dates, gaps = find_coverage_gaps(file_ranges)

    min_date = min(covered_dates)
    max_date = max(covered_dates)
    expected_days = (max_date - min_date).days + 1
    covered_days = len(covered_dates)
    completeness_pct = (covered_days / expected_days * 100) if expected_days > 0 else 0

    sizes = [s for _, _, _, s in file_ranges]
    total_size_gb = sum(sizes) / 1024

    days_behind = (datetime.now() - max_date).days

    # Determine status
    if days_behind <= 2 and completeness_pct >= 98:
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
        "file_count": len(file_ranges),
        "date_range": {
            "start": min_date.strftime('%Y-%m-%d'),
            "end": max_date.strftime('%Y-%m-%d'),
            "expected_days": expected_days,
            "covered_days": covered_days,
        },
        "completeness_pct": round(completeness_pct, 1),
        "days_behind": days_behind,
        "gaps": [
            {
                "start": gap_start.strftime('%Y-%m-%d'),
                "end": gap_end.strftime('%Y-%m-%d'),
                "days": (gap_end - gap_start).days + 1
            }
            for gap_start, gap_end in gaps
        ],
        "files": [
            {
                "start": start.strftime('%Y-%m-%d'),
                "end": end.strftime('%Y-%m-%d'),
                "days": (end - start).days + 1,
                "size_mb": round(size, 2),
                "name": path.name
            }
            for start, end, path, size in file_ranges
        ],
        "total_size_gb": round(total_size_gb, 2),
    }


def print_status_report(statuses: List[Dict], verbose: bool = False):
    """Print human-readable status report."""
    print("=" * 80)
    print("ENTSO-E DATA STATUS REPORT")
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
            print(f"  No data found")
            print()
            continue

        date_range = status["date_range"]
        print(f"  Files: {status['file_count']} ({status['total_size_gb']} GB)")
        print(f"  Date Range: {date_range['start']} → {date_range['end']}")
        print(f"  Coverage: {date_range['covered_days']:,}/{date_range['expected_days']:,} days ({status['completeness_pct']}%)")
        print(f"  Days Behind: {status['days_behind']}")

        if status['gaps']:
            total_gap_days = sum(g['days'] for g in status['gaps'])
            print(f"  Gaps: {len(status['gaps'])} gaps ({total_gap_days:,} days missing)")
            if verbose:
                for gap in status['gaps'][:5]:
                    print(f"    - {gap['start']} → {gap['end']} ({gap['days']} days)")
                if len(status['gaps']) > 5:
                    print(f"    ... and {len(status['gaps']) - 5} more gaps")

        if verbose and len(status['files']) <= 10:
            print(f"  Files:")
            for f in status['files']:
                if f['days'] == 1:
                    print(f"    - {f['name']} ({f['size_mb']} MB)")
                else:
                    print(f"    - {f['name']} ({f['days']} days, {f['size_mb']} MB)")

        print()

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    status_counts = defaultdict(int)
    for status in statuses:
        status_counts[status["status"]] += 1

    total_files = sum(s["file_count"] for s in statuses)
    total_size = sum(s["total_size_gb"] for s in statuses)
    total_coverage = sum(s["date_range"]["covered_days"] for s in statuses if s["status"] != "NO_DATA")

    print(f"Total Files: {total_files:,}")
    print(f"Total Size: {total_size:.2f} GB")
    print(f"Total Coverage: {total_coverage:,} dataset-days")
    print()

    for status_type in ["GOOD", "OK", "NEEDS_UPDATE", "CRITICAL", "NO_DATA"]:
        if status_counts[status_type] > 0:
            icon = status_icons[status_type]
            print(f"{icon} {status_type}: {status_counts[status_type]} dataset(s)")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Check status of ENTSO-E datasets"
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

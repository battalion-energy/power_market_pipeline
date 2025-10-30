#!/usr/bin/env python3
"""
MISO Data Status Checker

Provides comprehensive status report on all MISO datasets:
- File counts and date ranges
- Gap detection
- Data quality indicators
- Recommended next actions

Usage:
    python check_miso_data_status.py
    python check_miso_data_status.py --verbose    # Show detailed gaps
    python check_miso_data_status.py --json       # Output as JSON
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
MISO_DATA_DIR = Path(os.getenv('MISO_DATA_DIR', '/pool/ssd8tb/data/iso/MISO'))

# Dataset configurations
DATASETS = {
    "da_expost": {
        "dir": MISO_DATA_DIR / "csv_files/da_expost",
        "pattern": "*_da_expost_lmp.csv",
        "description": "Day-Ahead Ex-Post LMP",
        "min_file_size_mb": 5,
    },
    "da_exante": {
        "dir": MISO_DATA_DIR / "csv_files/da_exante",
        "pattern": "*_da_exante_lmp.csv",
        "description": "Day-Ahead Ex-Ante LMP",
        "min_file_size_mb": 5,
    },
    "rt_final": {
        "dir": MISO_DATA_DIR / "csv_files/rt_final",
        "pattern": "*_rt_lmp_final.csv",
        "description": "Real-Time Final LMP",
        "min_file_size_mb": 5,
    },
    "rt_5min": {
        "dir": MISO_DATA_DIR / "csv_files/rt_5min_lmp",
        "pattern": "*.csv",
        "description": "Real-Time 5-Min LMP",
        "min_file_size_mb": 20,
    },
    "ancillary_da": {
        "dir": MISO_DATA_DIR / "ancillary_services/da_expost_asm",
        "pattern": "*_asm_expost_damcp.csv",
        "description": "Ancillary Services DA Ex-Post",
        "min_file_size_kb": 10,
    },
    "ancillary_rt": {
        "dir": MISO_DATA_DIR / "ancillary_services/rt_final_asm",
        "pattern": "*_asm_rtmcp_final.csv",
        "description": "Ancillary Services RT Final",
        "min_file_size_kb": 10,
    },
    "load_actual": {
        "dir": MISO_DATA_DIR / "load/local_resource_zone",
        "pattern": "*_df_al.xls",
        "description": "Load - Actual (Local Resource Zone)",
        "min_file_size_kb": 5,
    },
    "load_forecast": {
        "dir": MISO_DATA_DIR / "load/regional",
        "pattern": "*_rf_al.xls",
        "description": "Load - Regional Forecast",
        "min_file_size_kb": 5,
    },
    "generation_fuel_mix": {
        "dir": MISO_DATA_DIR / "eia_fuel_mix",
        "pattern": "eia_fuel_mix_*.csv",
        "description": "Generation - EIA Fuel Mix",
        "min_file_size_kb": 5,
        "use_end_date": True,  # Files have date ranges (e.g., eia_fuel_mix_20241005_20251029.csv)
        "expected_daily": False,  # Date-range files, not daily files
    },
}


def extract_date_from_filename(filename: str) -> Optional[datetime]:
    """Extract date from filename (supports YYYY-MM-DD and YYYYMMDD patterns)."""
    # Try YYYY-MM-DD
    date_match = re.search(r'(\d{4})-(\d{2})-(\d{2})', filename)
    if date_match:
        try:
            date_str = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"
            return datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            pass

    # Try YYYYMMDD
    date_match = re.search(r'(\d{4})(\d{2})(\d{2})', filename)
    if date_match:
        try:
            date_str = f"{date_match.group(1)}{date_match.group(2)}{date_match.group(3)}"
            return datetime.strptime(date_str, '%Y%m%d')
        except ValueError:
            pass

    return None


def extract_date_range_from_filename(filename: str) -> Optional[Tuple[datetime, datetime]]:
    """
    Extract date range from filename with two dates.
    Example: eia_fuel_mix_20241005_20251029.csv
    """
    date_matches = re.findall(r'(\d{4})(\d{2})(\d{2})', filename)
    if len(date_matches) >= 2:
        try:
            start_date = datetime.strptime(f"{date_matches[0][0]}{date_matches[0][1]}{date_matches[0][2]}", '%Y%m%d')
            end_date = datetime.strptime(f"{date_matches[1][0]}{date_matches[1][1]}{date_matches[1][2]}", '%Y%m%d')
            return (start_date, end_date)
        except ValueError:
            pass

    # Fall back to single date
    single_date = extract_date_from_filename(filename)
    if single_date:
        return (single_date, single_date)

    return None


def get_file_dates(data_dir: Path, pattern: str, use_end_date: bool = False) -> List[Tuple[datetime, Path, float]]:
    """Get list of (date, filepath, size_mb) tuples from files.

    Args:
        data_dir: Directory to search
        pattern: Glob pattern for files
        use_end_date: If True, extract end date from range files (e.g., file_20240101_20241231.csv)
    """
    files = []

    if not data_dir.exists():
        return files

    for csv_file in data_dir.glob(pattern):
        if use_end_date:
            # Try to extract date range and use end date
            date_range = extract_date_range_from_filename(csv_file.name)
            if date_range:
                date = date_range[1]  # Use end date
            else:
                continue
        else:
            # Use single date extraction
            date = extract_date_from_filename(csv_file.name)
            if not date:
                continue

        size_mb = csv_file.stat().st_size / (1024 * 1024)
        files.append((date, csv_file, size_mb))

    return sorted(files, key=lambda x: x[0])


def find_gaps(file_dates: List[Tuple[datetime, Path, float]],
              start_date: Optional[datetime] = None,
              end_date: Optional[datetime] = None) -> List[Tuple[datetime, datetime]]:
    """Find date gaps in the dataset."""
    if not file_dates:
        return []

    dates = [d for d, _, _ in file_dates]

    if start_date is None:
        start_date = min(dates)
    if end_date is None:
        end_date = max(dates)

    date_set = set(dates)
    gaps = []
    gap_start = None

    current_date = start_date
    while current_date <= end_date:
        if current_date not in date_set:
            if gap_start is None:
                gap_start = current_date
        else:
            if gap_start is not None:
                gaps.append((gap_start, current_date - timedelta(days=1)))
                gap_start = None
        current_date += timedelta(days=1)

    if gap_start is not None:
        gaps.append((gap_start, end_date))

    return gaps


def find_small_files(file_dates: List[Tuple[datetime, Path, float]],
                     min_size_mb: float) -> List[Tuple[datetime, Path, float]]:
    """Find files below minimum size threshold."""
    return [(d, p, s) for d, p, s in file_dates if s < min_size_mb]


def get_dataset_status(dataset_name: str, config: Dict) -> Dict:
    """Get comprehensive status for a dataset."""
    data_dir = config["dir"]
    pattern = config["pattern"]
    use_end_date = config.get("use_end_date", False)

    file_dates = get_file_dates(data_dir, pattern, use_end_date)

    if not file_dates:
        return {
            "dataset": dataset_name,
            "description": config["description"],
            "status": "NO_DATA",
            "file_count": 0,
            "date_range": None,
            "gaps": [],
            "small_files": [],
            "total_size_gb": 0,
        }

    dates = [d for d, _, _ in file_dates]
    sizes = [s for _, _, s in file_dates]
    min_date = min(dates)
    max_date = max(dates)
    total_size_gb = sum(sizes) / 1024

    expected_daily = config.get("expected_daily", True)

    # For non-daily files (e.g., date-range files), skip gap detection and use file count
    if expected_daily:
        gaps = find_gaps(file_dates, min_date, max_date)
        expected_days = (max_date - min_date).days + 1
        actual_files = len(file_dates)
        completeness_pct = (actual_files / expected_days * 100) if expected_days > 0 else 0
    else:
        gaps = []
        expected_days = 0
        actual_files = len(file_dates)
        completeness_pct = 100.0 if actual_files > 0 else 0

    min_size_mb = config.get("min_file_size_mb", config.get("min_file_size_kb", 0) / 1024)
    small_files = find_small_files(file_dates, min_size_mb)

    days_behind = (datetime.now() - max_date).days

    if days_behind <= 2 and completeness_pct >= 95 and len(small_files) == 0:
        status = "GOOD"
    elif days_behind <= 7 and completeness_pct >= 90:
        status = "OK"
    elif days_behind > 30 or (expected_daily and completeness_pct < 50):
        status = "CRITICAL"
    else:
        status = "NEEDS_UPDATE"

    return {
        "dataset": dataset_name,
        "description": config["description"],
        "status": status,
        "file_count": actual_files,
        "date_range": {
            "start": min_date.strftime('%Y-%m-%d'),
            "end": max_date.strftime('%Y-%m-%d'),
            "days": expected_days,
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
        "small_files": [
            {
                "date": d.strftime('%Y-%m-%d'),
                "size_mb": round(s, 2),
                "path": str(p.name)
            }
            for d, p, s in small_files
        ],
        "total_size_gb": round(total_size_gb, 2),
    }


def print_status_report(statuses: List[Dict], verbose: bool = False):
    """Print human-readable status report."""
    print("=" * 80)
    print("MISO DATA STATUS REPORT")
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
        print(f"  Date Range: {date_range['start']} → {date_range['end']} ({date_range['days']} days)")
        print(f"  Completeness: {status['completeness_pct']}%")
        print(f"  Days Behind: {status['days_behind']}")

        if status['gaps']:
            total_gap_days = sum(g['days'] for g in status['gaps'])
            print(f"  Gaps: {len(status['gaps'])} gaps ({total_gap_days} days missing)")
            if verbose:
                for gap in status['gaps'][:10]:
                    print(f"    - {gap['start']} → {gap['end']} ({gap['days']} days)")
                if len(status['gaps']) > 10:
                    print(f"    ... and {len(status['gaps']) - 10} more gaps")

        if status['small_files']:
            print(f"  Small/Incomplete Files: {len(status['small_files'])}")
            if verbose:
                for sf in status['small_files'][:5]:
                    print(f"    - {sf['date']}: {sf['size_mb']} MB")
                if len(status['small_files']) > 5:
                    print(f"    ... and {len(status['small_files']) - 5} more")

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


def main():
    parser = argparse.ArgumentParser(
        description="Check status of MISO datasets"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed gap information"
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

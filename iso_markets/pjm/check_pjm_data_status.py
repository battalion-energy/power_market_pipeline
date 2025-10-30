#!/usr/bin/env python3
"""
PJM Data Status Checker

Provides comprehensive status report on all PJM datasets:
- File counts and date ranges
- Gap detection
- Data quality indicators
- Recommended next actions

Usage:
    python check_pjm_data_status.py
    python check_pjm_data_status.py --verbose    # Show detailed gaps
    python check_pjm_data_status.py --json       # Output as JSON
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
PJM_DATA_DIR = Path(os.getenv('PJM_DATA_DIR', '/pool/ssd8tb/data/iso/PJM_data'))

# Dataset configurations
DATASETS = {
    "da_nodal": {
        "dir": PJM_DATA_DIR / "csv_files/da_nodal",
        "pattern": "nodal_da_lmp_*.csv",
        "description": "Day-Ahead Nodal LMPs",
        "min_file_size_mb": 10,
        "expected_daily": True,
    },
    "rt_hourly_nodal": {
        "dir": PJM_DATA_DIR / "csv_files/rt_hourly_nodal",
        "pattern": "nodal_rt_hourly_lmp_*.csv",
        "description": "Real-Time Hourly Nodal LMPs",
        "min_file_size_mb": 25,
        "expected_daily": True,
    },
    "da_ancillary_services": {
        "dir": PJM_DATA_DIR / "csv_files/da_ancillary_services",
        "pattern": "ancillary_services_*.csv",
        "description": "Day-Ahead Ancillary Services",
        "min_file_size_kb": 1,
        "expected_daily": False,  # Quarterly files, not daily
        "date_range_files": True,  # Files contain date ranges
    },
    "rt_5min_nodal": {
        "dir": PJM_DATA_DIR / "csv_files/rt_5min_nodal",
        "pattern": "nodal_rt_5min_lmp_*.csv",
        "description": "Real-Time 5-Min Nodal LMPs",
        "min_file_size_mb": 100,
        "expected_daily": True,
        "retention_days": 186,  # PJM only keeps 6 months
    },
}


def extract_date_from_filename(filename: str) -> Optional[datetime]:
    """Extract date from filename (YYYY-MM-DD pattern)."""
    date_match = re.search(r'(\d{4})-(\d{2})-(\d{2})', filename)
    if date_match:
        try:
            date_str = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"
            return datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            return None
    return None


def extract_date_range_from_filename(filename: str) -> Optional[Tuple[datetime, datetime]]:
    """
    Extract date range from filename with two dates.
    Example: ancillary_services_2024-01-01_2024-03-31.csv
    """
    # Look for two dates in YYYY-MM-DD format
    date_matches = re.findall(r'(\d{4})-(\d{2})-(\d{2})', filename)

    if len(date_matches) >= 2:
        try:
            start_date = datetime.strptime(f"{date_matches[0][0]}-{date_matches[0][1]}-{date_matches[0][2]}", '%Y-%m-%d')
            end_date = datetime.strptime(f"{date_matches[1][0]}-{date_matches[1][1]}-{date_matches[1][2]}", '%Y-%m-%d')
            return (start_date, end_date)
        except ValueError:
            pass

    # If only one date found, treat as single-day file
    if len(date_matches) == 1:
        try:
            date = datetime.strptime(f"{date_matches[0][0]}-{date_matches[0][1]}-{date_matches[0][2]}", '%Y-%m-%d')
            return (date, date)
        except ValueError:
            pass

    return None


def get_file_dates(data_dir: Path, pattern: str) -> List[Tuple[datetime, Path, float]]:
    """Get list of (date, filepath, size_mb) tuples from CSV files."""
    files = []

    if not data_dir.exists():
        return files

    for csv_file in data_dir.glob(pattern):
        date = extract_date_from_filename(csv_file.name)
        if date:
            size_mb = csv_file.stat().st_size / (1024 * 1024)
            files.append((date, csv_file, size_mb))

    return sorted(files, key=lambda x: x[0])


def get_date_range_file_info(data_dir: Path, pattern: str) -> List[Dict]:
    """Get list of file information for date-range files (like quarterly ancillary services)."""
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

    # Close final gap if it extends to end
    if gap_start is not None:
        gaps.append((gap_start, end_date))

    return gaps


def find_small_files(file_dates: List[Tuple[datetime, Path, float]],
                     min_size_mb: float) -> List[Tuple[datetime, Path, float]]:
    """Find files below minimum size threshold."""
    return [(d, p, s) for d, p, s in file_dates if s < min_size_mb]


def find_gap_metadata_files(data_dir: Path) -> List[Path]:
    """Find .gaps.json files indicating incomplete downloads."""
    if not data_dir.exists():
        return []
    return list(data_dir.glob("*.gaps.json"))


def get_date_range_dataset_status(dataset_name: str, config: Dict) -> Dict:
    """Get status for datasets using date-range files (like quarterly ancillary services)."""
    data_dir = config["dir"]
    pattern = config["pattern"]

    file_info = get_date_range_file_info(data_dir, pattern)

    if not file_info:
        return {
            "dataset": dataset_name,
            "description": config["description"],
            "status": "NO_DATA",
            "file_count": 0,
            "date_range": None,
            "total_days_covered": 0,
            "gaps": [],
            "small_files": [],
            "gap_metadata_files": [],
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

    # Find gap metadata files
    gap_metadata_files = find_gap_metadata_files(data_dir)

    # Determine status
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
        "gaps": [],  # For date-range files, gaps are less relevant
        "small_files": [
            {
                "date_range": f"{f['start_date'].strftime('%Y-%m-%d')} → {f['end_date'].strftime('%Y-%m-%d')}",
                "size_mb": round(f["size_mb"], 2),
                "path": str(f["path"].name)
            }
            for f in small_files
        ],
        "gap_metadata_files": [str(p.name) for p in gap_metadata_files],
        "total_size_gb": round(total_size_gb, 2),
    }


def get_dataset_status(dataset_name: str, config: Dict) -> Dict:
    """Get comprehensive status for a dataset."""
    data_dir = config["dir"]
    pattern = config["pattern"]

    # Check if this dataset uses date-range files (like quarterly ancillary services)
    if config.get("date_range_files", False):
        return get_date_range_dataset_status(dataset_name, config)

    # Get all files with dates
    file_dates = get_file_dates(data_dir, pattern)

    if not file_dates:
        return {
            "dataset": dataset_name,
            "description": config["description"],
            "status": "NO_DATA",
            "file_count": 0,
            "date_range": None,
            "gaps": [],
            "small_files": [],
            "gap_metadata_files": [],
            "total_size_gb": 0,
        }

    # Basic stats
    dates = [d for d, _, _ in file_dates]
    sizes = [s for _, _, s in file_dates]
    min_date = min(dates)
    max_date = max(dates)
    total_size_gb = sum(sizes) / 1024

    # Find gaps
    gaps = find_gaps(file_dates, min_date, max_date)

    # Find small files (likely incomplete)
    min_size_mb = config.get("min_file_size_mb", config.get("min_file_size_kb", 0) / 1024)
    small_files = find_small_files(file_dates, min_size_mb)

    # Find gap metadata files
    gap_metadata_files = find_gap_metadata_files(data_dir)

    # Calculate expected vs actual files
    expected_days = (max_date - min_date).days + 1
    actual_files = len(file_dates)
    completeness_pct = (actual_files / expected_days * 100) if expected_days > 0 else 0

    # Days behind
    days_behind = (datetime.now() - max_date).days

    # Determine status
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
        "gap_metadata_count": len(gap_metadata_files),
        "total_size_gb": round(total_size_gb, 2),
    }


def print_status_report(statuses: List[Dict], verbose: bool = False):
    """Print human-readable status report."""
    print("=" * 80)
    print("PJM DATA STATUS REPORT")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()

    # Status emoji mapping
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
        # Handle both daily files (days) and date-range files (total_days_possible)
        days_in_range = date_range.get('days', date_range.get('total_days_possible', 0))
        print(f"  Date Range: {date_range['start']} → {date_range['end']} ({days_in_range} days)")

        # Show coverage for date-range files
        if 'total_days_covered' in status:
            print(f"  Coverage: {status['total_days_covered']}/{days_in_range} days ({status['completeness_pct']}%)")
        else:
            print(f"  Completeness: {status['completeness_pct']}%")
        print(f"  Days Behind: {status['days_behind']}")

        # Show gap summary
        if status['gaps']:
            total_gap_days = sum(g['days'] for g in status['gaps'])
            print(f"  Gaps: {len(status['gaps'])} gaps ({total_gap_days} days missing)")
            if verbose:
                for gap in status['gaps']:
                    print(f"    - {gap['start']} → {gap['end']} ({gap['days']} days)")

        # Show small files
        if status['small_files']:
            print(f"  Small/Incomplete Files: {len(status['small_files'])}")
            if verbose:
                for sf in status['small_files'][:5]:  # Show first 5
                    # Handle both daily files (date) and date-range files (date_range)
                    if 'date' in sf:
                        print(f"    - {sf['date']}: {sf['size_mb']} MB")
                    elif 'date_range' in sf:
                        print(f"    - {sf['date_range']}: {sf['size_mb']} MB")
                if len(status['small_files']) > 5:
                    print(f"    ... and {len(status['small_files']) - 5} more")

        # Show gap metadata files
        if status.get('gap_metadata_files') and len(status['gap_metadata_files']) > 0:
            print(f"  Partial Downloads: {len(status['gap_metadata_files'])} files with gap metadata")

        print()

    # Summary
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

    # Recommendations
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    needs_update = [s for s in statuses if s["status"] in ["NEEDS_UPDATE", "CRITICAL"]]
    has_gaps = [s for s in statuses if s.get("gaps") and len(s["gaps"]) > 0]
    has_partials = [s for s in statuses if s.get("gap_metadata_count", 0) > 0]

    if needs_update:
        print("⚠ Update needed for:")
        for s in needs_update:
            print(f"  - {s['description']}: {s['days_behind']} days behind")
        print()
        print("  Run: python iso_markets/pjm/update_pjm_with_resume.py")
        print()

    if has_gaps:
        print("⚠ Gaps detected in:")
        for s in has_gaps:
            total_gap_days = sum(g['days'] for g in s['gaps'])
            print(f"  - {s['description']}: {len(s['gaps'])} gaps ({total_gap_days} days)")
        print()
        print("  Run: python iso_markets/pjm/fill_pjm_gaps.py")
        print()

    if has_partials:
        print("⚠ Partial downloads found in:")
        for s in has_partials:
            print(f"  - {s['description']}: {s['gap_metadata_count']} files")
        print()
        print("  Run: python iso_markets/pjm/fill_pjm_gaps.py --partial-only")
        print()

    if not needs_update and not has_gaps and not has_partials:
        print("✓ All datasets are up to date!")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Check status of PJM datasets"
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

    # Check datasets
    datasets_to_check = [args.dataset] if args.dataset else list(DATASETS.keys())
    statuses = []

    for dataset_name in datasets_to_check:
        config = DATASETS[dataset_name]
        status = get_dataset_status(dataset_name, config)
        statuses.append(status)

    # Output
    if args.json:
        print(json.dumps(statuses, indent=2))
    else:
        print_status_report(statuses, verbose=args.verbose)


if __name__ == "__main__":
    main()

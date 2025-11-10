#!/usr/bin/env python3
"""
ERCOT Energy Storage Resources (ESR) Daily Data Downloader - CSV Version

Downloads and appends ESR data to a contiguous CSV file.
Creates a single time series log with all historical data.

Data includes 5-minute resolution:
- Total Discharging (MW) - positive values, ESRs injecting to grid
- Total Charging (MW) - negative values, ESRs consuming from grid
- Net Output (MW) - sum of discharging and charging

Data Source: https://www.ercot.com/api/1/services/read/dashboards/energy-storage-resources.json
Dashboard: https://www.ercot.com/gridmktinfo/dashboards/energystorageresources
"""

import csv
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Set
import urllib.request
import urllib.error

# Configuration
API_URL = "https://www.ercot.com/api/1/services/read/dashboards/energy-storage-resources.json"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"

# Determine script directory
SCRIPT_DIR = Path(__file__).parent.absolute()
CSV_FILE = SCRIPT_DIR / "esr_historical_data.csv"
BACKUP_DIR = SCRIPT_DIR / "esr_backups"

# CSV columns
CSV_COLUMNS = [
    'timestamp',
    'epoch',
    'total_charging_mw',
    'total_discharging_mw',
    'net_output_mw',
    'dst_flag',
    'downloaded_at',
    'download_timestamp'
]


def setup_directories() -> Path:
    """Create backup directory if it doesn't exist."""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    return BACKUP_DIR


def download_esr_data() -> Dict[str, Any]:
    """
    Download ESR data from ERCOT API.

    Returns:
        dict: JSON data containing lastUpdated, previousDay, and currentDay

    Raises:
        urllib.error.URLError: If download fails
    """
    req = urllib.request.Request(API_URL, headers={"User-Agent": USER_AGENT})

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode('utf-8'))
        return data
    except urllib.error.URLError as e:
        print(f"Error downloading data: {e}", file=sys.stderr)
        raise
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}", file=sys.stderr)
        raise


def get_existing_timestamps(csv_file: Path) -> Set[str]:
    """
    Get set of existing timestamps from CSV to avoid duplicates.

    Args:
        csv_file: Path to CSV file

    Returns:
        set: Set of existing timestamp strings
    """
    if not csv_file.exists():
        return set()

    existing = set()
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing.add(row['timestamp'])
    except Exception as e:
        print(f"Warning: Error reading existing timestamps: {e}", file=sys.stderr)
        return set()

    return existing


def create_backup(csv_file: Path, backup_dir: Path):
    """
    Create a backup of the CSV file before modifying.

    Args:
        csv_file: Path to CSV file
        backup_dir: Directory to save backups
    """
    if not csv_file.exists():
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = backup_dir / f"esr_historical_data_{timestamp}.csv"

    try:
        import shutil
        shutil.copy2(csv_file, backup_file)
        print(f"Created backup: {backup_file.name}")

        # Keep only last 7 days of backups
        cutoff_date = datetime.now() - timedelta(days=7)
        for old_backup in backup_dir.glob("esr_historical_data_*.csv"):
            try:
                timestamp_str = old_backup.stem.replace("esr_historical_data_", "")
                file_date = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                if file_date < cutoff_date:
                    old_backup.unlink()
                    print(f"Removed old backup: {old_backup.name}")
            except (ValueError, IndexError):
                continue
    except Exception as e:
        print(f"Warning: Could not create backup: {e}", file=sys.stderr)


def append_to_csv(data: Dict[str, Any], csv_file: Path) -> tuple[int, int]:
    """
    Append new ESR data to CSV file, avoiding duplicates.

    Args:
        data: ESR data from API
        csv_file: Path to CSV file

    Returns:
        tuple: (new_rows_added, duplicate_rows_skipped)
    """
    file_exists = csv_file.exists()
    existing_timestamps = get_existing_timestamps(csv_file)

    download_time = data.get('lastUpdated', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    download_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Collect all data points from both previousDay and currentDay
    all_data_points = []

    for day_key in ['previousDay', 'currentDay']:
        if day_key in data and 'data' in data[day_key]:
            for point in data[day_key]['data']:
                timestamp = point['timestamp']

                # Skip duplicates
                if timestamp in existing_timestamps:
                    continue

                all_data_points.append({
                    'timestamp': timestamp,
                    'epoch': point['epoch'],
                    'total_charging_mw': point['totalCharging'],
                    'total_discharging_mw': point['totalDischarging'],
                    'net_output_mw': point['netOutput'],
                    'dst_flag': point.get('dstFlag', 'N'),
                    'downloaded_at': download_time,
                    'download_timestamp': download_timestamp
                })

    # Sort by epoch time to maintain chronological order
    all_data_points.sort(key=lambda x: x['epoch'])

    # Append to CSV
    mode = 'a' if file_exists else 'w'
    with open(csv_file, mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)

        # Write header only for new file
        if not file_exists:
            writer.writeheader()

        # Write data rows
        writer.writerows(all_data_points)

    skipped = sum(1 for day_key in ['previousDay', 'currentDay']
                  if day_key in data and 'data' in data[day_key]
                  for point in data[day_key]['data']
                  if point['timestamp'] in existing_timestamps)

    return len(all_data_points), skipped


def get_csv_stats(csv_file: Path) -> Dict[str, Any]:
    """
    Get statistics about the CSV file.

    Args:
        csv_file: Path to CSV file

    Returns:
        dict: Statistics including row count, date range, file size
    """
    if not csv_file.exists():
        return {
            'exists': False,
            'row_count': 0,
            'file_size': 0
        }

    stats = {
        'exists': True,
        'file_size': csv_file.stat().st_size,
        'row_count': 0,
        'first_timestamp': None,
        'last_timestamp': None,
        'date_range_days': 0
    }

    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            stats['row_count'] = len(rows)

            if rows:
                stats['first_timestamp'] = rows[0]['timestamp']
                stats['last_timestamp'] = rows[-1]['timestamp']

                # Calculate date range
                first_dt = datetime.strptime(
                    stats['first_timestamp'].rsplit('-', 1)[0].strip(),
                    "%Y-%m-%d %H:%M:%S"
                )
                last_dt = datetime.strptime(
                    stats['last_timestamp'].rsplit('-', 1)[0].strip(),
                    "%Y-%m-%d %H:%M:%S"
                )
                stats['date_range_days'] = (last_dt - first_dt).days

    except Exception as e:
        print(f"Warning: Error reading CSV stats: {e}", file=sys.stderr)

    return stats


def print_summary(data: Dict[str, Any], added: int, skipped: int, stats: Dict[str, Any]):
    """Print summary of download and append operation."""
    print("\n" + "="*80)
    print("ERCOT ESR Data Download Summary")
    print("="*80)
    print(f"Last Updated: {data.get('lastUpdated', 'Unknown')}")
    print(f"Download Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Data from API
    for day_key in ['previousDay', 'currentDay']:
        if day_key in data:
            day_data = data[day_key]
            data_points = len(day_data.get('data', []))
            day_date = day_data.get('dayDate', 'Unknown')
            print(f"{day_key.capitalize():15} : {day_date} ({data_points} data points)")

    print()
    print(f"New Rows Added:       {added:,}")
    print(f"Duplicate Rows Skipped: {skipped:,}")
    print()

    # CSV file stats
    print(f"CSV File: {CSV_FILE.name}")
    print(f"  Total Rows:         {stats['row_count']:,}")
    print(f"  File Size:          {stats['file_size']:,} bytes")

    if stats['row_count'] > 0:
        print(f"  First Timestamp:    {stats['first_timestamp']}")
        print(f"  Last Timestamp:     {stats['last_timestamp']}")
        print(f"  Date Range:         {stats['date_range_days']} days")

        # Calculate expected vs actual
        expected_rows = (stats['date_range_days'] + 1) * 288  # 288 5-min intervals per day
        coverage_pct = (stats['row_count'] / expected_rows * 100) if expected_rows > 0 else 0
        print(f"  Coverage:           {coverage_pct:.1f}% ({stats['row_count']:,}/{expected_rows:,} expected)")

    print("="*80 + "\n")


def main():
    """Main execution function."""
    try:
        # Setup
        backup_dir = setup_directories()
        print(f"CSV File: {CSV_FILE}")

        # Create backup if file exists
        create_backup(CSV_FILE, backup_dir)

        # Download data
        print(f"Downloading ESR data from ERCOT...")
        data = download_esr_data()

        # Append to CSV
        added, skipped = append_to_csv(data, CSV_FILE)

        # Get updated stats
        stats = get_csv_stats(CSV_FILE)

        # Print summary
        print_summary(data, added, skipped, stats)

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

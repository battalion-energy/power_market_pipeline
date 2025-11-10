#!/usr/bin/env python3
"""
ERCOT Energy Storage Resources (ESR) Daily Data Downloader

Downloads and archives daily ESR data from ERCOT's dashboard API.
Saves both previousDay and currentDay data as JSON files with timestamps.

Data includes 5-minute resolution:
- Total Discharging (MW) - positive values, ESRs injecting to grid
- Total Charging (MW) - negative values, ESRs consuming from grid
- Net Output (MW) - sum of discharging and charging

Data Source: https://www.ercot.com/api/1/services/read/dashboards/energy-storage-resources.json
Dashboard: https://www.ercot.com/gridmktinfo/dashboards/energystorageresources
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any
import urllib.request
import urllib.error

# Configuration
API_URL = "https://www.ercot.com/api/1/services/read/dashboards/energy-storage-resources.json"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"

# Determine script directory
SCRIPT_DIR = Path(__file__).parent.absolute()
ARCHIVE_DIR = SCRIPT_DIR / "esr_archive"


def setup_archive_directory() -> Path:
    """Create archive directory if it doesn't exist."""
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    return ARCHIVE_DIR


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


def save_daily_archive(data: Dict[str, Any], archive_dir: Path) -> tuple[Path, Path]:
    """
    Save ESR data to archive with timestamp.

    Creates two files:
    1. Full timestamped archive with all data
    2. Daily file for each day of data (previousDay and currentDay)

    Args:
        data: ESR data from API
        archive_dir: Directory to save files

    Returns:
        tuple: (full_archive_path, summary info)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save full archive with timestamp
    full_archive_path = archive_dir / f"esr_full_{timestamp}.json"
    with open(full_archive_path, 'w') as f:
        json.dump(data, f, indent=2)

    saved_files = [str(full_archive_path)]

    # Save individual day files
    for day_key in ['previousDay', 'currentDay']:
        if day_key in data and 'dayDate' in data[day_key]:
            day_data = data[day_key]
            day_date_str = day_data['dayDate']

            # Parse the date from format: "2025-11-08 03:00:00-0600"
            try:
                day_date = datetime.strptime(day_date_str.split()[0], "%Y-%m-%d")
                date_str = day_date.strftime("%Y%m%d")

                # Create separate file for this day's data
                day_file_path = archive_dir / f"esr_{date_str}.json"

                # Create data package for this day
                day_package = {
                    "date": day_date_str,
                    "downloaded_at": data.get('lastUpdated'),
                    "download_timestamp": timestamp,
                    "data_points": len(day_data.get('data', [])),
                    "data": day_data['data']
                }

                # Only overwrite if newer data or file doesn't exist
                should_save = True
                if day_file_path.exists():
                    with open(day_file_path, 'r') as f:
                        existing = json.load(f)
                        # Keep the file with more data points
                        if existing.get('data_points', 0) >= day_package['data_points']:
                            should_save = False

                if should_save:
                    with open(day_file_path, 'w') as f:
                        json.dump(day_package, f, indent=2)
                    saved_files.append(str(day_file_path))

            except (ValueError, KeyError) as e:
                print(f"Warning: Could not parse date for {day_key}: {e}", file=sys.stderr)

    return full_archive_path, saved_files


def cleanup_old_full_archives(archive_dir: Path, keep_days: int = 7):
    """
    Remove full archive files older than keep_days.
    Daily files (esr_YYYYMMDD.json) are kept indefinitely.

    Args:
        archive_dir: Archive directory
        keep_days: Number of days to keep full archives
    """
    cutoff_date = datetime.now() - timedelta(days=keep_days)

    for file_path in archive_dir.glob("esr_full_*.json"):
        # Extract timestamp from filename: esr_full_YYYYMMDD_HHMMSS.json
        try:
            timestamp_str = file_path.stem.replace("esr_full_", "")
            file_date = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")

            if file_date < cutoff_date:
                file_path.unlink()
                print(f"Removed old archive: {file_path.name}")
        except (ValueError, IndexError):
            # Skip files that don't match expected format
            continue


def print_summary(data: Dict[str, Any], saved_files: list):
    """Print summary of downloaded data."""
    print("\n" + "="*80)
    print("ERCOT ESR Data Download Summary")
    print("="*80)
    print(f"Last Updated: {data.get('lastUpdated', 'Unknown')}")
    print(f"Download Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    for day_key in ['previousDay', 'currentDay']:
        if day_key in data:
            day_data = data[day_key]
            data_points = len(day_data.get('data', []))
            day_date = day_data.get('dayDate', 'Unknown')
            print(f"{day_key.capitalize():15} : {day_date} ({data_points} data points)")

    print()
    print(f"Files Saved ({len(saved_files)}):")
    for file_path in saved_files:
        file_size = Path(file_path).stat().st_size
        print(f"  - {Path(file_path).name} ({file_size:,} bytes)")
    print("="*80)


def main():
    """Main execution function."""
    try:
        # Setup
        archive_dir = setup_archive_directory()
        print(f"Archive directory: {archive_dir}")

        # Download data
        print(f"Downloading ESR data from ERCOT...")
        data = download_esr_data()

        # Save archives
        full_archive_path, saved_files = save_daily_archive(data, archive_dir)

        # Cleanup old files
        cleanup_old_full_archives(archive_dir, keep_days=7)

        # Print summary
        print_summary(data, saved_files)

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

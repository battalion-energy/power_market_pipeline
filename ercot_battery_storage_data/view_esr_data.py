#!/usr/bin/env python3
"""
ERCOT ESR Data Viewer

Quick utility to view and analyze collected ESR data.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any


ARCHIVE_DIR = Path(__file__).parent / "esr_archive"


def list_available_dates() -> List[str]:
    """List all available dates in the archive."""
    dates = []
    for file_path in sorted(ARCHIVE_DIR.glob("esr_*.json")):
        if file_path.stem.startswith("esr_") and not file_path.stem.startswith("esr_full_"):
            date_str = file_path.stem.replace("esr_", "")
            try:
                # Validate it's a date
                datetime.strptime(date_str, "%Y%m%d")
                dates.append(date_str)
            except ValueError:
                continue
    return dates


def load_date(date_str: str) -> Dict[str, Any]:
    """Load data for a specific date."""
    file_path = ARCHIVE_DIR / f"esr_{date_str}.json"
    if not file_path.exists():
        raise FileNotFoundError(f"No data found for {date_str}")

    with open(file_path, 'r') as f:
        return json.load(f)


def print_summary(date_str: str):
    """Print summary statistics for a date."""
    data = load_date(date_str)

    print(f"\n{'='*80}")
    print(f"ESR Data Summary for {date_str}")
    print(f"{'='*80}")
    print(f"Date: {data['date']}")
    print(f"Downloaded: {data['downloaded_at']}")
    print(f"Data Points: {data['data_points']}")
    print()

    if not data['data']:
        print("No data points available.")
        return

    # Calculate statistics
    charging = [p['totalCharging'] for p in data['data']]
    discharging = [p['totalDischarging'] for p in data['data']]
    net_output = [p['netOutput'] for p in data['data']]

    print(f"{'Metric':<20} {'Min':>12} {'Max':>12} {'Avg':>12}")
    print(f"{'-'*20} {'-'*12} {'-'*12} {'-'*12}")
    print(f"{'Charging (MW)':<20} {min(charging):>12.2f} {max(charging):>12.2f} {sum(charging)/len(charging):>12.2f}")
    print(f"{'Discharging (MW)':<20} {min(discharging):>12.2f} {max(discharging):>12.2f} {sum(discharging)/len(discharging):>12.2f}")
    print(f"{'Net Output (MW)':<20} {min(net_output):>12.2f} {max(net_output):>12.2f} {sum(net_output)/len(net_output):>12.2f}")

    print(f"\n{'='*80}")

    # Find peak events
    max_charging_idx = charging.index(min(charging))
    max_discharging_idx = discharging.index(max(discharging))

    print("\nPeak Events:")
    print(f"  Max Charging:    {charging[max_charging_idx]:,.2f} MW at {data['data'][max_charging_idx]['timestamp']}")
    print(f"  Max Discharging: {discharging[max_discharging_idx]:,.2f} MW at {data['data'][max_discharging_idx]['timestamp']}")
    print(f"{'='*80}\n")


def print_hourly_summary(date_str: str):
    """Print hourly average summary."""
    data = load_date(date_str)

    print(f"\n{'='*80}")
    print(f"Hourly Average Summary for {date_str}")
    print(f"{'='*80}")

    # Group by hour (12 5-minute intervals per hour)
    hourly_data = {}
    for point in data['data']:
        # Parse timestamp - format is "2025-11-08 00:00:00-0600"
        timestamp_str = point['timestamp'].rsplit('-', 1)[0].strip()  # Remove timezone
        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        hour = timestamp.hour

        if hour not in hourly_data:
            hourly_data[hour] = {
                'charging': [],
                'discharging': [],
                'net_output': []
            }

        hourly_data[hour]['charging'].append(point['totalCharging'])
        hourly_data[hour]['discharging'].append(point['totalDischarging'])
        hourly_data[hour]['net_output'].append(point['netOutput'])

    print(f"\n{'Hour':<6} {'Avg Charging':>14} {'Avg Discharging':>16} {'Avg Net Output':>15} {'Samples':>8}")
    print(f"{'-'*6} {'-'*14} {'-'*16} {'-'*15} {'-'*8}")

    for hour in sorted(hourly_data.keys()):
        avg_charging = sum(hourly_data[hour]['charging']) / len(hourly_data[hour]['charging'])
        avg_discharging = sum(hourly_data[hour]['discharging']) / len(hourly_data[hour]['discharging'])
        avg_net = sum(hourly_data[hour]['net_output']) / len(hourly_data[hour]['net_output'])
        samples = len(hourly_data[hour]['charging'])

        print(f"{hour:>4}:00 {avg_charging:>14.2f} {avg_discharging:>16.2f} {avg_net:>15.2f} {samples:>8}")

    print(f"{'='*80}\n")


def main():
    """Main execution function."""
    if not ARCHIVE_DIR.exists():
        print(f"Archive directory not found: {ARCHIVE_DIR}")
        sys.exit(1)

    dates = list_available_dates()

    if not dates:
        print("No ESR data found in archive.")
        print(f"Archive directory: {ARCHIVE_DIR}")
        sys.exit(1)

    if len(sys.argv) < 2:
        print("Available dates in archive:")
        print()
        for date_str in dates:
            formatted_date = datetime.strptime(date_str, "%Y%m%d").strftime("%Y-%m-%d")
            file_path = ARCHIVE_DIR / f"esr_{date_str}.json"
            file_size = file_path.stat().st_size
            print(f"  {formatted_date} ({date_str}) - {file_size:,} bytes")

        print()
        print(f"Total dates available: {len(dates)}")
        print()
        print("Usage:")
        print(f"  {sys.argv[0]} <YYYYMMDD>           - View summary for a date")
        print(f"  {sys.argv[0]} <YYYYMMDD> --hourly  - View hourly summary")
        print()
        print("Examples:")
        if dates:
            print(f"  {sys.argv[0]} {dates[-1]}")
            print(f"  {sys.argv[0]} {dates[-1]} --hourly")
        sys.exit(0)

    date_str = sys.argv[1]

    if date_str not in dates:
        print(f"No data found for {date_str}")
        print(f"Available dates: {', '.join(dates)}")
        sys.exit(1)

    if len(sys.argv) > 2 and sys.argv[2] == '--hourly':
        print_hourly_summary(date_str)
    else:
        print_summary(date_str)


if __name__ == "__main__":
    main()

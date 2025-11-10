#!/usr/bin/env python3
"""
ERCOT ESR CSV Data Viewer

Quick utility to view and analyze the contiguous CSV file.
"""

import csv
import sys
from datetime import datetime
from pathlib import Path
from collections import defaultdict


CSV_FILE = Path(__file__).parent / "esr_historical_data.csv"


def load_csv_data():
    """Load CSV data and return as list of dicts."""
    if not CSV_FILE.exists():
        print(f"CSV file not found: {CSV_FILE}")
        sys.exit(1)

    with open(CSV_FILE, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)


def print_summary():
    """Print summary statistics."""
    data = load_csv_data()

    if not data:
        print("No data in CSV file.")
        return

    print("\n" + "="*80)
    print("ERCOT ESR Historical Data Summary")
    print("="*80)
    print(f"CSV File: {CSV_FILE.name}")
    print(f"File Size: {CSV_FILE.stat().st_size:,} bytes")
    print(f"Total Rows: {len(data):,}")
    print()

    # Parse timestamps
    first_time = data[0]['timestamp']
    last_time = data[-1]['timestamp']

    first_dt = datetime.strptime(first_time.rsplit('-', 1)[0].strip(), "%Y-%m-%d %H:%M:%S")
    last_dt = datetime.strptime(last_time.rsplit('-', 1)[0].strip(), "%Y-%m-%d %H:%M:%S")
    date_range = (last_dt - first_dt).days

    print(f"First Timestamp: {first_time}")
    print(f"Last Timestamp:  {last_time}")
    print(f"Date Range:      {date_range} days")
    print()

    # Calculate expected rows
    expected_rows = (date_range + 1) * 288
    coverage = (len(data) / expected_rows * 100) if expected_rows > 0 else 0
    print(f"Expected Rows:   {expected_rows:,} (288 per day × {date_range + 1} days)")
    print(f"Coverage:        {coverage:.1f}%")
    print()

    # Calculate statistics
    charging = [float(row['total_charging_mw']) for row in data]
    discharging = [float(row['total_discharging_mw']) for row in data]
    net_output = [float(row['net_output_mw']) for row in data]

    print("Overall Statistics:")
    print(f"{'Metric':<25} {'Min':>12} {'Max':>12} {'Avg':>12}")
    print(f"{'-'*25} {'-'*12} {'-'*12} {'-'*12}")
    print(f"{'Charging (MW)':<25} {min(charging):>12.2f} {max(charging):>12.2f} {sum(charging)/len(charging):>12.2f}")
    print(f"{'Discharging (MW)':<25} {min(discharging):>12.2f} {max(discharging):>12.2f} {sum(discharging)/len(discharging):>12.2f}")
    print(f"{'Net Output (MW)':<25} {min(net_output):>12.2f} {max(net_output):>12.2f} {sum(net_output)/len(net_output):>12.2f}")
    print()

    # Find peak events
    max_charging_idx = charging.index(min(charging))
    max_discharging_idx = discharging.index(max(discharging))

    print("Peak Events:")
    print(f"  Max Charging:    {charging[max_charging_idx]:,.2f} MW at {data[max_charging_idx]['timestamp']}")
    print(f"  Max Discharging: {discharging[max_discharging_idx]:,.2f} MW at {data[max_discharging_idx]['timestamp']}")

    print("="*80 + "\n")


def print_daily_summary():
    """Print statistics by day."""
    data = load_csv_data()

    if not data:
        print("No data in CSV file.")
        return

    print("\n" + "="*80)
    print("Daily Summary")
    print("="*80)

    # Group by date
    daily_data = defaultdict(lambda: {
        'charging': [],
        'discharging': [],
        'net_output': [],
        'count': 0
    })

    for row in data:
        timestamp = row['timestamp'].rsplit('-', 1)[0].strip()
        dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        date_key = dt.strftime("%Y-%m-%d")

        daily_data[date_key]['charging'].append(float(row['total_charging_mw']))
        daily_data[date_key]['discharging'].append(float(row['total_discharging_mw']))
        daily_data[date_key]['net_output'].append(float(row['net_output_mw']))
        daily_data[date_key]['count'] += 1

    print(f"\n{'Date':<12} {'Points':>8} {'Avg Charging':>14} {'Avg Discharging':>16} {'Avg Net':>12}")
    print(f"{'-'*12} {'-'*8} {'-'*14} {'-'*16} {'-'*12}")

    for date_key in sorted(daily_data.keys()):
        day = daily_data[date_key]
        avg_charging = sum(day['charging']) / len(day['charging'])
        avg_discharging = sum(day['discharging']) / len(day['discharging'])
        avg_net = sum(day['net_output']) / len(day['net_output'])

        print(f"{date_key:<12} {day['count']:>8} {avg_charging:>14.2f} {avg_discharging:>16.2f} {avg_net:>12.2f}")

    print("="*80 + "\n")


def print_hourly_summary():
    """Print statistics by hour of day."""
    data = load_csv_data()

    if not data:
        print("No data in CSV file.")
        return

    print("\n" + "="*80)
    print("Hourly Summary (Average Across All Days)")
    print("="*80)

    # Group by hour
    hourly_data = defaultdict(lambda: {
        'charging': [],
        'discharging': [],
        'net_output': []
    })

    for row in data:
        timestamp = row['timestamp'].rsplit('-', 1)[0].strip()
        dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        hour = dt.hour

        hourly_data[hour]['charging'].append(float(row['total_charging_mw']))
        hourly_data[hour]['discharging'].append(float(row['total_discharging_mw']))
        hourly_data[hour]['net_output'].append(float(row['net_output_mw']))

    print(f"\n{'Hour':<6} {'Samples':>8} {'Avg Charging':>14} {'Avg Discharging':>16} {'Avg Net':>12}")
    print(f"{'-'*6} {'-'*8} {'-'*14} {'-'*16} {'-'*12}")

    for hour in sorted(hourly_data.keys()):
        data_hour = hourly_data[hour]
        avg_charging = sum(data_hour['charging']) / len(data_hour['charging'])
        avg_discharging = sum(data_hour['discharging']) / len(data_hour['discharging'])
        avg_net = sum(data_hour['net_output']) / len(data_hour['net_output'])

        print(f"{hour:>4}:00 {len(data_hour['charging']):>8} {avg_charging:>14.2f} {avg_discharging:>16.2f} {avg_net:>12.2f}")

    print("="*80 + "\n")


def print_recent():
    """Print most recent data points."""
    data = load_csv_data()

    if not data:
        print("No data in CSV file.")
        return

    print("\n" + "="*80)
    print("Most Recent Data Points")
    print("="*80)

    print(f"\n{'Timestamp':<26} {'Charging':>12} {'Discharging':>12} {'Net Output':>12}")
    print(f"{'-'*26} {'-'*12} {'-'*12} {'-'*12}")

    # Show last 12 points (1 hour)
    for row in data[-12:]:
        print(f"{row['timestamp']:<26} "
              f"{float(row['total_charging_mw']):>12.2f} "
              f"{float(row['total_discharging_mw']):>12.2f} "
              f"{float(row['net_output_mw']):>12.2f}")

    print("="*80 + "\n")


def main():
    """Main execution."""
    if not CSV_FILE.exists():
        print(f"CSV file not found: {CSV_FILE}")
        print("\nRun the download script first:")
        print("  python3 ercot_battery_storage_data/download_esr_daily_csv.py")
        sys.exit(1)

    if len(sys.argv) < 2:
        print("\nERCOT ESR CSV Data Viewer")
        print("\nUsage:")
        print(f"  {sys.argv[0]} summary    - Overall summary and statistics")
        print(f"  {sys.argv[0]} daily      - Statistics by day")
        print(f"  {sys.argv[0]} hourly     - Average statistics by hour of day")
        print(f"  {sys.argv[0]} recent     - Most recent 12 data points (1 hour)")
        print()
        sys.exit(0)

    command = sys.argv[1].lower()

    if command == 'summary':
        print_summary()
    elif command == 'daily':
        print_daily_summary()
    elif command == 'hourly':
        print_hourly_summary()
    elif command == 'recent':
        print_recent()
    else:
        print(f"Unknown command: {command}")
        print("Valid commands: summary, daily, hourly, recent")
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Monitor ISO parquet conversion progress.

Checks log files for errors, tracks progress, and spot-checks output.
Usage:
    python3 monitor_conversion.py
    python3 monitor_conversion.py --watch  # Continuous monitoring
"""

import sys
import argparse
import time
from pathlib import Path
from datetime import datetime
import pandas as pd
import pyarrow.parquet as pq

def check_log_files():
    """Check recent log files for errors and progress."""
    log_dir = Path('/pool/ssd8tb/data/iso/unified_iso_data/logs')

    if not log_dir.exists():
        print("❌ Log directory not found")
        return []

    # Get recent log files (from today)
    today = datetime.now().strftime('%Y%m%d')
    log_files = sorted(log_dir.glob(f"*_conversion_{today}*.log"), reverse=True)

    if not log_files:
        print(f"📝 No conversion logs found for today ({today})")
        return []

    print(f"\n{'='*80}")
    print(f"📝 LOG FILE ANALYSIS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")

    results = []

    for log_file in log_files:
        iso_name = log_file.name.split('_')[0].upper()
        size_mb = log_file.stat().st_size / 1024 / 1024

        print(f"📄 {iso_name}: {log_file.name} ({size_mb:.1f} MB)")

        # Read last 50 lines
        with open(log_file, 'r') as f:
            lines = f.readlines()
            last_lines = lines[-50:] if len(lines) > 50 else lines

        # Check for errors
        errors = [line for line in last_lines if 'ERROR' in line or 'Error' in line or 'FAILED' in line]
        warnings = [line for line in last_lines if 'WARNING' in line or 'Warning' in line]

        # Check for completion
        completed = any('completed' in line.lower() or 'success' in line.lower() for line in last_lines[-10:])

        # Check for progress indicators
        processing = [line for line in last_lines if 'Processing' in line or 'Converting' in line or 'Writing' in line]

        status = "✅ COMPLETED" if completed else "⏳ IN PROGRESS"
        if errors:
            status = "❌ ERRORS"

        print(f"  Status: {status}")

        if errors:
            print(f"  ⚠️  {len(errors)} errors found:")
            for err in errors[-3:]:  # Show last 3 errors
                print(f"    {err.strip()[:100]}")

        if warnings:
            print(f"  ⚠️  {len(warnings)} warnings")

        if processing:
            print(f"  📊 Latest progress: {processing[-1].strip()[:80]}")

        print()

        results.append({
            'iso': iso_name,
            'status': status,
            'errors': len(errors),
            'warnings': len(warnings),
            'size_mb': size_mb
        })

    return results


def check_parquet_files():
    """Spot-check parquet files for basic validity."""
    parquet_dir = Path('/pool/ssd8tb/data/iso/unified_iso_data/parquet')

    if not parquet_dir.exists():
        print("❌ Parquet directory not found")
        return

    print(f"\n{'='*80}")
    print(f"📦 PARQUET FILE SPOT-CHECK")
    print(f"{'='*80}\n")

    total_files = 0
    total_size_mb = 0

    for iso_dir in sorted(parquet_dir.glob("*")):
        if not iso_dir.is_dir():
            continue

        iso_name = iso_dir.name.upper()
        parquet_files = list(iso_dir.rglob("*.parquet"))

        if not parquet_files:
            continue

        iso_size = sum(f.stat().st_size for f in parquet_files) / 1024 / 1024
        total_files += len(parquet_files)
        total_size_mb += iso_size

        print(f"📊 {iso_name}:")
        print(f"  Files: {len(parquet_files)}")
        print(f"  Size: {iso_size:.1f} MB")

        # Spot check the most recent file
        latest_file = max(parquet_files, key=lambda f: f.stat().st_mtime)

        try:
            # Read parquet metadata
            table = pq.read_table(latest_file)
            num_rows = table.num_rows
            num_cols = table.num_columns

            print(f"  Latest file: {latest_file.name}")
            print(f"    Rows: {num_rows:,}")
            print(f"    Columns: {num_cols}")

            # Sample first few rows
            df_sample = table.to_pandas().head(3)

            # Check for key columns
            required_cols = ['datetime_utc', 'lmp_total', 'settlement_location']
            missing_cols = [col for col in required_cols if col not in df_sample.columns]

            if missing_cols:
                print(f"    ⚠️  Missing columns: {missing_cols}")
            else:
                print(f"    ✅ Schema valid")

            # Check for null prices
            if 'lmp_total' in df_sample.columns:
                null_prices = df_sample['lmp_total'].isnull().sum()
                if null_prices > 0:
                    print(f"    ⚠️  {null_prices} null prices in sample")

            # Show date range
            if 'datetime_utc' in df_sample.columns:
                df_full = table.to_pandas()
                date_min = pd.to_datetime(df_full['datetime_utc']).min()
                date_max = pd.to_datetime(df_full['datetime_utc']).max()
                print(f"    Date range: {date_min} to {date_max}")

                # Check unique locations
                if 'settlement_location' in df_full.columns:
                    unique_locs = df_full['settlement_location'].nunique()
                    print(f"    Unique locations: {unique_locs:,}")

        except Exception as e:
            print(f"    ❌ Error reading file: {e}")

        print()

    print(f"{'='*80}")
    print(f"💾 TOTALS:")
    print(f"  Files: {total_files}")
    print(f"  Size: {total_size_mb:.1f} MB ({total_size_mb/1024:.2f} GB)")
    print(f"{'='*80}\n")


def check_running_processes():
    """Check for running converter processes."""
    import subprocess

    try:
        result = subprocess.run(
            ['ps', 'aux'],
            capture_output=True,
            text=True,
            timeout=5
        )

        converter_procs = [
            line for line in result.stdout.split('\n')
            if 'parquet_converter.py' in line and 'grep' not in line
        ]

        if converter_procs:
            print(f"\n{'='*80}")
            print(f"🔄 RUNNING PROCESSES:")
            print(f"{'='*80}\n")

            for proc in converter_procs:
                parts = proc.split()
                if len(parts) > 10:
                    cpu = parts[2]
                    mem = parts[3]
                    cmd = ' '.join(parts[10:])
                    print(f"  CPU: {cpu}% | MEM: {mem}% | {cmd[:60]}")
            print()
        else:
            print("\n✅ No converter processes currently running\n")

    except Exception as e:
        print(f"⚠️  Could not check processes: {e}")


def main():
    parser = argparse.ArgumentParser(description='Monitor ISO parquet conversion')
    parser.add_argument('--watch', action='store_true', help='Continuous monitoring (updates every 60s)')
    parser.add_argument('--interval', type=int, default=60, help='Watch interval in seconds')

    args = parser.parse_args()

    try:
        while True:
            # Clear screen for watch mode
            if args.watch:
                print("\033[2J\033[H")  # Clear screen and move cursor to top

            check_running_processes()
            log_results = check_log_files()
            check_parquet_files()

            if not args.watch:
                break

            print(f"⏰ Next update in {args.interval} seconds... (Ctrl+C to stop)")
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped")


if __name__ == "__main__":
    main()

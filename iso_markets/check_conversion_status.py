#!/usr/bin/env python3
"""
Check status of ISO parquet conversions.
"""

import os
from pathlib import Path
from datetime import datetime
import json

def check_status():
    """Check conversion status."""
    parquet_dir = Path('/home/enrico/data/unified_iso_data/parquet')
    metadata_dir = Path('/home/enrico/data/unified_iso_data/metadata')

    print("=" * 80)
    print("UNIFIED ISO PARQUET CONVERSION STATUS")
    print("=" * 80)
    print(f"Checked at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Check each ISO
    isos = ['pjm', 'caiso', 'miso', 'nyiso', 'isone', 'spp', 'ercot']

    for iso in isos:
        iso_dir = parquet_dir / iso
        if iso_dir.exists():
            print(f"\n{iso.upper()}:")
            print("-" * 40)

            # Count parquet files by type
            market_types = {}
            total_size = 0

            for parquet_file in iso_dir.rglob("*.parquet"):
                rel_path = parquet_file.relative_to(iso_dir)
                market_type = rel_path.parts[0]  # First directory

                if market_type not in market_types:
                    market_types[market_type] = {'count': 0, 'size': 0, 'files': []}

                size = parquet_file.stat().st_size
                market_types[market_type]['count'] += 1
                market_types[market_type]['size'] += size
                market_types[market_type]['files'].append(parquet_file.name)
                total_size += size

            # Print summary
            for market_type, info in sorted(market_types.items()):
                size_mb = info['size'] / (1024 ** 2)
                size_gb = info['size'] / (1024 ** 3)

                if size_gb > 1:
                    size_str = f"{size_gb:.2f} GB"
                else:
                    size_str = f"{size_mb:.1f} MB"

                print(f"  {market_type:30s}: {info['count']:3d} files, {size_str:>12s}")

                # Show year coverage
                years = sorted(set([f.split('_')[-1].replace('.parquet', '') for f in info['files']]))
                if years:
                    print(f"    Years: {', '.join(years)}")

            total_gb = total_size / (1024 ** 3)
            print(f"\n  Total: {total_gb:.2f} GB")

        else:
            print(f"\n{iso.upper()}: No data")

    # Check metadata
    print("\n\n" + "=" * 80)
    print("METADATA FILES")
    print("=" * 80)

    for metadata_type in ['hubs', 'nodes', 'zones', 'ancillary_services']:
        metadata_subdir = metadata_dir / metadata_type
        if metadata_subdir.exists():
            json_files = list(metadata_subdir.glob("*.json"))
            if json_files:
                print(f"\n{metadata_type.upper()}:")
                for json_file in sorted(json_files):
                    with open(json_file) as f:
                        data = json.load(f)

                    iso_name = json_file.stem.replace(f'_{metadata_type}', '')
                    count_key = f"total_{metadata_type}" if f"total_{metadata_type}" in data else f"{metadata_type}"

                    if count_key in data:
                        count = data[count_key]
                        print(f"  {iso_name:10s}: {count} items")
                    elif metadata_type in data:
                        count = len(data[metadata_type])
                        print(f"  {iso_name:10s}: {count} items")
                    else:
                        print(f"  {iso_name:10s}: metadata file exists")

    # Check for running processes
    print("\n\n" + "=" * 80)
    print("RUNNING PROCESSES")
    print("=" * 80)

    import subprocess
    result = subprocess.run(
        ['ps', 'aux'],
        capture_output=True,
        text=True
    )

    converter_processes = [line for line in result.stdout.split('\n')
                          if 'parquet_converter.py' in line and 'grep' not in line]

    if converter_processes:
        print("\nActive converters:")
        for proc in converter_processes:
            parts = proc.split()
            if len(parts) >= 11:
                user = parts[0]
                pid = parts[1]
                cpu = parts[2]
                mem = parts[3]
                command = ' '.join(parts[10:])
                print(f"  PID {pid}: {command}")
                print(f"    CPU: {cpu}%,  MEM: {mem}%")
    else:
        print("\nNo active converters running")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    check_status()

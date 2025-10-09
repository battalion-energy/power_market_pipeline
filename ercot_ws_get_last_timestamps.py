#!/usr/bin/env python3
"""
Check last timestamps in ERCOT parquet files to determine what data needs to be downloaded
from the Web Service API.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd
import pyarrow.parquet as pq


def get_last_timestamp_from_parquet(parquet_path: Path, datetime_column: str = "datetime") -> datetime:
    """Read the last timestamp from a parquet file."""
    try:
        # Read parquet file metadata first to avoid loading entire file
        parquet_file = pq.ParquetFile(parquet_path)

        # Get the last row group
        last_row_group_idx = parquet_file.num_row_groups - 1

        # Read only the last row group
        df = parquet_file.read_row_group(last_row_group_idx).to_pandas()

        # Get the last timestamp
        if datetime_column in df.columns:
            last_ts = pd.to_datetime(df[datetime_column]).max()
            # Handle NaT (Not a Time) values
            if pd.isna(last_ts):
                return None
            return last_ts
        else:
            # Try common column variations
            for col in ['DeliveryDate', 'delivery_date', 'SCED Time Stamp', 'SCEDTimestamp', 'Delivery Date']:
                if col in df.columns:
                    last_ts = pd.to_datetime(df[col]).max()
                    # Handle NaT (Not a Time) values
                    if pd.isna(last_ts):
                        continue
                    return last_ts

            print(f"Warning: No datetime column found in {parquet_path}")
            print(f"Available columns: {df.columns.tolist()}")
            return None

    except Exception as e:
        print(f"Error reading {parquet_path}: {str(e)}")
        return None


def check_dataset_timestamps(base_dir: Path, dataset_name: str, pattern: str = "*.parquet") -> Dict:
    """Check timestamps for a dataset directory."""
    dataset_dir = base_dir / dataset_name

    if not dataset_dir.exists():
        return {
            "status": "not_found",
            "path": str(dataset_dir),
            "last_timestamp": None,
            "file_count": 0
        }

    # Find all parquet files (excluding reports)
    parquet_files = [
        f for f in dataset_dir.glob(pattern)
        if not f.name.startswith("gaps_") and not f.name.startswith("schema")
    ]

    if not parquet_files:
        return {
            "status": "no_files",
            "path": str(dataset_dir),
            "last_timestamp": None,
            "file_count": 0
        }

    # Sort files by year (assuming YYYY.parquet format)
    parquet_files.sort()

    # Get the last file (most recent year)
    last_file = parquet_files[-1]
    last_timestamp = get_last_timestamp_from_parquet(last_file)

    return {
        "status": "ok",
        "path": str(dataset_dir),
        "last_file": str(last_file.name),
        "last_timestamp": last_timestamp.isoformat() if last_timestamp else None,
        "file_count": len(parquet_files),
        "files": [f.name for f in parquet_files]
    }


def main():
    """Main function to check all ERCOT datasets."""

    # Base directory for ERCOT data
    base_dir = Path(os.getenv("ERCOT_DATA_DIR", "/pool/ssd8tb/data/iso/ERCOT"))
    rollup_dir = base_dir / "ercot_market_data" / "ERCOT_data" / "rollup_files"

    print(f"Checking ERCOT data in: {rollup_dir}")
    print("=" * 80)

    # Define datasets to check
    datasets = {
        "DA_prices": {
            "name": "Day-Ahead Market Settlement Point Prices",
            "web_service_endpoint": "np4-190-cd/dam_stlmnt_pnt_prices",
            "priority": 1
        },
        "RT_prices": {
            "name": "Real-Time Market Settlement Point Prices",
            "web_service_endpoint": "np4-191-cd/spp_node_zone_hub",
            "priority": 2
        },
        "AS_prices": {
            "name": "Ancillary Services Prices",
            "web_service_endpoint": "np6-788-cd/as_prices",
            "priority": 3
        },
        "DAM_Gen_Resources": {
            "name": "DAM Generation Resource Data (60-day disclosure)",
            "web_service_endpoint": "np3-966-cd/60d_dam_gen_res_data",
            "priority": 4
        },
        "DAM_Load_Resources": {
            "name": "DAM Load Resource Data (60-day disclosure)",
            "web_service_endpoint": "np3-966-cd/60d_dam_load_res_data",
            "priority": 6
        },
        "SCED_Gen_Resources": {
            "name": "SCED Generation Resource Data (60-day disclosure)",
            "web_service_endpoint": "np3-965-cd/60d_sced_gen_res_data",
            "priority": 5
        },
        "SCED_Load_Resources": {
            "name": "SCED Load Resource Data (60-day disclosure)",
            "web_service_endpoint": "np3-965-cd/60d_sced_load_res_data",
            "priority": 7
        }
    }

    results = {}

    for dataset_key, dataset_info in datasets.items():
        print(f"\n{dataset_info['name']}")
        print("-" * 80)

        result = check_dataset_timestamps(rollup_dir, dataset_key)
        result.update({
            "display_name": dataset_info["name"],
            "web_service_endpoint": dataset_info["web_service_endpoint"],
            "priority": dataset_info["priority"]
        })

        results[dataset_key] = result

        if result["status"] == "ok":
            print(f"✓ Status: OK")
            print(f"  Path: {result['path']}")
            print(f"  Files: {result['file_count']} parquet files")
            print(f"  Latest file: {result['last_file']}")
            print(f"  Last timestamp: {result['last_timestamp']}")

            if result['last_timestamp']:
                last_dt = datetime.fromisoformat(result['last_timestamp'])
                days_old = (datetime.now() - last_dt).days
                print(f"  Data age: {days_old} days old")

                if days_old > 7:
                    print(f"  ⚠️  WARNING: Data is more than 7 days old!")
        else:
            print(f"✗ Status: {result['status']}")
            print(f"  Path: {result['path']}")

    # Save results to JSON
    output_file = Path("ercot_data_status.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"Results saved to: {output_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY - Data freshness and download priorities:")
    print("=" * 80)

    # Sort by priority
    sorted_datasets = sorted(results.items(), key=lambda x: x[1].get("priority", 999))

    for dataset_key, result in sorted_datasets:
        if result["status"] == "ok" and result["last_timestamp"]:
            last_dt = datetime.fromisoformat(result["last_timestamp"])
            days_old = (datetime.now() - last_dt).days
            status_icon = "⚠️ " if days_old > 7 else "✓ "
            print(f"{status_icon} Priority {result['priority']}: {result['display_name']}")
            print(f"   Last data: {result['last_timestamp']} ({days_old} days old)")
            print(f"   Endpoint: {result['web_service_endpoint']}")
        else:
            print(f"✗ Priority {result['priority']}: {result['display_name']}")
            print(f"   Status: {result['status']}")

    return results


if __name__ == "__main__":
    main()

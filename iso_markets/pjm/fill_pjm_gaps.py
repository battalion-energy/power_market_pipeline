#!/usr/bin/env python3
"""
PJM Data Gap Detection and Filling Script

This script:
1. Scans all PJM data directories for .gaps.json metadata files
2. Attempts to re-download missing chunks
3. Updates the CSV files and removes .gaps.json files when complete

Usage:
    python fill_pjm_gaps.py                          # Fill all gaps
    python fill_pjm_gaps.py --data-type rt_hourly_nodal  # Specific type
    python fill_pjm_gaps.py --detect-only            # Just report gaps, don't fill
"""

import os
import sys
import argparse
import logging
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from pjm_api_client import PJMAPIClient

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PJM_DATA_DIR = Path(os.getenv('PJM_DATA_DIR', '/home/enrico/data/PJM_data'))

MAX_RETRIES = 10  # Increased from 5 to 10 for better resilience
BASE_RETRY_DELAY = 30  # Start with 30 seconds
RATE_LIMIT_BASE_DELAY = 120  # Start with 2 minutes for 429 errors (more aggressive)

DATA_TYPES = {
    "da_nodal": {
        "dir": PJM_DATA_DIR / "csv_files/da_nodal",
        "api_method": "get_day_ahead_lmps",
        "granularity": "hourly",
        "description": "Day-Ahead Nodal LMPs"
    },
    "rt_hourly_nodal": {
        "dir": PJM_DATA_DIR / "csv_files/rt_hourly_nodal",
        "api_method": "get_rt_hourly_lmps",
        "granularity": "hourly",
        "description": "Real-Time Hourly Nodal LMPs"
    },
    "rt_5min_nodal": {
        "dir": PJM_DATA_DIR / "csv_files/rt_5min_nodal",
        "api_method": "get_rt_fivemin_lmps",
        "granularity": "5min",
        "description": "Real-Time 5-Min Nodal LMPs"
    }
}


def find_gap_files(data_type: str) -> List[Path]:
    """Find all .gaps.json files for a data type."""
    config = DATA_TYPES[data_type]
    data_dir = config["dir"]

    if not data_dir.exists():
        return []

    gap_files = list(data_dir.glob("*.gaps.json"))
    return sorted(gap_files)


def download_chunk_with_retry(client: PJMAPIClient, api_method: str,
                               start_time_str: str, end_time_str: str,
                               chunk_label: str):
    """Download a single chunk with retry logic."""
    method = getattr(client, api_method)

    for attempt in range(MAX_RETRIES):
        try:
            data = method(
                start_date=start_time_str,
                end_date=end_time_str,
                pnode_id=None,
                use_exact_times=True
            )

            if not data:
                return None

            items = data.get('items', data.get('data', [data])) if isinstance(data, dict) else data
            if not items:
                return None

            df = pd.DataFrame(items)
            logger.info(f"  ✓ {chunk_label}: {len(df):,} rows")
            time.sleep(2)  # Rate limit protection
            return df

        except Exception as e:
            error_msg = str(e)
            is_rate_limit = '429' in error_msg or 'rate limit' in error_msg.lower()

            if attempt < MAX_RETRIES - 1:
                # Use much longer backoff for rate limit errors (429)
                if is_rate_limit:
                    wait_time = RATE_LIMIT_BASE_DELAY * (2 ** attempt)  # 2min, 4min, 8min, 16min, 32min...
                    logger.warning(f"  ⚠️  Rate limit (429) on attempt {attempt + 1}/{MAX_RETRIES} for {chunk_label}")
                else:
                    wait_time = BASE_RETRY_DELAY * (2 ** attempt)  # 30s, 60s, 120s, 240s...
                    logger.warning(f"  ⚠️  Attempt {attempt + 1}/{MAX_RETRIES} failed for {chunk_label}: {error_msg}")
                logger.warning(f"  ⏳ Waiting {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"  ✗ All {MAX_RETRIES} attempts failed for {chunk_label}: {error_msg}")
                return None

    return None


def fill_gaps_for_file(client: PJMAPIClient, gap_file: Path, data_type: str):
    """Fill gaps for a single file."""
    config = DATA_TYPES[data_type]

    # Load gap metadata
    with open(gap_file, 'r') as f:
        metadata = json.load(f)

    date_str = metadata['date']
    failed_chunks = metadata['failed_chunks']

    logger.info(f"\n{'='*80}")
    logger.info(f"Filling gaps for {data_type} - {date_str}")
    logger.info(f"  {len(failed_chunks)} chunks to retry")
    logger.info(f"{'='*80}")

    # Load existing CSV
    csv_file = gap_file.with_suffix('.csv')
    if not csv_file.exists():
        logger.error(f"  CSV file not found: {csv_file}")
        return False

    existing_df = pd.read_csv(csv_file)
    logger.info(f"  Existing data: {len(existing_df):,} records")

    # Download missing chunks
    new_dfs = []
    still_failed = []

    for chunk in failed_chunks:
        # Parse chunk time range
        if ':' in chunk and '-' in chunk:
            # Format: HH:MM-HH:MM or HH:MM
            parts = chunk.split('-') if '-' in chunk else [chunk, chunk]
            start_time = parts[0].strip()
            end_time = parts[1].strip() if len(parts) > 1 else start_time

            # Handle hourly chunks (HH:00-HH:59) vs 5-min chunks (HH:MM)
            if end_time.endswith(':59'):
                # Hourly chunk
                start_time_str = f"{date_str} {start_time}"
                end_time_str = f"{date_str} {end_time}"
            else:
                # 5-min chunk - add range
                start_hour, start_min = start_time.split(':')
                end_min = str(int(start_min) + 9).zfill(2)
                if int(end_min) >= 60:
                    end_min = '59'
                end_time_str = f"{date_str} {start_hour}:{end_min}"
                start_time_str = f"{date_str} {start_time}"
        else:
            logger.warning(f"  Skipping malformed chunk: {chunk}")
            still_failed.append(chunk)
            continue

        df = download_chunk_with_retry(
            client,
            config['api_method'],
            start_time_str,
            end_time_str,
            chunk
        )

        if df is not None:
            new_dfs.append(df)
        else:
            still_failed.append(chunk)

    # Merge new data with existing
    if new_dfs:
        all_df = pd.concat([existing_df] + new_dfs, ignore_index=True)

        # Deduplicate
        if 'datetime_beginning_ept' in all_df.columns and 'pnode_id' in all_df.columns:
            all_df = all_df.drop_duplicates(subset=['datetime_beginning_ept', 'pnode_id'], keep='last')

        # Save updated CSV
        all_df.to_csv(csv_file, index=False)
        logger.info(f"  ✓ Updated {csv_file.name}: {len(all_df):,} records ({len(new_dfs)} new chunks)")

    # Update or remove gap metadata
    if still_failed:
        metadata['failed_chunks'] = still_failed
        metadata['last_fill_attempt'] = datetime.now().isoformat()
        with open(gap_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.warning(f"  ⚠️  {len(still_failed)} chunks still missing")
        return False
    else:
        gap_file.unlink()
        logger.info(f"  ✓ All gaps filled! Removed {gap_file.name}")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Detect and fill gaps in PJM data"
    )
    parser.add_argument(
        "--data-type",
        type=str,
        choices=list(DATA_TYPES.keys()),
        help="Specific data type to check (default: all)"
    )
    parser.add_argument(
        "--detect-only",
        action="store_true",
        help="Only detect gaps, don't fill them"
    )

    args = parser.parse_args()

    # Determine data types to check
    data_types = [args.data_type] if args.data_type else list(DATA_TYPES.keys())

    logger.info(f"\n{'='*80}")
    logger.info(f"PJM Data Gap Detection {'and Filling' if not args.detect_only else ''}")
    logger.info(f"{'='*80}\n")

    # Scan for gaps
    all_gaps = {}
    total_gap_count = 0

    for data_type in data_types:
        gap_files = find_gap_files(data_type)
        if gap_files:
            all_gaps[data_type] = gap_files
            total_gap_count += len(gap_files)
            logger.info(f"{DATA_TYPES[data_type]['description']:35} {len(gap_files)} files with gaps")
            for gap_file in gap_files[:5]:  # Show first 5
                with open(gap_file, 'r') as f:
                    metadata = json.load(f)
                logger.info(f"  - {metadata['date']}: {len(metadata['failed_chunks'])} chunks missing")
            if len(gap_files) > 5:
                logger.info(f"  ... and {len(gap_files) - 5} more")

    if total_gap_count == 0:
        logger.info("✓ No gaps found! All data is complete.")
        return 0

    logger.info(f"\nTotal: {total_gap_count} files with gaps")

    if args.detect_only:
        return 0

    # Fill gaps
    logger.info(f"\n{'='*80}")
    logger.info(f"Starting Gap Filling...")
    logger.info(f"{'='*80}")

    try:
        client = PJMAPIClient(requests_per_minute=6)
    except ValueError as e:
        logger.error(f"Failed to initialize API client: {e}")
        return 1

    filled_count = 0
    partial_count = 0
    failed_count = 0

    for data_type, gap_files in all_gaps.items():
        for gap_file in gap_files:
            try:
                success = fill_gaps_for_file(client, gap_file, data_type)
                if success:
                    filled_count += 1
                else:
                    partial_count += 1
            except Exception as e:
                logger.error(f"Error filling {gap_file}: {e}")
                failed_count += 1

    # Summary
    logger.info(f"\n{'='*80}")
    logger.info(f"Gap Filling Summary")
    logger.info(f"{'='*80}")
    logger.info(f"  Fully filled: {filled_count} files")
    logger.info(f"  Partially filled: {partial_count} files")
    logger.info(f"  Failed: {failed_count} files")

    return 0


if __name__ == "__main__":
    sys.exit(main())

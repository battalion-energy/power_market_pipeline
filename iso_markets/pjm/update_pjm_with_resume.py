#!/usr/bin/env python3
"""
PJM Data Updater with Auto-Resume Capability

Smart updater that:
1. Finds the last downloaded date for each data type
2. Automatically resumes from that point
3. Handles gaps if cron job fails for a few days
4. Perfect for daily cron jobs
5. Auto-throttles if API rate limits are hit

Usage:
    python update_pjm_with_resume.py                       # Auto-resume all data types
    python update_pjm_with_resume.py --dry-run             # Check what would be updated
    python update_pjm_with_resume.py --start-date 2024-01-01  # Force start date
    python update_pjm_with_resume.py --data-types da_nodal rt_hourly_nodal  # Specific types
"""

import os
import sys
import argparse
import logging
import re
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from pjm_api_client import PJMAPIClient

# Import download functions
import importlib.util
import types

def load_module_from_file(module_name: str, file_path: Path) -> types.ModuleType:
    """Load a module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load the download modules
da_module = load_module_from_file("da_lmps", Path(__file__).parent / "download_historical_nodal_da_lmps.py")
rt_module = load_module_from_file("rt_lmps", Path(__file__).parent / "download_historical_nodal_rt_lmps.py")

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Data directory
PJM_DATA_DIR = Path(os.getenv('PJM_DATA_DIR', '/pool/ssd8tb/data/iso/PJM_data'))

# Retry configuration
MAX_RETRIES = 10  # Increased from 5 to 10 for better resilience
BASE_RETRY_DELAY = 30  # Start with 30 seconds for general errors
RATE_LIMIT_BASE_DELAY = 10  # Start with 10 seconds for 429 errors (6 req/min = ~10s interval)
MAX_BACKOFF = 120  # Cap exponential backoff at 2 minutes (was 5 min)

# Data types and their directory structures
DATA_TYPES = {
    "da_nodal": {
        "dir": PJM_DATA_DIR / "csv_files/da_nodal",
        "pattern": "nodal_da_lmp_*.csv",
        "description": "Day-Ahead Nodal LMPs",
        "priority": 1
    },
    "rt_hourly_nodal": {
        "dir": PJM_DATA_DIR / "csv_files/rt_hourly_nodal",
        "pattern": "nodal_rt_hourly_lmp_*.csv",
        "description": "Real-Time Hourly Nodal LMPs",
        "priority": 2
    },
    "ancillary_services": {
        "dir": PJM_DATA_DIR / "csv_files/da_ancillary_services",
        "pattern": "ancillary_services_*.csv",
        "description": "DA Ancillary Services",
        "priority": 3
    },
    "rt_5min_nodal": {
        "dir": PJM_DATA_DIR / "csv_files/rt_5min_nodal",
        "pattern": "nodal_rt_5min_lmp_*.csv",
        "description": "Real-Time 5-Min Nodal LMPs (last 6 months)",
        "priority": 4,
        "retention_days": 186  # PJM only keeps 6 months
    }
}


def find_last_date(data_type: str) -> Optional[datetime]:
    """Find the most recent date downloaded for a data type."""
    config = DATA_TYPES[data_type]
    data_dir = config["dir"]

    if not data_dir.exists():
        logger.warning(f"{data_type}: Directory not found - {data_dir}")
        return None

    # Get all CSV files
    csv_files = list(data_dir.glob(config["pattern"]))

    if not csv_files:
        logger.warning(f"{data_type}: No existing files found")
        return None

    # Extract dates from filenames (YYYY-MM-DD pattern)
    dates = []
    for csv_file in csv_files:
        filename = csv_file.stem
        # Try to find YYYY-MM-DD pattern in filename
        date_match = re.search(r'(\d{4})-(\d{2})-(\d{2})', filename)
        if date_match:
            try:
                date_str = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"
                date = datetime.strptime(date_str, '%Y-%m-%d')
                dates.append(date)
            except ValueError:
                continue

    if not dates:
        logger.warning(f"{data_type}: No valid dates found in filenames")
        return None

    latest = max(dates)
    logger.info(f"{data_type}: Latest date = {latest.date()} ({len(dates)} files total)")
    return latest


def get_resume_dates(data_types: List[str], fallback_date: datetime) -> Dict[str, datetime]:
    """Get resume dates for each data type."""
    resume_dates = {}

    for data_type in data_types:
        last_date = find_last_date(data_type)

        if last_date:
            # Resume from next day after last downloaded
            resume_date = last_date + timedelta(days=1)
            resume_dates[data_type] = resume_date
        else:
            # No existing data, use fallback
            resume_dates[data_type] = fallback_date

    return resume_dates


def update_da_nodal(client: PJMAPIClient, start_date: datetime, end_date: datetime):
    """Update day-ahead nodal LMP data. ALWAYS saves partial data to avoid gaps."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Updating DA Nodal LMPs: {start_date.date()} to {end_date.date()}")
    logger.info(f"{'='*80}")

    output_dir = DATA_TYPES["da_nodal"]["dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    current_date = start_date
    success_count = 0
    fail_count = 0

    while current_date <= end_date:
        output_file = output_dir / f"nodal_da_lmp_{current_date.strftime('%Y-%m-%d')}.csv"

        # Skip if already exists
        if output_file.exists():
            file_size_mb = output_file.stat().st_size / 1024 / 1024
            if file_size_mb > 10:  # Valid file
                current_date += timedelta(days=1)
                continue

        # Download the day
        try:
            date_str = current_date.strftime('%Y-%m-%d')
            logger.info(f"Downloading DA nodal {date_str}")

            all_dfs = []
            failed_chunks = []

            # Download in 2-hour chunks
            for hour_start in range(0, 24, 2):
                hour_end = min(hour_start + 1, 23)
                start_time_str = f"{date_str} {hour_start:02d}:00"
                end_time_str = f"{date_str} {hour_end:02d}:59"

                # Retry logic with exponential backoff
                chunk_success = False
                for attempt in range(MAX_RETRIES):
                    try:
                        data = client.get_day_ahead_lmps(
                            start_date=start_time_str,
                            end_date=end_time_str,
                            pnode_id=None,  # All nodes
                            use_exact_times=True
                        )

                        if data:
                            items = data.get('items', data.get('data', [data])) if isinstance(data, dict) else data
                            if items:
                                df = pd.DataFrame(items)
                                all_dfs.append(df)
                                chunk_success = True
                                # Rate limiter in API client handles delays
                                break
                        else:
                            break  # No data available, not a retry-able error

                    except Exception as e:
                        error_msg = str(e)
                        is_rate_limit = '429' in error_msg or 'rate limit' in error_msg.lower()

                        if attempt < MAX_RETRIES - 1:
                            # Use backoff for rate limit errors (429)
                            if is_rate_limit:
                                # API client now provides smart retry_after based on response times or Retry-After header
                                retry_after = getattr(e, 'retry_after', None)
                                if retry_after:
                                    wait_time = retry_after + 2  # Small buffer
                                    logger.warning(f"  ⚠️  Rate limit (429) on attempt {attempt + 1}/{MAX_RETRIES} for {hour_start:02d}:00 - waiting {wait_time:.1f}s")
                                else:
                                    # Fallback: exponential backoff with cap (10s, 20s, 40s, 80s, 120s max)
                                    wait_time = min(RATE_LIMIT_BASE_DELAY * (2 ** attempt), MAX_BACKOFF)
                                    logger.warning(f"  ⚠️  Rate limit (429) on attempt {attempt + 1}/{MAX_RETRIES} for {hour_start:02d}:00")
                            else:
                                wait_time = BASE_RETRY_DELAY * (2 ** attempt)  # 30s, 60s, 120s, 240s...
                                logger.warning(f"  ⚠️  Attempt {attempt + 1}/{MAX_RETRIES} failed for {hour_start:02d}:00: {error_msg}")
                            logger.warning(f"  ⏳ Waiting {wait_time} seconds before retry...")
                            time.sleep(wait_time)
                        else:
                            logger.error(f"  ✗ All {MAX_RETRIES} attempts failed for {hour_start:02d}:00: {error_msg}")
                            break

                if not chunk_success:
                    failed_chunks.append(f"{hour_start:02d}:00-{hour_end:02d}:59")

            # ALWAYS save data if we have ANY successful chunks (don't discard partial data)
            if all_dfs:
                df_combined = pd.concat(all_dfs, ignore_index=True)
                df_combined = df_combined.drop_duplicates(subset=['datetime_beginning_ept', 'pnode_id'], keep='last')
                df_combined.to_csv(output_file, index=False)

                if failed_chunks:
                    # Save metadata about failed chunks for gap-filling later
                    metadata_file = output_file.with_suffix('.gaps.json')
                    metadata = {
                        'date': date_str,
                        'data_type': 'da_nodal',
                        'failed_chunks': failed_chunks,
                        'timestamp': datetime.now().isoformat()
                    }
                    with open(metadata_file, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    logger.warning(f"  ⚠️  PARTIAL data for {date_str}: {len(df_combined)} records, {len(failed_chunks)} chunks missing (saved to {metadata_file.name})")
                    success_count += 1  # Count as success since we saved data
                else:
                    logger.info(f"  ✓ {date_str}: {len(df_combined)} records (complete)")
                    success_count += 1
            else:
                logger.error(f"  ✗ NO data retrieved for {date_str}")
                fail_count += 1

        except Exception as e:
            logger.error(f"  Error: {e}")
            fail_count += 1

        current_date += timedelta(days=1)

    logger.info(f"DA Nodal: {success_count} days updated, {fail_count} failed")


def update_rt_hourly_nodal(client: PJMAPIClient, start_date: datetime, end_date: datetime):
    """Update real-time hourly nodal LMP data. ALWAYS saves partial data to avoid gaps."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Updating RT Hourly Nodal LMPs: {start_date.date()} to {end_date.date()}")
    logger.info(f"{'='*80}")

    output_dir = DATA_TYPES["rt_hourly_nodal"]["dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    current_date = start_date
    success_count = 0
    fail_count = 0

    while current_date <= end_date:
        output_file = output_dir / f"nodal_rt_hourly_lmp_{current_date.strftime('%Y-%m-%d')}.csv"

        # Skip if already exists
        if output_file.exists():
            file_size_mb = output_file.stat().st_size / 1024 / 1024
            if file_size_mb > 25:  # Valid file
                current_date += timedelta(days=1)
                continue

        # Download the day
        try:
            date_str = current_date.strftime('%Y-%m-%d')
            logger.info(f"Downloading RT hourly nodal {date_str}")

            all_dfs = []
            failed_chunks = []

            # Download in 2-hour chunks
            for hour_start in range(0, 24, 2):
                hour_end = min(hour_start + 1, 23)
                start_time_str = f"{date_str} {hour_start:02d}:00"
                end_time_str = f"{date_str} {hour_end:02d}:59"

                # Retry logic with exponential backoff
                chunk_success = False
                for attempt in range(MAX_RETRIES):
                    try:
                        data = client.get_rt_hourly_lmps(
                            start_date=start_time_str,
                            end_date=end_time_str,
                            pnode_id=None,  # All nodes
                            use_exact_times=True
                        )

                        if data:
                            items = data.get('items', data.get('data', [data])) if isinstance(data, dict) else data
                            if items:
                                df = pd.DataFrame(items)
                                all_dfs.append(df)
                                chunk_success = True
                                # Rate limiter in API client handles delays
                                break
                        else:
                            break  # No data available

                    except Exception as e:
                        error_msg = str(e)
                        is_rate_limit = '429' in error_msg or 'rate limit' in error_msg.lower()

                        if attempt < MAX_RETRIES - 1:
                            # Use backoff for rate limit errors (429)
                            if is_rate_limit:
                                # API client now provides smart retry_after based on response times or Retry-After header
                                retry_after = getattr(e, 'retry_after', None)
                                if retry_after:
                                    wait_time = retry_after + 2  # Small buffer
                                    logger.warning(f"  ⚠️  Rate limit (429) on attempt {attempt + 1}/{MAX_RETRIES} for {hour_start:02d}:00 - waiting {wait_time:.1f}s")
                                else:
                                    # Fallback: exponential backoff with cap (10s, 20s, 40s, 80s, 120s max)
                                    wait_time = min(RATE_LIMIT_BASE_DELAY * (2 ** attempt), MAX_BACKOFF)
                                    logger.warning(f"  ⚠️  Rate limit (429) on attempt {attempt + 1}/{MAX_RETRIES} for {hour_start:02d}:00")
                            else:
                                wait_time = BASE_RETRY_DELAY * (2 ** attempt)  # 30s, 60s, 120s, 240s...
                                logger.warning(f"  ⚠️  Attempt {attempt + 1}/{MAX_RETRIES} failed for {hour_start:02d}:00: {error_msg}")
                            logger.warning(f"  ⏳ Waiting {wait_time} seconds before retry...")
                            time.sleep(wait_time)
                        else:
                            logger.error(f"  ✗ All {MAX_RETRIES} attempts failed for {hour_start:02d}:00: {error_msg}")
                            break

                if not chunk_success:
                    failed_chunks.append(f"{hour_start:02d}:00-{hour_end:02d}:59")

            # ALWAYS save data if we have ANY successful chunks (don't discard partial data)
            if all_dfs:
                df_combined = pd.concat(all_dfs, ignore_index=True)
                df_combined = df_combined.drop_duplicates(subset=['datetime_beginning_ept', 'pnode_id'], keep='last')
                df_combined.to_csv(output_file, index=False)

                if failed_chunks:
                    # Save metadata about failed chunks for gap-filling later
                    metadata_file = output_file.with_suffix('.gaps.json')
                    metadata = {
                        'date': date_str,
                        'data_type': 'rt_hourly_nodal',
                        'failed_chunks': failed_chunks,
                        'timestamp': datetime.now().isoformat()
                    }
                    with open(metadata_file, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    logger.warning(f"  ⚠️  PARTIAL data for {date_str}: {len(df_combined)} records, {len(failed_chunks)} chunks missing (saved to {metadata_file.name})")
                    success_count += 1  # Count as success since we saved data
                else:
                    logger.info(f"  ✓ {date_str}: {len(df_combined)} records (complete)")
                    success_count += 1
            else:
                logger.error(f"  ✗ NO data retrieved for {date_str}")
                fail_count += 1

        except Exception as e:
            logger.error(f"  Error: {e}")
            fail_count += 1

        current_date += timedelta(days=1)

    logger.info(f"RT Hourly Nodal: {success_count} days updated, {fail_count} failed")


def update_ancillary_services(client: PJMAPIClient, start_date: datetime, end_date: datetime):
    """Update ancillary services data."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Updating Ancillary Services: {start_date.date()} to {end_date.date()}")
    logger.info(f"{'='*80}")

    output_dir = DATA_TYPES["ancillary_services"]["dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    current_date = start_date
    success_count = 0
    fail_count = 0

    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        output_file = output_dir / f"ancillary_services_{date_str}.csv"

        # Skip if already exists
        if output_file.exists():
            file_size_kb = output_file.stat().st_size / 1024
            if file_size_kb > 1:  # Valid file
                current_date += timedelta(days=1)
                continue

        # Download the day
        try:
            data = client.get_ancillary_services(
                start_date=date_str,
                end_date=date_str
            )

            if data:
                items = data.get('items', data.get('data', [data])) if isinstance(data, dict) else data
                if items:
                    df = pd.DataFrame(items)
                    df.to_csv(output_file, index=False)
                    logger.info(f"  ✓ {date_str}: {len(df)} records")
                    success_count += 1
                else:
                    fail_count += 1
            else:
                fail_count += 1

        except Exception as e:
            logger.error(f"  ✗ {date_str}: {e}")
            fail_count += 1

        current_date += timedelta(days=1)

    logger.info(f"Ancillary Services: {success_count} days updated, {fail_count} failed")


def update_rt_5min_nodal(client: PJMAPIClient, start_date: datetime, end_date: datetime):
    """
    Update real-time 5-minute nodal LMP data. ALWAYS saves partial data to avoid gaps.

    Note: PJM only retains 5-minute data for ~6 months (186 days).
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Updating RT 5-Min Nodal LMPs: {start_date.date()} to {end_date.date()}")
    logger.info(f"⚠️  Note: PJM only keeps 5-min data for ~6 months")
    logger.info(f"{'='*80}")

    # Check if start_date is within retention window
    max_days_back = DATA_TYPES["rt_5min_nodal"].get("retention_days", 186)
    earliest_available = end_date - timedelta(days=max_days_back)

    if start_date < earliest_available:
        logger.warning(f"  Adjusting start date from {start_date.date()} to {earliest_available.date()} (retention limit)")
        start_date = earliest_available

    output_dir = DATA_TYPES["rt_5min_nodal"]["dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    current_date = start_date
    success_count = 0
    fail_count = 0

    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        output_file = output_dir / f"nodal_rt_5min_lmp_{date_str}.csv"

        # Skip if already exists (> 100 MB = complete file)
        if output_file.exists():
            file_size_mb = output_file.stat().st_size / 1024 / 1024
            if file_size_mb > 100:
                current_date += timedelta(days=1)
                continue

        # Download the day using 10-minute chunks (144 calls per day)
        try:
            logger.info(f"Downloading RT 5-min nodal {date_str}")
            all_dfs = []
            failed_chunks = []

            # Download in 10-minute chunks to stay under 50K row limit
            for minute_start in range(0, 24 * 60, 10):
                hour_start = minute_start // 60
                min_start = minute_start % 60
                minute_end = minute_start + 9
                hour_end = minute_end // 60
                min_end = minute_end % 60

                start_time_str = f"{date_str} {hour_start:02d}:{min_start:02d}"
                end_time_str = f"{date_str} {hour_end:02d}:{min_end:02d}"

                # Retry logic with exponential backoff
                chunk_success = False
                for attempt in range(MAX_RETRIES):
                    try:
                        data = client.get_rt_fivemin_lmps(
                            start_date=start_time_str,
                            end_date=end_time_str,
                            pnode_id=None,  # All nodes
                            use_exact_times=True
                        )

                        if data:
                            items = data.get('items', data.get('data', [data])) if isinstance(data, dict) else data
                            if items:
                                df = pd.DataFrame(items)
                                all_dfs.append(df)
                                chunk_success = True
                                # Rate limiter in API client handles delays
                                break
                        else:
                            break  # No data available

                    except Exception as e:
                        error_msg = str(e)
                        is_rate_limit = '429' in error_msg or 'rate limit' in error_msg.lower()

                        if attempt < MAX_RETRIES - 1:
                            # Use backoff for rate limit errors (429)
                            if is_rate_limit:
                                # Check if API provided Retry-After hint
                                retry_after = getattr(e, 'retry_after', None)
                                if retry_after:
                                    wait_time = retry_after + 5  # Add 5 second buffer
                                    logger.warning(f"  ⚠️  Rate limit (429) on attempt {attempt + 1}/{MAX_RETRIES} for {hour_start:02d}:{min_start:02d} - API says retry after {retry_after}s")
                                else:
                                    # Exponential backoff with cap
                                    wait_time = min(RATE_LIMIT_BASE_DELAY * (2 ** attempt), MAX_BACKOFF)  # 45s, 90s, 180s, capped at 300s
                                    logger.warning(f"  ⚠️  Rate limit (429) on attempt {attempt + 1}/{MAX_RETRIES} for {hour_start:02d}:{min_start:02d}")
                            else:
                                wait_time = BASE_RETRY_DELAY * (2 ** attempt)  # 30s, 60s, 120s, 240s...
                                logger.warning(f"  ⚠️  Attempt {attempt + 1}/{MAX_RETRIES} failed for {hour_start:02d}:{min_start:02d}: {error_msg}")
                            logger.warning(f"  ⏳ Waiting {wait_time} seconds before retry...")
                            time.sleep(wait_time)
                        else:
                            logger.error(f"  ✗ All {MAX_RETRIES} attempts failed for {hour_start:02d}:{min_start:02d}: {error_msg}")
                            break

                if not chunk_success:
                    failed_chunks.append(f"{hour_start:02d}:{min_start:02d}")

            # ALWAYS save data if we have ANY successful chunks (don't discard partial data)
            if all_dfs:
                df_combined = pd.concat(all_dfs, ignore_index=True)
                # Deduplicate
                datetime_col = next((col for col in df_combined.columns if 'datetime_beginning' in col.lower()), None)
                if datetime_col and 'pnode_id' in df_combined.columns:
                    df_combined = df_combined.drop_duplicates(subset=[datetime_col, 'pnode_id'], keep='last')
                df_combined.to_csv(output_file, index=False)

                if failed_chunks:
                    # Save metadata about failed chunks for gap-filling later
                    metadata_file = output_file.with_suffix('.gaps.json')
                    metadata = {
                        'date': date_str,
                        'data_type': 'rt_5min_nodal',
                        'failed_chunks': failed_chunks,
                        'timestamp': datetime.now().isoformat()
                    }
                    with open(metadata_file, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    logger.warning(f"  ⚠️  PARTIAL data for {date_str}: {len(df_combined):,} records, {len(failed_chunks)} chunks missing (saved to {metadata_file.name})")
                    success_count += 1  # Count as success since we saved data
                else:
                    logger.info(f"  ✓ {date_str}: {len(df_combined):,} records (complete)")
                    success_count += 1
            else:
                logger.error(f"  ✗ NO data retrieved for {date_str}")
                fail_count += 1

        except Exception as e:
            logger.error(f"  Error: {e}")
            fail_count += 1

        current_date += timedelta(days=1)

    logger.info(f"RT 5-Min Nodal: {success_count} days updated, {fail_count} failed")


def main():
    parser = argparse.ArgumentParser(
        description="PJM data updater with auto-resume capability"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Force start date (YYYY-MM-DD), otherwise auto-resume from last date"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date (YYYY-MM-DD), defaults to today"
    )
    parser.add_argument(
        "--data-types",
        nargs="+",
        choices=list(DATA_TYPES.keys()),
        help="Specific data types to update (default: all except rt_5min_nodal)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without downloading"
    )
    parser.add_argument(
        "--fallback-days",
        type=int,
        default=7,
        help="If no existing data, download last N days (default: 7)"
    )

    args = parser.parse_args()

    # Determine data types to update (default: include RT 5-min with 6-month window)
    if args.data_types:
        data_types = args.data_types
    else:
        data_types = ["da_nodal", "rt_hourly_nodal", "ancillary_services", "rt_5min_nodal"]

    # Parse end date
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d") if args.end_date else datetime.now()

    # Get resume dates
    logger.info(f"\n{'='*80}")
    logger.info(f"PJM Data Updater with Auto-Resume")
    logger.info(f"{'='*80}")
    logger.info(f"Checking existing data to determine resume points...\n")

    fallback_date = end_date - timedelta(days=args.fallback_days)

    if args.start_date:
        # Force start date for all types
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        resume_dates = {dt: start_date for dt in data_types}
        logger.info(f"Using forced start date: {start_date.date()}")
    else:
        # Auto-resume
        resume_dates = get_resume_dates(data_types, fallback_date)

    # Show resume plan
    logger.info(f"\n{'='*80}")
    logger.info(f"Resume Plan:")
    logger.info(f"{'='*80}")
    for data_type in sorted(data_types, key=lambda x: DATA_TYPES[x]["priority"]):
        start = resume_dates[data_type]
        days = (end_date - start).days + 1
        logger.info(f"{DATA_TYPES[data_type]['description']:35} {start.date()} -> {end_date.date()} ({days} days)")

    if args.dry_run:
        logger.info(f"\n[DRY RUN] Would update {len(data_types)} data types")
        return 0

    # Initialize API client (optimized settings)
    try:
        client = PJMAPIClient(requests_per_minute=6, min_delay_between_requests=2.0)
    except ValueError as e:
        logger.error(f"Failed to initialize API client: {e}")
        logger.error("Please set PJM_API_KEY in your .env file")
        return 1

    # Update each data type
    logger.info(f"\n{'='*80}")
    logger.info(f"Starting Updates...")
    logger.info(f"{'='*80}\n")

    success_count = 0

    try:
        # Update in priority order
        for data_type in sorted(data_types, key=lambda x: DATA_TYPES[x]["priority"]):
            if data_type == "da_nodal":
                update_da_nodal(client, resume_dates[data_type], end_date)
                success_count += 1
            elif data_type == "rt_hourly_nodal":
                update_rt_hourly_nodal(client, resume_dates[data_type], end_date)
                success_count += 1
            elif data_type == "ancillary_services":
                update_ancillary_services(client, resume_dates[data_type], end_date)
                success_count += 1
            elif data_type == "rt_5min_nodal":
                update_rt_5min_nodal(client, resume_dates[data_type], end_date)
                success_count += 1

    except Exception as e:
        logger.error(f"Update failed: {e}", exc_info=True)
        return 1

    # Summary
    logger.info(f"\n{'='*80}")
    logger.info(f"Update Summary: {success_count}/{len(data_types)} data types updated successfully")
    logger.info(f"{'='*80}")

    return 0 if success_count == len(data_types) else 1


if __name__ == "__main__":
    sys.exit(main())

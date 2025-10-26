#!/usr/bin/env python3
"""
Download Recent PJM RT 5-Minute Nodal LMP Data

IMPORTANT: PJM only retains RT 5-minute data for ~6 months (186 days).
This script downloads the most recent 6 months of 5-minute data.

For historical analysis beyond 6 months, use RT hourly data instead.

Usage:
    python download_rt_5min_recent.py                    # Last 180 days
    python download_rt_5min_recent.py --days-back 90     # Last 90 days
    python download_rt_5min_recent.py --start-date 2025-09-01 --end-date 2025-10-25
    python download_rt_5min_recent.py --quick-skip       # Skip existing files
"""

import os
import sys
import argparse
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from pjm_api_client import PJMAPIClient

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Data directory
PJM_DATA_DIR = Path(os.getenv('PJM_DATA_DIR', '/home/enrico/data/PJM_data'))
OUTPUT_DIR = PJM_DATA_DIR / 'csv_files' / 'rt_5min_nodal'

# Retry configuration
MAX_RETRIES = 5
RETRY_DELAYS = [30, 60, 120, 240, 480]  # seconds

# PJM's data retention limit
MAX_DAYS_BACK = 186  # ~6 months


def download_chunk_with_retry(client: PJMAPIClient, start_time_str: str, end_time_str: str,
                               chunk_label: str) -> pd.DataFrame:
    """Download a single time chunk with retry logic for 429 errors."""
    for attempt in range(MAX_RETRIES):
        try:
            data = client.get_rt_fivemin_lmps(
                start_date=start_time_str,
                end_date=end_time_str,
                pnode_id=None,  # All nodes
                use_exact_times=True
            )

            if not data:
                logger.warning(f"  No data for {chunk_label}")
                return None

            # Extract items
            if isinstance(data, dict):
                items = data.get('items', data.get('data', [data]))
            else:
                items = data

            if not items:
                logger.warning(f"  No items for {chunk_label}")
                return None

            df = pd.DataFrame(items)
            logger.info(f"  ✓ {chunk_label}: {len(df)} rows, {df['pnode_id'].nunique() if 'pnode_id' in df.columns else 0} nodes")
            return df

        except Exception as e:
            error_msg = str(e)

            # Check for rate limiting (429 errors)
            if '429' in error_msg or 'too many' in error_msg.lower():
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAYS[attempt]
                    logger.warning(f"  ⚠️  Attempt {attempt + 1}/{MAX_RETRIES} failed for {chunk_label}: {error_msg}")
                    logger.warning(f"  ⏳ Waiting {delay} seconds before retry...")
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"  ✗ Max retries exceeded for {chunk_label}")
                    return None
            else:
                logger.error(f"  ✗ Error for {chunk_label}: {e}")
                return None

    return None


def download_rt_5min_day(client: PJMAPIClient, date: datetime, output_dir: Path) -> bool:
    """
    Download ALL nodal RT 5-minute LMP data for a single day.
    Makes 144 API calls (10-minute chunks) to stay under 50K row limit.
    """
    date_str = date.strftime('%Y-%m-%d')
    logger.info(f"Downloading RT 5-min {date_str}")

    all_dfs = []
    failed_chunks = []

    # Download in 10-minute chunks to stay under 50K row limit
    # 22,528 nodes × 2 intervals (5-min each) = ~45K rows (safe margin)
    for minute_start in range(0, 24 * 60, 10):
        hour_start = minute_start // 60
        min_start = minute_start % 60

        minute_end = minute_start + 9
        hour_end = minute_end // 60
        min_end = minute_end % 60

        start_time_str = f"{date_str} {hour_start:02d}:{min_start:02d}"
        end_time_str = f"{date_str} {hour_end:02d}:{min_end:02d}"
        chunk_label = f"{hour_start:02d}:{min_start:02d}-{hour_end:02d}:{min_end:02d}"

        df = download_chunk_with_retry(client, start_time_str, end_time_str, chunk_label)

        if df is not None:
            all_dfs.append(df)
        else:
            failed_chunks.append(chunk_label)

    # Check if we have any failed chunks
    if failed_chunks:
        logger.error(f"  ⚠️  NOT saving incomplete data for {date_str} ({len(failed_chunks)} failed chunks)")
        return False

    # Combine, deduplicate, and save
    if all_dfs:
        df_combined = pd.concat(all_dfs, ignore_index=True)

        # Deduplicate based on timestamp and pnode_id
        datetime_col = next((col for col in df_combined.columns if 'datetime_beginning' in col.lower()), None)
        if datetime_col and 'pnode_id' in df_combined.columns:
            df_combined = df_combined.drop_duplicates(subset=[datetime_col, 'pnode_id'], keep='last')

        # Save
        filename = f"nodal_rt_5min_lmp_{date_str}.csv"
        output_path = output_dir / filename
        df_combined.to_csv(output_path, index=False)

        # Get file size
        file_size_mb = output_path.stat().st_size / 1024 / 1024

        logger.info(f"✓ Saved {date_str}: {len(df_combined):,} rows, {df_combined['pnode_id'].nunique():,} nodes, {file_size_mb:.1f} MB")
        return True
    else:
        logger.error(f"  ⚠️  No data collected for {date_str}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Download recent PJM RT 5-minute nodal LMP data (max 6 months back)'
    )
    parser.add_argument(
        '--days-back',
        type=int,
        default=180,
        help=f'Number of days to download (max {MAX_DAYS_BACK}, default: 180)'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date (YYYY-MM-DD), overrides --days-back'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date (YYYY-MM-DD), defaults to today'
    )
    parser.add_argument(
        '--quick-skip',
        action='store_true',
        help='Quick skip mode: check file existence only (fast restart)'
    )

    args = parser.parse_args()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Determine date range
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d') if args.end_date else datetime.now()

    if args.start_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    else:
        start_date = end_date - timedelta(days=args.days_back)

    # Check if start_date is within retention window
    max_start_date = end_date - timedelta(days=MAX_DAYS_BACK)
    if start_date < max_start_date:
        logger.warning(f"⚠️  WARNING: Requested start date {start_date.date()} is beyond PJM's {MAX_DAYS_BACK}-day retention window")
        logger.warning(f"   Data may not be available before {max_start_date.date()}")
        logger.warning(f"   Adjusting start date to {max_start_date.date()}")
        start_date = max_start_date

    total_days = (end_date - start_date).days + 1

    logger.info("="*80)
    logger.info("PJM RT 5-Minute Nodal LMP Download (Recent Data Only)")
    logger.info("="*80)
    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    logger.info(f"Total days: {total_days}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"Quick skip mode: {args.quick_skip}")
    logger.info(f"Data retention limit: {MAX_DAYS_BACK} days (~6 months)")
    logger.info("")
    logger.info(f"Estimated:")
    logger.info(f"  API calls: {total_days * 144:,} (144 per day)")
    logger.info(f"  Time: ~{total_days * 240:.0f} minutes (~{total_days * 4:.1f} hours)")
    logger.info(f"  Size: ~{total_days * 140:.0f} MB (~{total_days * 140 / 1024:.1f} GB)")
    logger.info("")

    # Initialize API client
    try:
        client = PJMAPIClient(requests_per_minute=6)
    except ValueError as e:
        logger.error(f"Failed to initialize API client: {e}")
        logger.error("Please set PJM_API_KEY in your .env file")
        return 1

    # Download the data
    current_date = start_date
    success_count = 0
    skip_count = 0
    fail_count = 0

    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        output_file = OUTPUT_DIR / f"nodal_rt_5min_lmp_{date_str}.csv"

        # Check if file already exists
        if output_file.exists():
            if args.quick_skip:
                # Quick mode: just check file size (should be >100MB for full day)
                file_size_mb = output_file.stat().st_size / 1024 / 1024
                if file_size_mb > 100:
                    skip_count += 1
                    current_date += timedelta(days=1)
                    continue

        # Download the day
        if download_rt_5min_day(client, current_date, OUTPUT_DIR):
            success_count += 1
        else:
            fail_count += 1

        current_date += timedelta(days=1)

    logger.info(f"\n{'='*80}")
    logger.info(f"Download Summary:")
    logger.info(f"  Success: {success_count} days")
    logger.info(f"  Skipped: {skip_count} days")
    logger.info(f"  Failed:  {fail_count} days")
    logger.info(f"  Total:   {total_days} days")
    logger.info(f"{'='*80}")

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

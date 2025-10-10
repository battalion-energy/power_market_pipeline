#!/usr/bin/env python3
"""
Download PJM Historical Day-Ahead Nodal LMP Prices (Archived Data: 2019-2023)

For archived data (older than 731 days), PJM API does NOT allow filtering by pnode_id.
Therefore, we must download ALL nodes for historical data.

The API returns max 50,000 rows per request.
With ~22,528 nodes, that's only ~2 hours of data per request.

This script downloads DAY BY DAY, making multiple requests per day to get all 24 hours.

Usage:
    python download_historical_nodal_da_lmps.py --start-date 2019-01-01 --end-date 2023-10-06
    python download_historical_nodal_da_lmps.py --year 2022
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

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))
from pjm_api_client import PJMAPIClient

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

MAX_RETRIES = 5
BASE_RETRY_DELAY = 30  # Start with 30 seconds


def download_chunk_with_retry(client: PJMAPIClient, start_time_str: str, end_time_str: str, hour_start: int):
    """
    Download a single 2-hour chunk with retry logic for 429 errors.

    Args:
        client: PJMAPIClient instance
        start_time_str: Start time string (YYYY-MM-DD HH:mm)
        end_time_str: End time string (YYYY-MM-DD HH:mm)
        hour_start: Starting hour (for logging)

    Returns:
        DataFrame or None if all retries failed
    """
    for attempt in range(MAX_RETRIES):
        try:
            data = client.get_day_ahead_lmps(
                start_date=start_time_str,
                end_date=end_time_str,
                pnode_id=None,  # No filter = all nodes (required for archived data)
                use_exact_times=True  # Use times as-is, don't add 00:00/23:59
            )

            if not data:
                logger.warning(f"  No data for {start_time_str} to {end_time_str}")
                return None

            # Extract items
            if isinstance(data, dict):
                items = data.get('items', data.get('data', [data]))
            else:
                items = data

            if not items:
                return None

            df = pd.DataFrame(items)
            logger.info(f"  ✓ {hour_start:02d}:00-{hour_start+1:02d}:59: {len(df):,} rows, {df['pnode_id'].nunique() if 'pnode_id' in df.columns else 0} nodes")

            # Add small delay between successful requests to be conservative
            time.sleep(2)

            return df

        except Exception as e:
            error_msg = str(e)
            is_rate_limit = '429' in error_msg or 'rate limit' in error_msg.lower()

            if attempt < MAX_RETRIES - 1:
                # Exponential backoff: 30s, 60s, 120s, 240s, 480s
                wait_time = BASE_RETRY_DELAY * (2 ** attempt)
                logger.warning(f"  ⚠️  Attempt {attempt + 1}/{MAX_RETRIES} failed for {start_time_str} to {end_time_str}: {error_msg}")
                logger.warning(f"  ⏳ Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                logger.error(f"  ✗ All {MAX_RETRIES} attempts failed for {start_time_str} to {end_time_str}: {error_msg}")
                return None

    return None


def download_nodal_day(client: PJMAPIClient, date: datetime, output_dir: Path):
    """
    Download ALL nodal DA LMP data for a single day.

    Makes multiple API calls (2-hour chunks) to get full 24 hours.

    Args:
        client: PJMAPIClient instance
        date: Date to download
        output_dir: Output directory
    """
    date_str = date.strftime('%Y-%m-%d')
    logger.info(f"Downloading {date_str}")

    all_dfs = []
    hours_covered = set()
    failed_chunks = []

    # Download in 2-hour chunks to stay under 50K row limit
    # 22,528 nodes × 2 hours = ~45K rows (safe margin)
    for hour_start in range(0, 24, 2):
        hour_end = min(hour_start + 1, 23)  # Actually get 2 hours (e.g., 00 and 01)

        start_time_str = f"{date_str} {hour_start:02d}:00"
        end_time_str = f"{date_str} {hour_end:02d}:59"

        # Use retry logic to download chunk
        df = download_chunk_with_retry(client, start_time_str, end_time_str, hour_start)

        if df is not None:
            all_dfs.append(df)

            # Track hours covered
            if 'datetime_beginning_ept' in df.columns:
                for dt_str in df['datetime_beginning_ept'].unique():
                    try:
                        dt = pd.to_datetime(dt_str)
                        hours_covered.add(dt.hour)
                    except:
                        pass
        else:
            # Track failed chunk
            failed_chunks.append(f"{hour_start:02d}:00-{hour_end:02d}:59")

    # Check if we have any failed chunks
    if failed_chunks:
        logger.error(f"⚠️  {date_str} has {len(failed_chunks)} FAILED chunks: {', '.join(failed_chunks)}")
        logger.error(f"⚠️  NOT saving incomplete data for {date_str} - will retry on next run")
        return False

    if not all_dfs:
        logger.warning(f"No data retrieved for {date_str}")
        return False

    # Combine all chunks
    df_combined = pd.concat(all_dfs, ignore_index=True)

    # Remove duplicates (in case of overlap)
    if 'datetime_beginning_ept' in df_combined.columns and 'pnode_id' in df_combined.columns:
        df_combined = df_combined.drop_duplicates(subset=['datetime_beginning_ept', 'pnode_id'], keep='last')

    # Save
    filename = f"nodal_da_lmp_{date_str}.csv"
    output_path = output_dir / filename
    df_combined.to_csv(output_path, index=False)

    file_size_mb = output_path.stat().st_size / 1024 / 1024
    unique_nodes = df_combined['pnode_id'].nunique() if 'pnode_id' in df_combined.columns else 0

    logger.info(f"✓ Saved {date_str}: {len(df_combined):,} rows, {unique_nodes:,} nodes, {len(hours_covered)} hours covered, {file_size_mb:.1f} MB")

    if len(hours_covered) < 24:
        logger.warning(f"  ⚠️  Only {len(hours_covered)}/24 hours covered: {sorted(hours_covered)}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Download PJM Historical Day-Ahead Nodal LMPs (ALL nodes, archived data)'
    )
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--year', type=int, help='Download specific year')

    args = parser.parse_args()

    # Determine date range
    if args.year:
        start_date = datetime(args.year, 1, 1)
        end_date = datetime(args.year, 12, 31)
    elif args.start_date and args.end_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    else:
        logger.error("Must specify either --year or --start-date and --end-date")
        return

    # Calculate days
    total_days = (end_date - start_date).days + 1
    logger.info(f"Historical Data Download (Archived Data)")
    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    logger.info(f"Total days: {total_days}")
    logger.info(f"Estimated API calls: {total_days * 12} (12 per day for 2-hour chunks)")
    logger.info(f"Estimated time: {total_days * 12 / 6:.1f} minutes ({total_days * 2:.1f} hours)")

    # Setup
    data_dir = Path(os.getenv('PJM_DATA_DIR', '/home/enrico/data/PJM_data'))
    output_dir = data_dir / 'csv_files' / 'da_nodal'
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        client = PJMAPIClient(requests_per_minute=6)
    except ValueError as e:
        logger.error(f"Failed to initialize API client: {e}")
        return

    logger.info("⚠️  WARNING: This will take a LONG time!")
    logger.info("⚠️  Downloading ALL nodes (archived data does not support pnode_id filtering)")
    logger.info("⚠️  Running at lower priority (nice)")
    logger.info("")

    # Download day by day
    current_date = start_date
    success_count = 0
    fail_count = 0
    skip_count = 0

    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        output_file = output_dir / f"nodal_da_lmp_{date_str}.csv"

        # Skip if file already exists and appears complete
        if output_file.exists():
            try:
                df_check = pd.read_csv(output_file)
                datetime_col = next((col for col in df_check.columns if 'datetime_beginning' in col.lower()), None)
                if datetime_col:
                    hours = df_check[datetime_col].apply(lambda x: pd.to_datetime(x).hour).nunique()
                    if hours >= 24:
                        logger.info(f"⏭️  Skipping {date_str} - already complete ({hours} hours)")
                        skip_count += 1
                        current_date += timedelta(days=1)
                        continue
            except:
                pass  # If check fails, try downloading anyway

        try:
            if download_nodal_day(client, current_date, output_dir):
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            logger.error(f"Failed to download {current_date.date()}: {e}")
            fail_count += 1

        current_date += timedelta(days=1)

    logger.info(f"\n✓ Download complete!")
    logger.info(f"  Success: {success_count} days")
    logger.info(f"  Failed: {fail_count} days")
    logger.info(f"  Skipped (already complete): {skip_count} days")
    logger.info(f"  Data saved to: {output_dir}")


if __name__ == "__main__":
    main()

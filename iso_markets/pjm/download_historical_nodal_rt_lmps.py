#!/usr/bin/env python3
"""
Download PJM Historical Real-Time Nodal LMP Prices (ALL nodes, 2019-2025)

For both 5-minute and hourly granularity real-time data.

RT data is MASSIVE:
- 5-minute: ~22,528 nodes × 288 intervals/day = ~6.5M records/day
- Hourly: ~22,528 nodes × 24 hours/day = ~540K records/day

API has 50K row limit, so we download in very small chunks:
- 5-minute: 10-minute chunks (2 intervals × 22,528 nodes = ~45K rows)
- Hourly: 2-hour chunks (2 hours × 22,528 nodes = ~45K rows)

Usage:
    # Download 5-minute data for 2025 (working backwards)
    python download_historical_nodal_rt_lmps.py --granularity 5min --year 2025

    # Download hourly data for 2024
    python download_historical_nodal_rt_lmps.py --granularity hourly --year 2024

    # Download both granularities for 2023
    python download_historical_nodal_rt_lmps.py --granularity both --year 2023

    # Custom date range with quick-skip
    python download_historical_nodal_rt_lmps.py --granularity hourly --start-date 2019-01-01 --end-date 2019-12-31 --quick-skip
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


def download_chunk_with_retry(client: PJMAPIClient, start_time_str: str, end_time_str: str,
                               chunk_label: str, granularity: str = '5min'):
    """
    Download a single time chunk with retry logic for 429 errors.

    Args:
        client: PJMAPIClient instance
        start_time_str: Start time string (YYYY-MM-DD HH:mm)
        end_time_str: End time string (YYYY-MM-DD HH:mm)
        chunk_label: Label for logging (e.g., "00:00-00:09")
        granularity: '5min' or 'hourly'

    Returns:
        DataFrame or None if all retries failed
    """
    for attempt in range(MAX_RETRIES):
        try:
            # Use the appropriate method based on granularity
            if granularity == '5min':
                data = client.get_rt_fivemin_lmps(
                    start_date=start_time_str,
                    end_date=end_time_str,
                    pnode_id=None,  # No filter = all nodes
                    use_exact_times=True
                )
            else:  # hourly
                data = client.get_rt_hourly_lmps(
                    start_date=start_time_str,
                    end_date=end_time_str,
                    pnode_id=None,  # No filter = all nodes
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
                return None

            df = pd.DataFrame(items)
            logger.info(f"  ✓ {chunk_label}: {len(df):,} rows, {df['pnode_id'].nunique() if 'pnode_id' in df.columns else 0} nodes")

            # Add small delay between successful requests
            time.sleep(2)

            return df

        except Exception as e:
            error_msg = str(e)
            is_rate_limit = '429' in error_msg or 'rate limit' in error_msg.lower()

            if attempt < MAX_RETRIES - 1:
                # Exponential backoff: 30s, 60s, 120s, 240s, 480s
                wait_time = BASE_RETRY_DELAY * (2 ** attempt)
                logger.warning(f"  ⚠️  Attempt {attempt + 1}/{MAX_RETRIES} failed for {chunk_label}: {error_msg}")
                logger.warning(f"  ⏳ Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                logger.error(f"  ✗ All {MAX_RETRIES} attempts failed for {chunk_label}: {error_msg}")
                return None

    return None


def download_nodal_rt_day_5min(client: PJMAPIClient, date: datetime, output_dir: Path):
    """
    Download ALL nodal RT 5-minute LMP data for a single day.

    Makes multiple API calls (10-minute chunks = 144 chunks per day) to stay under 50K row limit.

    Args:
        client: PJMAPIClient instance
        date: Date to download
        output_dir: Output directory

    Returns:
        True if successful, False if failed
    """
    date_str = date.strftime('%Y-%m-%d')
    logger.info(f"Downloading RT 5-min {date_str}")

    all_dfs = []
    intervals_covered = set()
    failed_chunks = []

    # Download in 10-minute chunks (2 5-minute intervals)
    # This gives us: 22,528 nodes × 2 intervals = ~45K rows (safe under 50K limit)
    for hour in range(24):
        for minute_start in range(0, 60, 10):
            minute_end = minute_start + 9

            start_time_str = f"{date_str} {hour:02d}:{minute_start:02d}"
            end_time_str = f"{date_str} {hour:02d}:{minute_end:02d}"
            chunk_label = f"{hour:02d}:{minute_start:02d}-{hour:02d}:{minute_end:02d}"

            df = download_chunk_with_retry(client, start_time_str, end_time_str, chunk_label, '5min')

            if df is not None:
                all_dfs.append(df)

                # Track intervals covered
                if 'datetime_beginning_ept' in df.columns:
                    for dt_str in df['datetime_beginning_ept'].unique():
                        try:
                            dt = pd.to_datetime(dt_str)
                            intervals_covered.add(dt.strftime('%H:%M'))
                        except:
                            pass
            else:
                failed_chunks.append(chunk_label)

    # Check if we have any failed chunks
    if failed_chunks:
        logger.error(f"⚠️  {date_str} has {len(failed_chunks)} FAILED chunks: {', '.join(failed_chunks[:10])}{'...' if len(failed_chunks) > 10 else ''}")
        logger.error(f"⚠️  NOT saving incomplete data for {date_str} - will retry on next run")
        return False

    if not all_dfs:
        logger.warning(f"No data retrieved for {date_str}")
        return False

    # Combine all chunks
    df_combined = pd.concat(all_dfs, ignore_index=True)

    # Remove duplicates
    if 'datetime_beginning_ept' in df_combined.columns and 'pnode_id' in df_combined.columns:
        df_combined = df_combined.drop_duplicates(subset=['datetime_beginning_ept', 'pnode_id'], keep='last')

    # Save
    filename = f"nodal_rt_5min_lmp_{date_str}.csv"
    output_path = output_dir / filename
    df_combined.to_csv(output_path, index=False)

    file_size_mb = output_path.stat().st_size / 1024 / 1024
    unique_nodes = df_combined['pnode_id'].nunique() if 'pnode_id' in df_combined.columns else 0

    logger.info(f"✓ Saved {date_str}: {len(df_combined):,} rows, {unique_nodes:,} nodes, {len(intervals_covered)} intervals, {file_size_mb:.1f} MB")

    if len(intervals_covered) < 288:  # 288 5-minute intervals in a day
        logger.warning(f"  ⚠️  Only {len(intervals_covered)}/288 intervals covered")

    return True


def download_nodal_rt_day_hourly(client: PJMAPIClient, date: datetime, output_dir: Path):
    """
    Download ALL nodal RT hourly LMP data for a single day.

    Makes multiple API calls (2-hour chunks = 12 chunks per day) to stay under 50K row limit.

    Args:
        client: PJMAPIClient instance
        date: Date to download
        output_dir: Output directory

    Returns:
        True if successful, False if failed
    """
    date_str = date.strftime('%Y-%m-%d')
    logger.info(f"Downloading RT hourly {date_str}")

    all_dfs = []
    hours_covered = set()
    failed_chunks = []

    # Download in 2-hour chunks to stay under 50K row limit
    # 22,528 nodes × 2 hours = ~45K rows (safe margin)
    for hour_start in range(0, 24, 2):
        hour_end = min(hour_start + 1, 23)  # Get 2 hours

        start_time_str = f"{date_str} {hour_start:02d}:00"
        end_time_str = f"{date_str} {hour_end:02d}:59"
        chunk_label = f"{hour_start:02d}:00-{hour_end:02d}:59"

        df = download_chunk_with_retry(client, start_time_str, end_time_str, chunk_label, 'hourly')

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
            failed_chunks.append(chunk_label)

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

    # Remove duplicates
    if 'datetime_beginning_ept' in df_combined.columns and 'pnode_id' in df_combined.columns:
        df_combined = df_combined.drop_duplicates(subset=['datetime_beginning_ept', 'pnode_id'], keep='last')

    # Save
    filename = f"nodal_rt_hourly_lmp_{date_str}.csv"
    output_path = output_dir / filename
    df_combined.to_csv(output_path, index=False)

    file_size_mb = output_path.stat().st_size / 1024 / 1024
    unique_nodes = df_combined['pnode_id'].nunique() if 'pnode_id' in df_combined.columns else 0

    logger.info(f"✓ Saved {date_str}: {len(df_combined):,} rows, {unique_nodes:,} nodes, {len(hours_covered)} hours, {file_size_mb:.1f} MB")

    if len(hours_covered) < 24:
        logger.warning(f"  ⚠️  Only {len(hours_covered)}/24 hours covered: {sorted(hours_covered)}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Download PJM Historical Real-Time Nodal LMPs (ALL nodes, archived data)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download 5-minute RT data for 2025 (most recent year first)
    python download_historical_nodal_rt_lmps.py --granularity 5min --year 2025

    # Download hourly RT data for 2024
    python download_historical_nodal_rt_lmps.py --granularity hourly --year 2024

    # Download BOTH granularities for 2023
    python download_historical_nodal_rt_lmps.py --granularity both --year 2023

    # Custom date range with quick-skip for fast restart
    python download_historical_nodal_rt_lmps.py --granularity hourly --start-date 2023-01-01 --end-date 2023-12-31 --quick-skip

    # Work backwards from most recent
    python download_historical_nodal_rt_lmps.py --granularity both --start-date 2019-01-01 --end-date 2025-10-06 --reverse

Note: RT data is HUGE. 5-minute data takes ~140 MB/day, hourly takes ~35 MB/day.
      For full 2019-2025 range, that's ~350 GB (5min) or ~90 GB (hourly).
        """
    )

    parser.add_argument('--granularity', type=str, required=True,
                        choices=['5min', 'hourly', 'both'],
                        help='Data granularity: 5min, hourly, or both')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--year', type=int, help='Download specific year')
    parser.add_argument('--reverse', action='store_true',
                        help='Download in reverse (most recent first)')
    parser.add_argument('--quick-skip', action='store_true',
                        help='Quick skip mode: only check file existence (fast)')

    args = parser.parse_args()

    # Determine date range
    if args.year:
        start_date = datetime(args.year, 1, 1)
        end_date = datetime(args.year, 12, 31)
        # Don't go past today
        today = datetime.now()
        if end_date > today:
            end_date = today
    elif args.start_date and args.end_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    else:
        logger.error("Must specify either --year or --start-date and --end-date")
        return

    # Calculate days
    total_days = (end_date - start_date).days + 1

    # Determine which granularities to download
    granularities = []
    if args.granularity in ['5min', 'both']:
        granularities.append('5min')
    if args.granularity in ['hourly', 'both']:
        granularities.append('hourly')

    logger.info(f"Historical RT Nodal Data Download")
    logger.info(f"Granularity: {args.granularity}")
    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    logger.info(f"Total days: {total_days}")
    logger.info(f"Direction: {'Reverse (most recent first)' if args.reverse else 'Forward (oldest first)'}")

    if '5min' in granularities:
        logger.info(f"5-minute: {total_days * 144} API calls (144 per day)")
        logger.info(f"  Estimated time: {total_days * 144 / 6:.1f} minutes ({total_days * 24:.1f} hours)")
        logger.info(f"  Estimated size: {total_days * 140:.1f} MB ({total_days * 140 / 1024:.1f} GB)")

    if 'hourly' in granularities:
        logger.info(f"Hourly: {total_days * 12} API calls (12 per day)")
        logger.info(f"  Estimated time: {total_days * 12 / 6:.1f} minutes ({total_days * 2:.1f} hours)")
        logger.info(f"  Estimated size: {total_days * 35:.1f} MB ({total_days * 35 / 1024:.1f} GB)")

    # Setup
    data_dir = Path(os.getenv('PJM_DATA_DIR', '/home/enrico/data/PJM_data'))

    try:
        client = PJMAPIClient(requests_per_minute=6)
    except ValueError as e:
        logger.error(f"Failed to initialize API client: {e}")
        return

    logger.info("⚠️  WARNING: RT nodal data downloads take DAYS to WEEKS!")
    logger.info("⚠️  Running at lower priority (nice)")
    logger.info("")

    # Process each granularity
    for granularity in granularities:
        if granularity == '5min':
            output_dir = data_dir / 'csv_files' / 'rt_5min_nodal'
            download_func = download_nodal_rt_day_5min
            file_prefix = 'nodal_rt_5min_lmp'
        else:  # hourly
            output_dir = data_dir / 'csv_files' / 'rt_hourly_nodal'
            download_func = download_nodal_rt_day_hourly
            file_prefix = 'nodal_rt_hourly_lmp'

        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\n{'='*70}")
        logger.info(f"Starting {granularity.upper()} download")
        logger.info(f"{'='*70}\n")

        # Generate date list
        if args.reverse:
            # Most recent first (2025 → 2019)
            dates = [end_date - timedelta(days=x) for x in range(total_days)]
        else:
            # Oldest first (2019 → 2025)
            dates = [start_date + timedelta(days=x) for x in range(total_days)]

        success_count = 0
        fail_count = 0
        skip_count = 0

        for current_date in dates:
            date_str = current_date.strftime('%Y-%m-%d')
            output_file = output_dir / f"{file_prefix}_{date_str}.csv"

            # Skip if file already exists
            if output_file.exists():
                if args.quick_skip:
                    # Quick mode: just check file size
                    file_size_mb = output_file.stat().st_size / 1024 / 1024
                    min_size = 100 if granularity == '5min' else 25
                    if file_size_mb > min_size:
                        skip_count += 1
                        continue
                else:
                    # Full verify mode: check CSV completeness (SLOW)
                    try:
                        df_check = pd.read_csv(output_file)
                        datetime_col = next((col for col in df_check.columns if 'datetime_beginning' in col.lower()), None)
                        if datetime_col:
                            if granularity == '5min':
                                intervals = df_check[datetime_col].nunique()
                                if intervals >= 288:  # 288 5-min intervals per day
                                    logger.info(f"⏭️  Skipping {date_str} - already complete ({intervals} intervals)")
                                    skip_count += 1
                                    continue
                            else:  # hourly
                                hours = df_check[datetime_col].apply(lambda x: pd.to_datetime(x).hour).nunique()
                                if hours >= 24:
                                    logger.info(f"⏭️  Skipping {date_str} - already complete ({hours} hours)")
                                    skip_count += 1
                                    continue
                    except:
                        pass  # If check fails, try downloading anyway

            try:
                if download_func(client, current_date, output_dir):
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                logger.error(f"Failed to download {date_str}: {e}")
                fail_count += 1

        logger.info(f"\n{'='*70}")
        logger.info(f"{granularity.upper()} download complete!")
        logger.info(f"{'='*70}")
        logger.info(f"  Success: {success_count} days")
        logger.info(f"  Failed: {fail_count} days")
        logger.info(f"  Skipped (already complete): {skip_count} days")
        logger.info(f"  Data saved to: {output_dir}")
        logger.info("")


if __name__ == "__main__":
    main()

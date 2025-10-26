#!/usr/bin/env python3
"""
Download historical PJM Day-Ahead Ancillary Services data.

This script downloads DA ancillary services prices for all reserve zones back to 2019.
Includes regulation, synchronized reserve, and non-synchronized reserve prices.

Usage:
    python download_historical_ancillary_services.py --year 2025
    python download_historical_ancillary_services.py --start-date 2019-01-01 --end-date 2023-12-31
    python download_historical_ancillary_services.py --year 2024 --quick-skip
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
OUTPUT_DIR = PJM_DATA_DIR / 'csv_files' / 'da_ancillary_services'

# Retry configuration
MAX_RETRIES = 5
RETRY_DELAYS = [30, 60, 120, 240, 480]  # seconds


def download_day_with_retry(client: PJMAPIClient, date: datetime) -> pd.DataFrame:
    """Download ancillary services for a single day with retry logic."""
    date_str = date.strftime('%Y-%m-%d')

    for attempt in range(MAX_RETRIES):
        try:
            data = client.get_ancillary_services(
                start_date=date_str,
                end_date=date_str
            )

            if not data:
                logger.warning(f"  No data returned for {date_str}")
                return None

            # Extract items
            if isinstance(data, dict):
                items = data.get('items', data.get('data', [data]))
            else:
                items = data

            if not items:
                logger.warning(f"  No items for {date_str}")
                return None

            df = pd.DataFrame(items)
            logger.info(f"  ✓ {date_str}: {len(df)} records")
            return df

        except Exception as e:
            error_msg = str(e)

            # Check for rate limiting (429 errors)
            if '429' in error_msg or 'too many' in error_msg.lower():
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAYS[attempt]
                    logger.warning(f"  Rate limit hit for {date_str}, retrying in {delay}s (attempt {attempt + 1}/{MAX_RETRIES})")
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"  ✗ Max retries exceeded for {date_str}")
                    return None
            else:
                logger.error(f"  ✗ Error for {date_str}: {e}")
                return None

    return None


def download_ancillary_services_range(
    client: PJMAPIClient,
    start_date: datetime,
    end_date: datetime,
    quick_skip: bool = False
):
    """Download ancillary services for a date range."""

    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")

    total_days = (end_date - start_date).days + 1
    logger.info(f"Total days: {total_days}")

    current_date = start_date
    success_count = 0
    skip_count = 0
    fail_count = 0

    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        output_file = OUTPUT_DIR / f"ancillary_services_{date_str}.csv"

        # Check if file already exists
        if output_file.exists():
            if quick_skip:
                # Quick mode: just check file size (should be >1KB)
                file_size_kb = output_file.stat().st_size / 1024
                if file_size_kb > 1:
                    skip_count += 1
                    current_date += timedelta(days=1)
                    continue
            else:
                # Check if file has data
                try:
                    df_check = pd.read_csv(output_file)
                    if len(df_check) > 0:
                        logger.info(f"⏭️  Skipping {date_str} - already exists ({len(df_check)} records)")
                        skip_count += 1
                        current_date += timedelta(days=1)
                        continue
                except:
                    pass  # If check fails, try downloading anyway

        # Download the day
        df = download_day_with_retry(client, current_date)

        if df is not None and len(df) > 0:
            # Save to file
            df.to_csv(output_file, index=False)
            success_count += 1
        else:
            logger.warning(f"  ⚠️  No data saved for {date_str}")
            fail_count += 1

        current_date += timedelta(days=1)

    logger.info(f"\n{'='*80}")
    logger.info(f"Download Summary:")
    logger.info(f"  Success: {success_count} days")
    logger.info(f"  Skipped: {skip_count} days")
    logger.info(f"  Failed:  {fail_count} days")
    logger.info(f"  Total:   {total_days} days")
    logger.info(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description='Download historical PJM Day-Ahead Ancillary Services data'
    )
    parser.add_argument(
        '--year',
        type=int,
        help='Year to download (e.g., 2025)'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date (YYYY-MM-DD)'
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
    if args.year:
        start_date = datetime(args.year, 1, 1)
        end_date = datetime(args.year, 12, 31)
        # Don't go past today
        today = datetime.now()
        if end_date > today:
            end_date = today
    elif args.start_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        if args.end_date:
            end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
        else:
            end_date = datetime.now()
    else:
        logger.error("Must specify either --year or --start-date")
        return 1

    logger.info("="*80)
    logger.info("PJM Day-Ahead Ancillary Services Download")
    logger.info("="*80)
    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"Quick skip mode: {args.quick_skip}")
    logger.info("")

    # Initialize API client
    try:
        client = PJMAPIClient(requests_per_minute=6)
    except ValueError as e:
        logger.error(f"Failed to initialize API client: {e}")
        logger.error("Please set PJM_API_KEY in your .env file")
        return 1

    # Download the data
    download_ancillary_services_range(
        client,
        start_date,
        end_date,
        quick_skip=args.quick_skip
    )

    logger.info("\n✓ Download complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

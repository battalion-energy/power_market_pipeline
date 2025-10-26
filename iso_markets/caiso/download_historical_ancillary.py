#!/usr/bin/env python3
"""
Download historical CAISO Ancillary Services pricing data.
Downloads data from 2019-01-01 to present, one file per day.

Ancillary Services:
- RU: Regulation Up
- RD: Regulation Down
- SR: Spinning Reserve
- NR: Non-Spinning Reserve

Storage: One CSV file per day with all AS types
Format: ancillary_services_YYYY-MM-DD.csv
"""

import os
import sys
import logging
import time
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv

from iso_markets.caiso.caiso_api_client import CAISOAPIClient, format_datetime_for_caiso

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
CAISO_DATA_DIR = Path(os.getenv('CAISO_DATA_DIR', '/pool/ssd8tb/data/iso/CAISO_data'))
OUTPUT_DIR = CAISO_DATA_DIR / "csv_files/da_ancillary_services"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Retry configuration
MAX_RETRIES = 5
BASE_RETRY_DELAY = 30  # seconds


def get_latest_downloaded_date() -> datetime:
    """
    Find the latest date that has been successfully downloaded.

    Returns:
        Latest datetime, or None if no files exist
    """
    existing_files = list(OUTPUT_DIR.glob("ancillary_services_*.csv"))

    if not existing_files:
        return None

    # Extract dates from filenames
    dates = []
    for f in existing_files:
        try:
            date_str = f.stem.replace("ancillary_services_", "")
            dates.append(datetime.strptime(date_str, "%Y-%m-%d"))
        except ValueError:
            continue

    if not dates:
        return None

    return max(dates)


def download_day(client: CAISOAPIClient, date: datetime) -> bool:
    """
    Download ancillary services data for a single day.

    Args:
        client: CAISO API client
        date: Date to download

    Returns:
        True if successful, False otherwise
    """
    date_str = date.strftime("%Y-%m-%d")
    output_file = OUTPUT_DIR / f"ancillary_services_{date_str}.csv"

    # Check if already exists (quick skip)
    if output_file.exists():
        file_size = output_file.stat().st_size
        if file_size > 500:  # At least 500 bytes means likely valid
            logger.info(f"✓ {date_str}: Already exists ({file_size:,} bytes) - skipping")
            return True

    logger.info(f"⏳ Downloading AS prices for {date_str}...")

    # CAISO AS data: request full day (00:00 to 23:59)
    start_dt = date.replace(hour=0, minute=0, second=0, microsecond=0)
    end_dt = date.replace(hour=23, minute=59, second=0, microsecond=0)

    start_str = format_datetime_for_caiso(start_dt)
    end_str = format_datetime_for_caiso(end_dt)

    # Retry loop with exponential backoff
    for attempt in range(MAX_RETRIES):
        try:
            # Get all AS types (API returns all types when anc_type='ALL')
            df = client.get_ancillary_services(start_str, end_str, anc_type="ALL", anc_region="ALL")

            if df is None or len(df) == 0:
                logger.warning(f"⚠️  No data returned for {date_str}")
                return False

            # Basic validation
            logger.info(f"   Downloaded {len(df):,} rows for {date_str}")

            # Save to CSV
            df.to_csv(output_file, index=False)
            file_size = output_file.stat().st_size

            logger.info(f"✓ {date_str}: Saved {len(df):,} rows ({file_size:,} bytes)")
            return True

        except Exception as e:
            error_msg = str(e)
            if attempt < MAX_RETRIES - 1:
                wait_time = BASE_RETRY_DELAY * (2 ** attempt)  # Exponential backoff: 30s, 60s, 120s, 240s, 480s
                logger.warning(f"⚠️  Attempt {attempt + 1}/{MAX_RETRIES} failed: {error_msg}")
                logger.warning(f"⏳ Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                logger.error(f"❌ All {MAX_RETRIES} attempts failed for {date_str}: {error_msg}")
                return False

    return False


def main():
    """Main download loop."""
    print("=" * 80)
    print("CAISO Ancillary Services Historical Download")
    print("=" * 80)

    # Create API client
    client = CAISOAPIClient(min_delay_between_requests=5.0)

    # Determine start date
    latest_date = get_latest_downloaded_date()

    if latest_date:
        start_date = latest_date + timedelta(days=1)
        logger.info(f"Latest downloaded date: {latest_date.strftime('%Y-%m-%d')}")
        logger.info(f"Resuming from: {start_date.strftime('%Y-%m-%d')}")
    else:
        start_date = datetime(2019, 1, 1)
        logger.info(f"No existing data found. Starting from: {start_date.strftime('%Y-%m-%d')}")

    # Download up to yesterday (today's data may not be complete)
    end_date = datetime.now() - timedelta(days=1)
    end_date = end_date.replace(hour=0, minute=0, second=0, microsecond=0)

    if start_date > end_date:
        logger.info("✓ All data is up to date!")
        return

    total_days = (end_date - start_date).days + 1
    logger.info(f"\nDownloading {total_days} days ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"AS types: RU (Reg Up), RD (Reg Down), SR (Spinning), NR (Non-Spinning)")
    print("=" * 80)

    # Download day by day
    success_count = 0
    fail_count = 0
    current_date = start_date

    while current_date <= end_date:
        success = download_day(client, current_date)

        if success:
            success_count += 1
        else:
            fail_count += 1

        current_date += timedelta(days=1)

    # Summary
    print("\n" + "=" * 80)
    print("Download Summary")
    print("=" * 80)
    print(f"Total days processed: {total_days}")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 80)

    if fail_count > 0:
        logger.warning(f"⚠️  {fail_count} days failed to download. You may want to retry.")
        sys.exit(1)
    else:
        logger.info("✓ All data downloaded successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()

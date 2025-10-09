#!/usr/bin/env python3
"""
Download PJM Day-Ahead Ancillary Services Prices (2 years historical data)

This script downloads day-ahead hourly ancillary services pricing data,
respecting the 6 connections/minute rate limit for non-member accounts.

Ancillary Services include:
- RegA (Regulation A)
- RegD (Regulation D)
- Sync_Reserve (Synchronized Reserve)
- Non-Sync_Reserve (Non-Synchronized Reserve)
- Primary_Reserve (Primary Reserve)

Usage:
    python download_da_ancillary_services.py --start-date 2023-10-07 --end-date 2025-10-06
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

# Add parent directory to path to import pjm_api_client
sys.path.insert(0, str(Path(__file__).parent))
from pjm_api_client import PJMAPIClient

# Load environment variables
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def ensure_data_dir(data_dir: Path) -> Path:
    """Ensure data directory exists."""
    csv_dir = data_dir / 'csv_files' / 'da_ancillary_services'
    csv_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Data directory: {csv_dir}")
    return csv_dir


def download_ancillary_services(client: PJMAPIClient, start_date: str, end_date: str,
                                output_dir: Path):
    """
    Download day-ahead ancillary services prices.

    Args:
        client: PJMAPIClient instance
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        output_dir: Output directory for CSV files
    """
    logger.info(f"Downloading ancillary services: {start_date} to {end_date}")

    try:
        # Download data
        data = client.get_ancillary_services(
            start_date=start_date,
            end_date=end_date
        )

        if not data:
            logger.warning(f"No data returned for {start_date} to {end_date}")
            return

        # Check if data is in expected format
        if isinstance(data, dict):
            # Extract items from response
            if 'items' in data:
                items = data['items']
            elif 'data' in data:
                items = data['data']
            else:
                items = [data]
        else:
            items = data

        if not items:
            logger.warning(f"No items in response")
            return

        # Convert to DataFrame
        df = pd.DataFrame(items)

        if df.empty:
            logger.warning(f"Empty dataframe")
            return

        # Save to CSV
        filename = f"ancillary_services_{start_date}_{end_date}.csv"
        output_path = output_dir / filename
        df.to_csv(output_path, index=False)

        logger.info(f"✓ Saved {len(df)} records to {filename}")

    except Exception as e:
        logger.error(f"Error downloading ancillary services: {e}")
        raise


def download_by_quarter(client: PJMAPIClient, start_date: datetime, end_date: datetime,
                       output_dir: Path):
    """
    Download data quarter-by-quarter to respect API limits.

    The API has:
    - 50,000 row limit per request
    - 365 day date range limit per request

    For ancillary services hourly data, we download by quarter (~90 days).
    """
    current = start_date

    while current < end_date:
        # Calculate end of current quarter (roughly 3 months)
        if current.month <= 3:
            quarter_end = datetime(current.year, 3, 31, 23, 59, 59)
        elif current.month <= 6:
            quarter_end = datetime(current.year, 6, 30, 23, 59, 59)
        elif current.month <= 9:
            quarter_end = datetime(current.year, 9, 30, 23, 59, 59)
        else:
            quarter_end = datetime(current.year, 12, 31, 23, 59, 59)

        # Don't go past the overall end date
        quarter_end = min(quarter_end, end_date)

        # Download this quarter
        download_ancillary_services(
            client=client,
            start_date=current.strftime('%Y-%m-%d'),
            end_date=quarter_end.strftime('%Y-%m-%d'),
            output_dir=output_dir
        )

        # Move to next quarter
        if current.month <= 3:
            current = datetime(current.year, 4, 1)
        elif current.month <= 6:
            current = datetime(current.year, 7, 1)
        elif current.month <= 9:
            current = datetime(current.year, 10, 1)
        else:
            current = datetime(current.year + 1, 1, 1)


def main():
    parser = argparse.ArgumentParser(
        description='Download PJM Day-Ahead Ancillary Services Prices'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        required=True,
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        required=True,
        help='End date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        help='Data directory (default: from PJM_DATA_DIR env var)'
    )

    args = parser.parse_args()

    # Parse dates
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d')

    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    logger.info(f"Duration: {(end_date - start_date).days} days")

    # Determine data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = Path(os.getenv('PJM_DATA_DIR', '/home/enrico/data/PJM_data'))

    csv_dir = ensure_data_dir(data_dir)

    # Initialize API client (rate limited to 6 requests/minute)
    try:
        client = PJMAPIClient(requests_per_minute=6)
    except ValueError as e:
        logger.error(f"Failed to initialize API client: {e}")
        logger.error("Please set PJM_API_KEY in your .env file")
        return

    # Download data
    logger.info("Downloading ancillary services data...")

    try:
        download_by_quarter(
            client=client,
            start_date=start_date,
            end_date=end_date,
            output_dir=csv_dir
        )
    except Exception as e:
        logger.error(f"Failed to download: {e}")
        return

    logger.info(f"\n✓ Download complete! Data saved to: {csv_dir}")
    logger.info(f"To combine CSV files:")
    logger.info(f"  cd {csv_dir}")
    logger.info(f"  cat ancillary_services_*.csv > ancillary_services_combined.csv")


if __name__ == "__main__":
    main()

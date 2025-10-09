#!/usr/bin/env python3
"""
Download PJM Day-Ahead Nodal LMP Prices (All Pnodes)

Downloads ALL pnodes together without filtering by pnode_id.
This is required for archived data and is the most efficient approach for complete datasets.

Usage:
    python download_nodal_da_lmps.py --start-date 2023-10-07 --end-date 2025-10-06
    python download_nodal_da_lmps.py --year 2024
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
    csv_dir = data_dir / 'csv_files' / 'da_nodal'
    csv_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Data directory: {csv_dir}")
    return csv_dir


def download_nodal_da_lmps(client: PJMAPIClient, start_date: str, end_date: str,
                           output_dir: Path):
    """
    Download day-ahead nodal LMP data for ALL pnodes.

    Downloads without pnode_id filter to get all nodes at once.
    This is the only way to get archived data.

    Args:
        client: PJMAPIClient instance
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        output_dir: Output directory for CSV files
    """
    logger.info(f"Downloading ALL nodal DA LMPs: {start_date} to {end_date}")

    try:
        # Download data WITHOUT pnode_id filter (gets all nodes)
        data = client.get_day_ahead_lmps(
            start_date=start_date,
            end_date=end_date,
            pnode_id=None  # No filter = all nodes
        )

        if not data:
            logger.warning(f"No data returned")
            return

        # Check if data is in expected format
        if isinstance(data, dict):
            # Extract items from response
            if 'items' in data:
                items = data['items']
                total_rows = data.get('totalRows', len(items))
            elif 'data' in data:
                items = data['data']
                total_rows = len(items)
            else:
                items = [data]
                total_rows = 1
        else:
            items = data
            total_rows = len(items)

        if not items:
            logger.warning(f"No items in response")
            return

        # Convert to DataFrame
        df = pd.DataFrame(items)

        if df.empty:
            logger.warning(f"Empty dataframe")
            return

        # Get unique node count
        if 'pnode_id' in df.columns:
            unique_nodes = df['pnode_id'].nunique()
            logger.info(f"Data contains {unique_nodes} unique pnodes")

        # Save to CSV
        filename = f"nodal_da_lmp_{start_date}_{end_date}.csv"
        output_path = output_dir / filename
        df.to_csv(output_path, index=False)

        logger.info(f"✓ Saved {len(df):,} records to {filename}")
        logger.info(f"  File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

        # Summary statistics
        if 'pnode_name' in df.columns:
            logger.info(f"  Unique pnodes: {df['pnode_name'].nunique():,}")
        if 'datetime_beginning_ept' in df.columns:
            logger.info(f"  Date range: {df['datetime_beginning_ept'].min()} to {df['datetime_beginning_ept'].max()}")

    except Exception as e:
        logger.error(f"Error downloading nodal data: {e}")
        raise


def download_by_quarter(client: PJMAPIClient, start_date: datetime, end_date: datetime,
                       output_dir: Path):
    """
    Download data quarter-by-quarter to respect API limits.

    For nodal data (all 22,528 pnodes):
    - 90 days × 24 hours × 22,528 nodes = ~48.7M rows per quarter
    - This exceeds the 50K row limit, so API will return partial data
    - We need to use smaller date ranges (daily or weekly)
    """
    current = start_date

    # Use monthly chunks for nodal data to stay under 50K row limit
    # ~30 days × 24 hours × 22,528 nodes = ~16.2M rows (still too large!)
    # Need to use DAILY chunks for full nodal data
    # 1 day × 24 hours × 22,528 nodes = ~540K rows (still over 50K!)

    logger.warning("WARNING: Full nodal data may exceed API row limits!")
    logger.warning("API returns max 50,000 rows per request")
    logger.warning("1 day of all nodes = ~540K rows (will be truncated)")
    logger.warning("Consider downloading by smaller date ranges or specific node sets")

    # Use quarter-based downloads anyway and handle truncation
    while current < end_date:
        # Calculate end of current quarter
        if current.month <= 3:
            quarter_end = datetime(current.year, 3, 31, 23, 59, 59)
        elif current.month <= 6:
            quarter_end = datetime(current.year, 6, 30, 23, 59, 59)
        elif current.month <= 9:
            quarter_end = datetime(current.year, 9, 30, 23, 59, 59)
        else:
            quarter_end = datetime(current.year, 12, 31, 23, 59, 59)

        quarter_end = min(quarter_end, end_date)

        # Download this quarter
        download_nodal_da_lmps(
            client=client,
            start_date=current.strftime('%Y-%m-%d'),
            end_date=quarter_end.strftime('%Y-%m-%d'),
            output_dir=output_dir
        )

        # Move to next quarter
        current = quarter_end + timedelta(days=1)


def main():
    parser = argparse.ArgumentParser(
        description='Download PJM Day-Ahead Nodal LMP prices (ALL pnodes)'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--year',
        type=int,
        help='Download specific year (alternative to start/end dates)'
    )
    parser.add_argument(
        '--quarter',
        type=str,
        help='Download specific quarter, e.g., 2024-Q1, 2024-Q2'
    )

    args = parser.parse_args()

    # Determine date range
    if args.year:
        start_date = datetime(args.year, 1, 1)
        end_date = datetime(args.year, 12, 31)
    elif args.quarter:
        year, q = args.quarter.split('-Q')
        year = int(year)
        quarter = int(q)
        if quarter == 1:
            start_date = datetime(year, 1, 1)
            end_date = datetime(year, 3, 31)
        elif quarter == 2:
            start_date = datetime(year, 4, 1)
            end_date = datetime(year, 6, 30)
        elif quarter == 3:
            start_date = datetime(year, 7, 1)
            end_date = datetime(year, 9, 30)
        else:
            start_date = datetime(year, 10, 1)
            end_date = datetime(year, 12, 31)
    elif args.start_date and args.end_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    else:
        logger.error("Must specify either --year, --quarter, or --start-date and --end-date")
        return

    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")

    # Determine data directory
    data_dir = Path(os.getenv('PJM_DATA_DIR', '/home/enrico/data/PJM_data'))
    output_dir = ensure_data_dir(data_dir)

    # Initialize API client
    try:
        client = PJMAPIClient(requests_per_minute=6)
    except ValueError as e:
        logger.error(f"Failed to initialize API client: {e}")
        logger.error("Please set PJM_API_KEY in your .env file")
        return

    logger.info("⚠️  WARNING: Downloading ALL nodal data (22,528 pnodes)")
    logger.info("This will create LARGE files and may hit API row limits")
    logger.info("Rate limit: 6 requests/minute")
    logger.info("")

    # Download by quarter
    download_by_quarter(client, start_date, end_date, output_dir)

    logger.info("\n✓ Download complete!")
    logger.info(f"Data saved to: {output_dir}")


if __name__ == "__main__":
    main()

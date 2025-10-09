#!/usr/bin/env python3
"""
Download PJM Day-Ahead Hub Prices (2-3 years historical data)

This script downloads day-ahead hourly LMP prices for all PJM pricing hubs,
respecting the 6 connections/minute rate limit for non-member accounts.

Usage:
    python download_da_hub_prices.py --years 3
    python download_da_hub_prices.py --start-date 2022-01-01 --end-date 2024-12-31
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


# Known PJM Hub Pnode IDs (major trading hubs)
PJM_HUBS = {
    'AEP': 51291,
    'APS': 51292,
    'ATSI': 116013,
    'BGE': 51293,
    'CHICAGO': 33092709,
    'COMED': 33092710,
    'DAY': 34508503,
    'DEOK': 51297,
    'DOMINION': 51300,
    'DPL': 51301,
    'DUQUESNE': 51302,
    'JCPL': 51303,
    'METED': 51304,
    'PECO': 51305,
    'PENELEC': 51306,
    'PEPCO': 51307,
    'PPL': 51308,
    'PSEG': 51309,
    'RECO': 37737283,
    'WESTERN': 116122887,
    'OHIO': 33092711,
    'NEW_JERSEY': 116013753,
    'WEST': 37737290,
    'EAST': 37737291,
}


def ensure_data_dir(data_dir: Path) -> Path:
    """Ensure data directory exists."""
    csv_dir = data_dir / 'csv_files' / 'da_hubs'
    csv_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Data directory: {csv_dir}")
    return csv_dir


def download_hub_prices(client: PJMAPIClient, hub_name: str, pnode_id: int,
                       start_date: str, end_date: str, output_dir: Path):
    """
    Download day-ahead prices for a specific hub.

    Args:
        client: PJMAPIClient instance
        hub_name: Name of the hub
        pnode_id: Pnode ID for the hub
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        output_dir: Output directory for CSV files
    """
    logger.info(f"Downloading {hub_name} (pnode_id={pnode_id}): {start_date} to {end_date}")

    try:
        # Download data
        data = client.get_day_ahead_lmps(
            start_date=start_date,
            end_date=end_date,
            pnode_id=str(pnode_id)
        )

        if not data:
            logger.warning(f"No data returned for {hub_name}")
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
            logger.warning(f"No items in response for {hub_name}")
            return

        # Convert to DataFrame
        df = pd.DataFrame(items)

        if df.empty:
            logger.warning(f"Empty dataframe for {hub_name}")
            return

        # Save to CSV
        filename = f"{hub_name}_{start_date}_{end_date}.csv"
        output_path = output_dir / filename
        df.to_csv(output_path, index=False)

        logger.info(f"✓ Saved {len(df)} records to {filename}")

    except Exception as e:
        logger.error(f"Error downloading {hub_name}: {e}")
        raise


def download_by_quarter(client: PJMAPIClient, hub_name: str, pnode_id: int,
                       start_date: datetime, end_date: datetime, output_dir: Path):
    """
    Download data quarter-by-quarter to respect API limits.

    The API has:
    - 50,000 row limit per request
    - 365 day date range limit per request (actually enforced)
    - Data older than 731 days is "archived" with limited filters

    For day-ahead hourly data (24 rows/day), 90 days = 2,160 rows, well under the limit.
    We download by quarter (~90 days) for clean organization and staying under 365-day limit.
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
        download_hub_prices(
            client=client,
            hub_name=hub_name,
            pnode_id=pnode_id,
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
        description='Download PJM Day-Ahead Hub Prices'
    )
    parser.add_argument(
        '--years',
        type=int,
        default=3,
        help='Number of years of historical data (default: 3)'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date (YYYY-MM-DD). Overrides --years'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date (YYYY-MM-DD). Defaults to today'
    )
    parser.add_argument(
        '--hubs',
        type=str,
        nargs='+',
        help='Specific hubs to download (default: all)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        help='Data directory (default: from PJM_DATA_DIR env var)'
    )
    parser.add_argument(
        '--list-hubs',
        action='store_true',
        help='List available hubs and exit'
    )

    args = parser.parse_args()

    # List hubs if requested
    if args.list_hubs:
        print("\nAvailable PJM Hubs:")
        print("=" * 50)
        for hub_name, pnode_id in sorted(PJM_HUBS.items()):
            print(f"  {hub_name:20s} (pnode_id={pnode_id})")
        print()
        return

    # Determine date range
    if args.start_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    else:
        # Default: N years ago from today
        start_date = datetime.now() - timedelta(days=365 * args.years)

    if args.end_date:
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    else:
        # Default: yesterday (today's data may not be complete)
        end_date = datetime.now() - timedelta(days=1)

    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    logger.info(f"Duration: {(end_date - start_date).days} days")

    # Determine data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = Path(os.getenv('PJM_DATA_DIR', '/home/enrico/data/PJM_data'))

    csv_dir = ensure_data_dir(data_dir)

    # Determine which hubs to download
    if args.hubs:
        hubs_to_download = {name: PJM_HUBS[name] for name in args.hubs
                           if name in PJM_HUBS}
        if not hubs_to_download:
            logger.error(f"No valid hubs found. Use --list-hubs to see available hubs.")
            return
    else:
        hubs_to_download = PJM_HUBS

    logger.info(f"Downloading {len(hubs_to_download)} hubs")

    # Initialize API client (rate limited to 6 requests/minute)
    try:
        client = PJMAPIClient(requests_per_minute=6)
    except ValueError as e:
        logger.error(f"Failed to initialize API client: {e}")
        logger.error("Please set PJM_API_KEY in your .env file")
        return

    # Download each hub
    total_hubs = len(hubs_to_download)
    for idx, (hub_name, pnode_id) in enumerate(hubs_to_download.items(), 1):
        logger.info(f"\n[{idx}/{total_hubs}] Processing {hub_name}")

        try:
            download_by_quarter(
                client=client,
                hub_name=hub_name,
                pnode_id=pnode_id,
                start_date=start_date,
                end_date=end_date,
                output_dir=csv_dir
            )
        except Exception as e:
            logger.error(f"Failed to download {hub_name}: {e}")
            continue

    logger.info(f"\n✓ Download complete! Data saved to: {csv_dir}")
    logger.info(f"To combine CSV files by hub:")
    logger.info(f"  cd {csv_dir}")
    logger.info(f"  for hub in {' '.join(list(hubs_to_download.keys())[:3])} ...; do")
    logger.info(f"    cat ${{hub}}_*.csv > ${{hub}}_combined.csv")
    logger.info(f"  done")


if __name__ == "__main__":
    main()

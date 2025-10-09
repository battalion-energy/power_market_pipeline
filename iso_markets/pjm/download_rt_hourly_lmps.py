#!/usr/bin/env python3
"""
Download PJM Real-Time Hourly LMP Prices

This script downloads real-time hourly LMP prices (settlement roll-up of 5-minute runs)
for all PJM pricing hubs, respecting the 6 connections/minute rate limit.

Usage:
    python download_rt_hourly_lmps.py --start-date 2023-10-07 --end-date 2025-10-06
    python download_rt_hourly_lmps.py --years 2
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
    csv_dir = data_dir / 'csv_files' / 'rt_hourly'
    csv_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Data directory: {csv_dir}")
    return csv_dir


def download_rt_hourly_prices(client: PJMAPIClient, hub_name: str, pnode_id: int,
                               start_date: str, end_date: str, output_dir: Path):
    """
    Download real-time hourly prices for a specific hub.

    Args:
        client: PJMAPIClient instance
        hub_name: Name of the hub
        pnode_id: Pnode ID for the hub
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        output_dir: Output directory for CSV files
    """
    logger.info(f"Downloading RT Hourly {hub_name} (pnode_id={pnode_id}): {start_date} to {end_date}")

    try:
        # Download data
        data = client.get_rt_hourly_lmps(
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
        filename = f"{hub_name}_RT_HOURLY_{start_date}_{end_date}.csv"
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
    - 365 day date range limit per request
    - Data older than 731 days is "archived" with limited filters

    For RT hourly data (24 rows/day), 90 days = 2,160 rows, well under the limit.
    """
    current = start_date

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
        download_rt_hourly_prices(
            client=client,
            hub_name=hub_name,
            pnode_id=pnode_id,
            start_date=current.strftime('%Y-%m-%d'),
            end_date=quarter_end.strftime('%Y-%m-%d'),
            output_dir=output_dir
        )

        # Move to next quarter
        current = quarter_end + timedelta(days=1)


def main():
    parser = argparse.ArgumentParser(
        description='Download PJM Real-Time Hourly LMP prices for all hubs'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date (YYYY-MM-DD). Defaults to 2 years ago.'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date (YYYY-MM-DD). Defaults to yesterday.'
    )
    parser.add_argument(
        '--years',
        type=int,
        help='Number of years back from yesterday (alternative to start-date)'
    )
    parser.add_argument(
        '--hubs',
        nargs='+',
        help='Specific hubs to download (default: all hubs)'
    )

    args = parser.parse_args()

    # Determine date range
    if args.end_date:
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    else:
        end_date = datetime.now() - timedelta(days=1)

    if args.start_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    elif args.years:
        start_date = end_date - timedelta(days=365 * args.years)
    else:
        # Default to 2 years
        start_date = end_date - timedelta(days=365 * 2)

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

    # Select hubs to download
    if args.hubs:
        hubs_to_download = {k: v for k, v in PJM_HUBS.items() if k in args.hubs}
        if not hubs_to_download:
            logger.error(f"No valid hubs found. Available hubs: {', '.join(PJM_HUBS.keys())}")
            return
    else:
        hubs_to_download = PJM_HUBS

    logger.info(f"Downloading {len(hubs_to_download)} hubs")
    logger.info(f"Rate limit: 6 requests/minute")
    logger.info(f"Estimated time: ~{len(hubs_to_download) * 2} minutes for 2 years\n")

    # Download each hub quarter-by-quarter
    for i, (hub_name, pnode_id) in enumerate(hubs_to_download.items(), 1):
        logger.info(f"\n[{i}/{len(hubs_to_download)}] Processing {hub_name}...")
        try:
            download_by_quarter(client, hub_name, pnode_id, start_date, end_date, output_dir)
        except Exception as e:
            logger.error(f"Failed to download {hub_name}: {e}")
            logger.warning("Continuing with next hub...")
            continue

    logger.info("\n✓ Download complete!")
    logger.info(f"Data saved to: {output_dir}")


if __name__ == "__main__":
    main()

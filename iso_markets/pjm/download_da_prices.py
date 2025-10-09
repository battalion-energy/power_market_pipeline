#!/usr/bin/env python3
"""
Download historical day-ahead energy and ancillary service prices from PJM.
Supports downloading data for hubs, specific pnodes, or all pnodes.
"""

import os
import argparse
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional
import logging
from pjm_api_client import PJMAPIClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PJMDayAheadDownloader:
    """Download day-ahead market data from PJM."""

    def __init__(self, api_key: Optional[str] = None, output_dir: str = 'data'):
        """
        Initialize downloader.

        Args:
            api_key: PJM API key
            output_dir: Directory to save downloaded data
        """
        self.client = PJMAPIClient(api_key=api_key)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _generate_date_chunks(self, start_date: str, end_date: str,
                             chunk_days: int = 30) -> List[tuple]:
        """
        Generate date chunks to avoid hitting API row limits.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            chunk_days: Days per chunk

        Returns:
            List of (start, end) date tuples
        """
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')

        chunks = []
        current = start

        while current < end:
            chunk_end = min(current + timedelta(days=chunk_days), end)
            chunks.append((
                current.strftime('%Y-%m-%d'),
                chunk_end.strftime('%Y-%m-%d')
            ))
            current = chunk_end + timedelta(days=1)

        return chunks

    def download_da_lmps_for_hub(self, hub_pnode_id: str, hub_name: str,
                                 start_date: str, end_date: str) -> pd.DataFrame:
        """
        Download day-ahead LMPs for a specific hub.

        Args:
            hub_pnode_id: Hub pnode ID
            hub_name: Hub name for output file
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with price data
        """
        logger.info(f"Downloading DA LMPs for hub {hub_name} ({hub_pnode_id})")

        all_data = []
        chunks = self._generate_date_chunks(start_date, end_date, chunk_days=30)

        for chunk_start, chunk_end in chunks:
            logger.info(f"  Fetching {chunk_start} to {chunk_end}")

            try:
                data = self.client.get_day_ahead_lmps(
                    chunk_start, chunk_end, pnode_id=hub_pnode_id
                )
                all_data.extend(data)
                logger.info(f"    Retrieved {len(data)} records")

            except Exception as e:
                logger.error(f"Error fetching data for {chunk_start} to {chunk_end}: {e}")
                continue

        if not all_data:
            logger.warning(f"No data retrieved for {hub_name}")
            return pd.DataFrame()

        df = pd.DataFrame(all_data)

        # Save to CSV
        output_file = self.output_dir / f"da_lmps_{hub_name}_{start_date}_{end_date}.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Saved {len(df)} records to {output_file}")

        return df

    def download_da_lmps_all_hubs(self, start_date: str, end_date: str) -> None:
        """
        Download day-ahead LMPs for all hubs.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        """
        logger.info("Fetching list of hubs...")
        hubs = self.client.get_hubs()
        logger.info(f"Found {len(hubs)} hubs")

        for hub in hubs:
            hub_id = hub.get('pnode_id')
            hub_name = hub.get('pnode_name', f'hub_{hub_id}').replace(' ', '_')

            self.download_da_lmps_for_hub(hub_id, hub_name, start_date, end_date)

    def download_da_lmps_all_pnodes(self, start_date: str, end_date: str) -> None:
        """
        Download day-ahead LMPs for all pnodes.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        """
        logger.info("Fetching list of all pnodes...")
        pnodes = self.client.get_pnodes()
        logger.info(f"Found {len(pnodes)} pnodes")

        for i, pnode in enumerate(pnodes, 1):
            pnode_id = pnode.get('pnode_id')
            pnode_name = pnode.get('pnode_name', f'pnode_{pnode_id}').replace(' ', '_')

            logger.info(f"Processing pnode {i}/{len(pnodes)}: {pnode_name}")
            self.download_da_lmps_for_hub(pnode_id, pnode_name, start_date, end_date)

    def download_ancillary_services(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Download ancillary services pricing data.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with ancillary service prices
        """
        logger.info("Downloading ancillary services data")

        all_data = []
        chunks = self._generate_date_chunks(start_date, end_date, chunk_days=30)

        for chunk_start, chunk_end in chunks:
            logger.info(f"  Fetching {chunk_start} to {chunk_end}")

            try:
                data = self.client.get_ancillary_services(chunk_start, chunk_end)
                all_data.extend(data)
                logger.info(f"    Retrieved {len(data)} records")

            except Exception as e:
                logger.error(f"Error fetching data for {chunk_start} to {chunk_end}: {e}")
                continue

        if not all_data:
            logger.warning("No ancillary services data retrieved")
            return pd.DataFrame()

        df = pd.DataFrame(all_data)

        # Save to CSV
        output_file = self.output_dir / f"ancillary_services_{start_date}_{end_date}.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Saved {len(df)} records to {output_file}")

        return df


def main():
    parser = argparse.ArgumentParser(
        description='Download historical day-ahead prices from PJM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download DA LMPs for all hubs for last 5 years
  python download_da_prices.py --mode hubs --years 5

  # Download for specific hub
  python download_da_prices.py --mode hub --pnode-id 51291 --hub-name AEP_GEN_HUB --years 5

  # Download for all pnodes for last year
  python download_da_prices.py --mode all-pnodes --years 1

  # Download ancillary services
  python download_da_prices.py --mode ancillary --years 5

  # Custom date range
  python download_da_prices.py --mode hubs --start-date 2020-01-01 --end-date 2025-01-01
        """
    )

    parser.add_argument('--mode', required=True,
                       choices=['hub', 'hubs', 'all-pnodes', 'ancillary'],
                       help='Download mode')
    parser.add_argument('--pnode-id', help='Specific pnode ID (for hub mode)')
    parser.add_argument('--hub-name', help='Hub name (for hub mode)')
    parser.add_argument('--years', type=int, help='Number of years back to download')
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    parser.add_argument('--output-dir', default='data',
                       help='Output directory (default: data)')
    parser.add_argument('--api-key', help='PJM API key (or set PJM_API_KEY env var)')

    args = parser.parse_args()

    # Calculate date range
    if args.start_date and args.end_date:
        start_date = args.start_date
        end_date = args.end_date
    elif args.years:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=args.years * 365)).strftime('%Y-%m-%d')
    else:
        parser.error("Provide either --years or both --start-date and --end-date")

    logger.info(f"Date range: {start_date} to {end_date}")

    # Initialize downloader
    downloader = PJMDayAheadDownloader(api_key=args.api_key, output_dir=args.output_dir)

    # Execute based on mode
    if args.mode == 'hub':
        if not args.pnode_id or not args.hub_name:
            parser.error("--pnode-id and --hub-name required for hub mode")
        downloader.download_da_lmps_for_hub(
            args.pnode_id, args.hub_name, start_date, end_date
        )

    elif args.mode == 'hubs':
        downloader.download_da_lmps_all_hubs(start_date, end_date)

    elif args.mode == 'all-pnodes':
        downloader.download_da_lmps_all_pnodes(start_date, end_date)

    elif args.mode == 'ancillary':
        downloader.download_ancillary_services(start_date, end_date)

    logger.info("Download complete!")


if __name__ == "__main__":
    main()

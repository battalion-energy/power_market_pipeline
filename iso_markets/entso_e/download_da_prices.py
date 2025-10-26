#!/usr/bin/env python3
"""
Download Day-Ahead Market Prices from ENTSO-E Transparency Platform

This script downloads hourly day-ahead market clearing prices for European
bidding zones and saves them to CSV files for further processing.

Usage:
    python download_da_prices.py --zones DE_LU FR NL --start-date 2024-01-01 --end-date 2024-12-31
    python download_da_prices.py --priority 1 --start-date 2024-01-01 --end-date 2024-01-31
    python download_da_prices.py --all-zones --start-date 2024-12-01 --end-date 2024-12-31
"""

import os
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple
import pandas as pd

from .entso_e_api_client import ENTSOEAPIClient
from .european_zones import BIDDING_ZONES, get_priority_1_zones, get_zones_by_priority

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DAMPriceDownloader:
    """Download day-ahead market prices from ENTSO-E."""

    def __init__(self, api_key: str = None, output_dir: str = None):
        """
        Initialize downloader.

        Args:
            api_key: ENTSO-E API key (uses ENTSO_E_API_KEY env var if not provided)
            output_dir: Output directory (uses ENTSO_E_DATA_DIR env var if not provided)
        """
        self.client = ENTSOEAPIClient(api_key=api_key)

        # Set up output directory
        if output_dir is None:
            output_dir = os.getenv('ENTSO_E_DATA_DIR', '/pool/ssd8tb/data/iso/ENTSO_E')

        self.output_dir = Path(output_dir) / 'csv_files' / 'da_prices'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Output directory: {self.output_dir}")

    def _generate_date_chunks(self, start_date: datetime, end_date: datetime,
                             chunk_days: int = 90) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Generate date chunks for API queries.

        Args:
            start_date: Start date
            end_date: End date
            chunk_days: Days per chunk (default 90 for day-ahead hourly data)

        Returns:
            List of (start, end) timestamp tuples
        """
        chunks = []
        current_start = start_date

        while current_start < end_date:
            current_end = min(current_start + timedelta(days=chunk_days), end_date)
            chunks.append((current_start, current_end))
            current_start = current_end

        logger.info(f"Split date range into {len(chunks)} chunks of ~{chunk_days} days")
        return chunks

    def download_zone(self, zone_name: str, start_date: datetime,
                     end_date: datetime, chunk_days: int = 90) -> pd.DataFrame:
        """
        Download day-ahead prices for a single zone.

        Args:
            zone_name: Bidding zone name (e.g., 'DE_LU', 'FR')
            start_date: Start date
            end_date: End date
            chunk_days: Days per API request chunk

        Returns:
            DataFrame with all prices for the date range
        """
        zone_config = BIDDING_ZONES.get(zone_name)
        if zone_config is None:
            raise ValueError(f"Unknown zone: {zone_name}")

        logger.info(f"\n{'='*80}")
        logger.info(f"Downloading DA prices for {zone_config.name} ({zone_name})")
        logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
        logger.info(f"{'='*80}")

        # Convert to timezone-aware timestamps
        tz = zone_config.timezone
        start_ts = pd.Timestamp(start_date, tz=tz)
        end_ts = pd.Timestamp(end_date, tz=tz)

        # Generate date chunks
        chunks = self._generate_date_chunks(start_date, end_date, chunk_days)

        # Download each chunk
        all_data = []
        for i, (chunk_start, chunk_end) in enumerate(chunks, 1):
            chunk_start_ts = pd.Timestamp(chunk_start, tz=tz)
            chunk_end_ts = pd.Timestamp(chunk_end, tz=tz)

            logger.info(f"Chunk {i}/{len(chunks)}: {chunk_start.date()} to {chunk_end.date()}")

            try:
                df = self.client.query_day_ahead_prices(
                    zone_name,
                    start=chunk_start_ts,
                    end=chunk_end_ts
                )

                if not df.empty:
                    all_data.append(df)
                    logger.info(f"  Retrieved {len(df)} records")
                else:
                    logger.warning(f"  No data returned for this chunk")

            except Exception as e:
                logger.error(f"  Failed to download chunk: {str(e)}")
                # Continue with next chunk rather than failing completely
                continue

        # Combine all chunks
        if all_data:
            combined = pd.concat(all_data, axis=0)
            combined = combined.sort_index()

            # Remove any duplicates (shouldn't happen but just in case)
            combined = combined[~combined.index.duplicated(keep='first')]

            logger.info(f"\nTotal records retrieved: {len(combined)}")
            logger.info(f"Date range: {combined.index[0]} to {combined.index[-1]}")

            return combined
        else:
            logger.warning(f"No data retrieved for {zone_name}")
            return pd.DataFrame()

    def save_to_csv(self, df: pd.DataFrame, zone_name: str,
                   start_date: datetime, end_date: datetime) -> Path:
        """
        Save DataFrame to CSV file.

        Args:
            df: DataFrame with prices
            zone_name: Bidding zone name
            start_date: Start date
            end_date: End date

        Returns:
            Path to saved file
        """
        if df.empty:
            logger.warning(f"No data to save for {zone_name}")
            return None

        # Generate filename
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        filename = f"da_prices_{zone_name}_{start_str}_{end_str}.csv"
        filepath = self.output_dir / filename

        # Save to CSV
        df.to_csv(filepath)
        logger.info(f"Saved to: {filepath}")
        logger.info(f"File size: {filepath.stat().st_size / 1024:.1f} KB")

        return filepath

    def download_and_save(self, zone_name: str, start_date: datetime,
                         end_date: datetime, chunk_days: int = 90) -> Path:
        """
        Download and save day-ahead prices for a zone.

        Args:
            zone_name: Bidding zone name
            start_date: Start date
            end_date: End date
            chunk_days: Days per chunk

        Returns:
            Path to saved CSV file
        """
        df = self.download_zone(zone_name, start_date, end_date, chunk_days)

        if not df.empty:
            return self.save_to_csv(df, zone_name, start_date, end_date)
        else:
            return None

    def download_multiple_zones(self, zone_names: List[str], start_date: datetime,
                               end_date: datetime, chunk_days: int = 90) -> dict:
        """
        Download day-ahead prices for multiple zones.

        Args:
            zone_names: List of zone names
            start_date: Start date
            end_date: End date
            chunk_days: Days per chunk

        Returns:
            Dictionary mapping zone names to output file paths
        """
        results = {}

        for zone_name in zone_names:
            try:
                filepath = self.download_and_save(zone_name, start_date, end_date, chunk_days)
                results[zone_name] = filepath
            except Exception as e:
                logger.error(f"Failed to download {zone_name}: {str(e)}")
                results[zone_name] = None

        # Summary
        successful = sum(1 for path in results.values() if path is not None)
        logger.info(f"\n{'='*80}")
        logger.info(f"DOWNLOAD COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Successful: {successful}/{len(zone_names)}")
        logger.info(f"Failed: {len(zone_names) - successful}/{len(zone_names)}")

        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Download day-ahead prices from ENTSO-E Transparency Platform',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download Germany for 2024
  python download_da_prices.py --zones DE_LU --start-date 2024-01-01 --end-date 2024-12-31

  # Download all priority 1 markets for January 2024
  python download_da_prices.py --priority 1 --start-date 2024-01-01 --end-date 2024-01-31

  # Download specific zones
  python download_da_prices.py --zones DE_LU FR NL BE --start-date 2024-01-01 --end-date 2024-12-31

  # Download all configured zones (use with caution!)
  python download_da_prices.py --all-zones --start-date 2024-12-01 --end-date 2024-12-31
        """
    )

    # Zone selection
    zone_group = parser.add_mutually_exclusive_group(required=True)
    zone_group.add_argument('--zones', nargs='+', help='Specific zone names (e.g., DE_LU FR NL)')
    zone_group.add_argument('--priority', type=int, choices=[1, 2, 3],
                           help='Download all zones with this priority level')
    zone_group.add_argument('--all-zones', action='store_true',
                           help='Download all configured zones')

    # Date range
    parser.add_argument('--start-date', required=True,
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=True,
                       help='End date (YYYY-MM-DD)')

    # Options
    parser.add_argument('--chunk-days', type=int, default=90,
                       help='Days per API request chunk (default: 90)')
    parser.add_argument('--output-dir', type=str,
                       help='Output directory (default: $ENTSO_E_DATA_DIR/csv_files/da_prices)')
    parser.add_argument('--api-key', type=str,
                       help='ENTSO-E API key (default: $ENTSO_E_API_KEY)')

    args = parser.parse_args()

    # Parse dates
    try:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    except ValueError as e:
        parser.error(f"Invalid date format: {e}")

    # Validate date range
    if end_date <= start_date:
        parser.error("End date must be after start date")

    # Determine which zones to download
    if args.zones:
        zone_names = args.zones
    elif args.priority:
        zones = get_zones_by_priority(args.priority)
        zone_names = [name for name, zone in BIDDING_ZONES.items() if zone.priority == args.priority]
        logger.info(f"Selected {len(zone_names)} priority {args.priority} zones")
    else:  # all-zones
        zone_names = list(BIDDING_ZONES.keys())
        logger.info(f"Selected all {len(zone_names)} configured zones")

    # Validate zone names
    invalid_zones = [z for z in zone_names if z not in BIDDING_ZONES]
    if invalid_zones:
        parser.error(f"Invalid zone names: {invalid_zones}")

    # Initialize downloader
    try:
        downloader = DAMPriceDownloader(
            api_key=args.api_key,
            output_dir=args.output_dir
        )
    except ValueError as e:
        parser.error(str(e))

    # Download
    logger.info(f"\nDownloading day-ahead prices")
    logger.info(f"Zones: {', '.join(zone_names)}")
    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    logger.info(f"Chunk size: {args.chunk_days} days")

    results = downloader.download_multiple_zones(
        zone_names,
        start_date,
        end_date,
        chunk_days=args.chunk_days
    )

    # Print summary
    print("\n" + "="*80)
    print("DOWNLOAD SUMMARY")
    print("="*80)
    for zone, filepath in results.items():
        status = "✓" if filepath else "✗"
        print(f"{status} {zone:15s} {filepath if filepath else 'FAILED'}")


if __name__ == '__main__':
    main()

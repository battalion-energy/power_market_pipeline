#!/usr/bin/env python3
"""
Download Imbalance Prices from ENTSO-E Transparency Platform

Imbalance prices represent the cost of deviating from scheduled positions in European
markets. These are the closest equivalent to "real-time" prices in European markets.

For Germany: Typically 15-minute resolution
For other markets: Resolution varies (15-min to hourly)

Usage:
    python download_imbalance_prices.py --zones DE_LU --start-date 2024-01-01 --end-date 2024-12-31
    python download_imbalance_prices.py --priority 1 --start-date 2024-12-01 --end-date 2024-12-31
"""

import os
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple
import pandas as pd

from .entso_e_api_client import ENTSOEAPIClient
from .european_zones import BIDDING_ZONES, get_zones_by_priority

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ImbalancePriceDownloader:
    """Download imbalance prices from ENTSO-E."""

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

        self.output_dir = Path(output_dir) / 'csv_files' / 'imbalance_prices'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Output directory: {self.output_dir}")

    def _generate_date_chunks(self, start_date: datetime, end_date: datetime,
                             chunk_days: int = 30) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Generate date chunks for API queries.

        Args:
            start_date: Start date
            end_date: End date
            chunk_days: Days per chunk (default 30 for 15-minute data)

        Returns:
            List of (start, end) timestamp tuples

        Note:
            Smaller chunks than day-ahead due to higher temporal resolution (15-min vs hourly)
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
                     end_date: datetime, chunk_days: int = 30) -> pd.DataFrame:
        """
        Download imbalance prices for a single zone.

        Args:
            zone_name: Bidding zone name (e.g., 'DE_LU', 'FR')
            start_date: Start date
            end_date: End date
            chunk_days: Days per API request chunk

        Returns:
            DataFrame with all imbalance prices for the date range
        """
        zone_config = BIDDING_ZONES.get(zone_name)
        if zone_config is None:
            raise ValueError(f"Unknown zone: {zone_name}")

        logger.info(f"\n{'='*80}")
        logger.info(f"Downloading imbalance prices for {zone_config.name} ({zone_name})")
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
                df = self.client.query_imbalance_prices(
                    zone_name,
                    start=chunk_start_ts,
                    end=chunk_end_ts
                )

                if not df.empty:
                    all_data.append(df)
                    logger.info(f"  Retrieved {len(df)} records")

                    # Log data characteristics
                    time_diff = df.index.to_series().diff().median()
                    if pd.notna(time_diff):
                        logger.info(f"  Temporal resolution: {time_diff}")

                else:
                    logger.warning(f"  No data returned for this chunk")

            except Exception as e:
                logger.error(f"  Failed to download chunk: {str(e)}")
                logger.warning(f"  Note: {zone_name} may not provide imbalance price data")
                # Continue with next chunk
                continue

        # Combine all chunks
        if all_data:
            combined = pd.concat(all_data, axis=0)
            combined = combined.sort_index()

            # Remove any duplicates
            combined = combined[~combined.index.duplicated(keep='first')]

            logger.info(f"\nTotal records retrieved: {len(combined)}")
            logger.info(f"Date range: {combined.index[0]} to {combined.index[-1]}")

            # Calculate temporal resolution
            time_diff = combined.index.to_series().diff().median()
            if pd.notna(time_diff):
                logger.info(f"Data resolution: {time_diff}")

            return combined
        else:
            logger.warning(f"No data retrieved for {zone_name}")
            logger.warning(f"This zone may not publish imbalance prices via ENTSO-E")
            return pd.DataFrame()

    def save_to_csv(self, df: pd.DataFrame, zone_name: str,
                   start_date: datetime, end_date: datetime) -> Path:
        """
        Save DataFrame to CSV file.

        Args:
            df: DataFrame with imbalance prices
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
        filename = f"imbalance_prices_{zone_name}_{start_str}_{end_str}.csv"
        filepath = self.output_dir / filename

        # Save to CSV
        df.to_csv(filepath)
        logger.info(f"Saved to: {filepath}")
        logger.info(f"File size: {filepath.stat().st_size / 1024:.1f} KB")

        return filepath

    def download_and_save(self, zone_name: str, start_date: datetime,
                         end_date: datetime, chunk_days: int = 30) -> Path:
        """
        Download and save imbalance prices for a zone.

        Args:
            zone_name: Bidding zone name
            start_date: Start date
            end_date: End date
            chunk_days: Days per chunk

        Returns:
            Path to saved CSV file (None if no data available)
        """
        df = self.download_zone(zone_name, start_date, end_date, chunk_days)

        if not df.empty:
            return self.save_to_csv(df, zone_name, start_date, end_date)
        else:
            return None

    def download_multiple_zones(self, zone_names: List[str], start_date: datetime,
                               end_date: datetime, chunk_days: int = 30) -> dict:
        """
        Download imbalance prices for multiple zones.

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
        unavailable = sum(1 for path in results.values() if path is None)

        logger.info(f"\n{'='*80}")
        logger.info(f"DOWNLOAD COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Successful: {successful}/{len(zone_names)}")
        logger.info(f"Unavailable/Failed: {unavailable}/{len(zone_names)}")
        logger.info(f"\nNote: Not all zones publish imbalance prices via ENTSO-E")

        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Download imbalance prices from ENTSO-E Transparency Platform',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download Germany for 2024
  python download_imbalance_prices.py --zones DE_LU --start-date 2024-01-01 --end-date 2024-12-31

  # Download all priority 1 markets for Q4 2024
  python download_imbalance_prices.py --priority 1 --start-date 2024-10-01 --end-date 2024-12-31

  # Download specific zones with smaller chunks
  python download_imbalance_prices.py --zones DE_LU FR --start-date 2024-01-01 --end-date 2024-03-31 --chunk-days 15

Note: Not all zones publish imbalance prices. Germany (DE_LU) has good coverage.
        """
    )

    # Zone selection
    zone_group = parser.add_mutually_exclusive_group(required=True)
    zone_group.add_argument('--zones', nargs='+', help='Specific zone names (e.g., DE_LU FR NL)')
    zone_group.add_argument('--priority', type=int, choices=[1, 2, 3],
                           help='Download all zones with this priority level')
    zone_group.add_argument('--all-zones', action='store_true',
                           help='Download all configured zones (many will have no data)')

    # Date range
    parser.add_argument('--start-date', required=True,
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=True,
                       help='End date (YYYY-MM-DD)')

    # Options
    parser.add_argument('--chunk-days', type=int, default=30,
                       help='Days per API request chunk (default: 30, smaller due to 15-min resolution)')
    parser.add_argument('--output-dir', type=str,
                       help='Output directory (default: $ENTSO_E_DATA_DIR/csv_files/imbalance_prices)')
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
        zone_names = [name for name, zone in BIDDING_ZONES.items() if zone.priority == args.priority]
        logger.info(f"Selected {len(zone_names)} priority {args.priority} zones")
    else:  # all-zones
        zone_names = list(BIDDING_ZONES.keys())
        logger.info(f"Selected all {len(zone_names)} configured zones")
        logger.warning("Many zones may not have imbalance price data available")

    # Validate zone names
    invalid_zones = [z for z in zone_names if z not in BIDDING_ZONES]
    if invalid_zones:
        parser.error(f"Invalid zone names: {invalid_zones}")

    # Initialize downloader
    try:
        downloader = ImbalancePriceDownloader(
            api_key=args.api_key,
            output_dir=args.output_dir
        )
    except ValueError as e:
        parser.error(str(e))

    # Download
    logger.info(f"\nDownloading imbalance prices")
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
        if filepath:
            status = "✓"
            msg = filepath
        else:
            status = "✗"
            msg = "No data available or failed"
        print(f"{status} {zone:15s} {msg}")


if __name__ == '__main__':
    main()

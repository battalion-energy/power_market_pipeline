#!/usr/bin/env python3
"""
Download German Ancillary Services from Regelleistung.net

Downloads FCR, aFRR, and mFRR tender results for Germany from Regelleistung.net.
This is critical data for BESS revenue optimization in Germany.

Products:
- FCR: Frequency Containment Reserve (primary, 30-second activation)
- aFRR: Automatic Frequency Restoration Reserve (secondary, 5-min activation) - BEST opportunity!
- mFRR: Manual Frequency Restoration Reserve (tertiary, 15-min activation)

Markets:
- CAPACITY: Capacity tender results (€/MW per hour or per 4-hour block)
- ENERGY: Energy activation prices (€/MWh)

Time Structure:
- 6 x 4-hour blocks per day:
  * 00-04, 04-08, 08-12, 12-16, 16-20, 20-24
- aFRR and mFRR have POS (positive/upward) and NEG (negative/downward) directions

Usage:
    # Download all products for 2024
    python download_ancillary_services.py --start-date 2024-01-01 --end-date 2024-12-31

    # Download only aFRR (best opportunity for BESS)
    python download_ancillary_services.py --products aFRR --start-date 2024-01-01 --end-date 2024-12-31

    # Download specific product and market
    python download_ancillary_services.py --products aFRR --markets CAPACITY --start-date 2024-12-01 --end-date 2024-12-31

    # Download from historical data (available from 2012+)
    python download_ancillary_services.py --start-date 2020-01-01 --end-date 2024-12-31
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List
import pandas as pd

# Handle both direct execution and module import
try:
    from .regelleistung_api_client import RegelleistungAPIClient
except ImportError:
    from regelleistung_api_client import RegelleistungAPIClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AncillaryServicesDownloader:
    """Download ancillary services data from Regelleistung.net."""

    def __init__(self, output_dir: str = None):
        """
        Initialize downloader.

        Args:
            output_dir: Output directory (uses ENTSO_E_DATA_DIR env var if not provided)
        """
        self.client = RegelleistungAPIClient()

        # Set up output directory
        if output_dir is None:
            base_dir = os.getenv('ENTSO_E_DATA_DIR', '/pool/ssd8tb/data/iso/ENTSO_E')
            output_dir = Path(base_dir) / 'csv_files' / 'de_ancillary_services'
        else:
            output_dir = Path(output_dir)

        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Output directory: {self.output_dir}")

    def download_product_market(self, product: str, market: str,
                               start_date: datetime, end_date: datetime) -> Path:
        """
        Download a specific product and market for a date range.

        Args:
            product: 'FCR', 'aFRR', or 'mFRR'
            market: 'CAPACITY' or 'ENERGY'
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            Path to saved CSV file, or None if no data
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Downloading {product} {market}")
        logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
        logger.info(f"{'='*80}")

        # Download all days
        all_dfs = self.client.download_date_range(product, market, start_date, end_date)

        if not all_dfs:
            logger.warning(f"No data retrieved for {product} {market}")
            return None

        # Combine all days
        combined = pd.concat(all_dfs, axis=0, ignore_index=True)
        logger.info(f"\nCombined {len(all_dfs)} days into {len(combined)} total records")

        # Save to CSV
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        filename = f"{product.lower()}_{market.lower()}_de_{start_str}_{end_str}.csv"
        filepath = self.output_dir / filename

        combined.to_csv(filepath, index=False)
        logger.info(f"\nSaved to: {filepath}")
        logger.info(f"File size: {filepath.stat().st_size / 1024:.1f} KB")

        return filepath

    def download_all_products(self, products: List[str], markets: List[str],
                             start_date: datetime, end_date: datetime) -> dict:
        """
        Download multiple products and markets.

        Args:
            products: List of products to download
            markets: List of markets to download
            start_date: Start date
            end_date: End date

        Returns:
            Dictionary mapping (product, market) to output file path
        """
        results = {}

        for product in products:
            for market in markets:
                # Skip invalid combinations
                if product == 'FCR' and market == 'ENERGY':
                    logger.info(f"\nSkipping FCR ENERGY (not applicable)")
                    continue

                try:
                    filepath = self.download_product_market(product, market, start_date, end_date)
                    results[(product, market)] = filepath
                except Exception as e:
                    logger.error(f"\nFailed to download {product} {market}: {e}")
                    results[(product, market)] = None

        # Summary
        logger.info(f"\n{'='*80}")
        logger.info(f"DOWNLOAD COMPLETE")
        logger.info(f"{'='*80}")

        successful = sum(1 for path in results.values() if path is not None)
        failed = sum(1 for path in results.values() if path is None)

        logger.info(f"Successful: {successful}/{len(results)}")
        logger.info(f"Failed: {failed}/{len(results)}")

        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Download German ancillary services from Regelleistung.net',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all products for 2024
  python download_ancillary_services.py --start-date 2024-01-01 --end-date 2024-12-31

  # Download only aFRR (best BESS opportunity)
  python download_ancillary_services.py --products aFRR --start-date 2024-01-01 --end-date 2024-12-31

  # Download aFRR capacity only
  python download_ancillary_services.py --products aFRR --markets CAPACITY --start-date 2024-01-01 --end-date 2024-12-31

  # Download recent data (last 30 days)
  python download_ancillary_services.py --start-date 2024-11-01 --end-date 2024-11-30

  # Download historical data
  python download_ancillary_services.py --start-date 2020-01-01 --end-date 2024-12-31

Product Priority for BESS:
  1. aFRR - HIGHEST opportunity (4-hour blocks, high volatility)
  2. mFRR - Good opportunity
  3. FCR - Good opportunity (but 30-second response time)
        """
    )

    # Product selection
    parser.add_argument('--products', nargs='+',
                       choices=['FCR', 'aFRR', 'mFRR'],
                       default=['FCR', 'aFRR', 'mFRR'],
                       help='Products to download (default: all)')

    # Market selection
    parser.add_argument('--markets', nargs='+',
                       choices=['CAPACITY', 'ENERGY'],
                       default=['CAPACITY', 'ENERGY'],
                       help='Markets to download (default: both)')

    # Date range
    parser.add_argument('--start-date', required=True,
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=True,
                       help='End date (YYYY-MM-DD)')

    # Options
    parser.add_argument('--output-dir', type=str,
                       help='Output directory (default: $ENTSO_E_DATA_DIR/csv_files/de_ancillary_services)')

    args = parser.parse_args()

    # Parse dates
    try:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    except ValueError as e:
        parser.error(f"Invalid date format: {e}")

    # Validate date range
    if end_date < start_date:
        parser.error("End date must be after start date")

    # Calculate days
    days = (end_date - start_date).days + 1

    # Initialize downloader
    downloader = AncillaryServicesDownloader(output_dir=args.output_dir)

    # Print download plan
    print("\n" + "="*80)
    print("DOWNLOAD PLAN")
    print("="*80)
    print(f"Date range: {start_date.date()} to {end_date.date()} ({days} days)")
    print(f"Products: {', '.join(args.products)}")
    print(f"Markets: {', '.join(args.markets)}")
    print(f"Output directory: {downloader.output_dir}")

    # Estimate combinations (skip FCR ENERGY)
    combinations = []
    for product in args.products:
        for market in args.markets:
            if not (product == 'FCR' and market == 'ENERGY'):
                combinations.append((product, market))

    print(f"\nDownloads to perform: {len(combinations)}")
    for product, market in combinations:
        print(f"  - {product} {market}")

    print("\nEstimated time: ~{:.1f} minutes (2 sec/day)".format(days * len(combinations) * 2 / 60))
    print("="*80)

    # Confirm to proceed
    if days > 365:
        response = input(f"\nDownloading {days} days of data. Continue? [y/N]: ")
        if response.lower() != 'y':
            print("Cancelled")
            return

    # Download
    results = downloader.download_all_products(
        args.products,
        args.markets,
        start_date,
        end_date
    )

    # Print summary
    print("\n" + "="*80)
    print("DOWNLOAD SUMMARY")
    print("="*80)
    for (product, market), filepath in results.items():
        status = "✓" if filepath else "✗"
        msg = filepath if filepath else "FAILED"
        print(f"{status} {product:5s} {market:10s} {msg}")


if __name__ == '__main__':
    main()

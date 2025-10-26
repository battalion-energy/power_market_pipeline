#!/usr/bin/env python3
"""
Germany Market Data Updater with Auto-Resume

Unified updater for all German electricity market data:
- ENTSO-E: Day-ahead prices, imbalance prices (from Transparency Platform)
- Regelleistung: FCR, aFRR, mFRR capacity and energy (from Regelleistung.net)

This is the recommended script for daily automated updates of German BESS market data.

Usage:
    # Update all data types
    python update_germany_with_resume.py

    # Update only ENTSO-E data
    python update_germany_with_resume.py --data-sources entso_e

    # Update only ancillary services
    python update_germany_with_resume.py --data-sources regelleistung

    # Dry run to see what would be updated
    python update_germany_with_resume.py --dry-run

    # Force start from specific date
    python update_germany_with_resume.py --start-date 2024-01-01
"""

import os
import sys
import argparse
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Tuple
import pandas as pd

from .entso_e_api_client import ENTSOEAPIClient
from .regelleistung_api_client import RegelleistungAPIClient
from .download_da_prices import DAMPriceDownloader
from .download_imbalance_prices import ImbalancePriceDownloader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Data directory
ENTSO_E_DATA_DIR = Path(os.getenv('ENTSO_E_DATA_DIR', '/pool/ssd8tb/data/iso/ENTSO_E'))

# Data types configuration
DATA_TYPES = {
    # ENTSO-E data
    "da_prices": {
        "dir": ENTSO_E_DATA_DIR / "csv_files/da_prices",
        "pattern": "da_prices_DE_LU_*.csv",
        "description": "Day-Ahead Prices (ENTSO-E)",
        "priority": 1,
        "source": "entso_e"
    },
    "imbalance_prices": {
        "dir": ENTSO_E_DATA_DIR / "csv_files/imbalance_prices",
        "pattern": "imbalance_prices_DE_LU_*.csv",
        "description": "Imbalance Prices / Real-Time (ENTSO-E)",
        "priority": 2,
        "source": "entso_e"
    },

    # Regelleistung data
    "fcr_capacity": {
        "dir": ENTSO_E_DATA_DIR / "csv_files/de_ancillary_services",
        "pattern": "fcr_capacity_de_*.csv",
        "description": "FCR Capacity (Regelleistung)",
        "priority": 3,
        "source": "regelleistung",
        "product": "FCR",
        "market": "CAPACITY"
    },
    "afrr_capacity": {
        "dir": ENTSO_E_DATA_DIR / "csv_files/de_ancillary_services",
        "pattern": "afrr_capacity_de_*.csv",
        "description": "aFRR Capacity (Regelleistung) ⭐ BEST OPPORTUNITY",
        "priority": 4,
        "source": "regelleistung",
        "product": "aFRR",
        "market": "CAPACITY"
    },
    "afrr_energy": {
        "dir": ENTSO_E_DATA_DIR / "csv_files/de_ancillary_services",
        "pattern": "afrr_energy_de_*.csv",
        "description": "aFRR Energy (Regelleistung)",
        "priority": 5,
        "source": "regelleistung",
        "product": "aFRR",
        "market": "ENERGY"
    },
    "mfrr_capacity": {
        "dir": ENTSO_E_DATA_DIR / "csv_files/de_ancillary_services",
        "pattern": "mfrr_capacity_de_*.csv",
        "description": "mFRR Capacity (Regelleistung)",
        "priority": 6,
        "source": "regelleistung",
        "product": "mFRR",
        "market": "CAPACITY"
    },
    "mfrr_energy": {
        "dir": ENTSO_E_DATA_DIR / "csv_files/de_ancillary_services",
        "pattern": "mfrr_energy_de_*.csv",
        "description": "mFRR Energy (Regelleistung)",
        "priority": 7,
        "source": "regelleistung",
        "product": "mFRR",
        "market": "ENERGY"
    },
}


def find_last_date(data_type: str) -> Optional[datetime]:
    """
    Find the most recent date downloaded for a data type.

    Args:
        data_type: Data type key from DATA_TYPES

    Returns:
        Latest date found, or None if no files exist
    """
    config = DATA_TYPES[data_type]
    data_dir = config["dir"]

    if not data_dir.exists():
        logger.warning(f"{data_type}: Directory not found - {data_dir}")
        return None

    # Get all CSV files matching pattern
    csv_files = list(data_dir.glob(config["pattern"]))

    if not csv_files:
        logger.warning(f"{data_type}: No existing files found")
        return None

    # Extract end dates from filenames (second YYYY-MM-DD is the end date)
    dates = []
    for csv_file in csv_files:
        filename = csv_file.stem
        # Find all YYYY-MM-DD patterns
        date_matches = re.findall(r'(\d{4})-(\d{2})-(\d{2})', filename)
        if len(date_matches) >= 2:
            # Take the second date (end date)
            try:
                date_str = f"{date_matches[1][0]}-{date_matches[1][1]}-{date_matches[1][2]}"
                date = datetime.strptime(date_str, '%Y-%m-%d')
                dates.append(date)
            except ValueError:
                continue

    if not dates:
        logger.warning(f"{data_type}: No valid dates found in filenames")
        return None

    latest = max(dates)
    logger.info(f"{data_type}: Latest date = {latest.date()} ({len(dates)} files total)")
    return latest


def update_entso_e_data(data_type: str, start_date: datetime, end_date: datetime,
                        downloader) -> bool:
    """Update ENTSO-E data (DA prices or imbalance prices)."""
    config = DATA_TYPES[data_type]

    logger.info(f"\n{'='*80}")
    logger.info(f"Updating {config['description']}")
    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    logger.info(f"{'='*80}")

    try:
        filepath = downloader.download_and_save(
            'DE_LU',
            start_date,
            end_date,
            chunk_days=90 if data_type == 'da_prices' else 30
        )

        if filepath:
            logger.info(f"✓ Successfully updated {data_type}")
            return True
        else:
            logger.warning(f"⚠ No data available for {data_type}")
            return False

    except Exception as e:
        logger.error(f"✗ Failed to update {data_type}: {str(e)}")
        return False


def update_regelleistung_data(data_type: str, start_date: datetime, end_date: datetime,
                              client: RegelleistungAPIClient) -> bool:
    """Update Regelleistung data (FCR/aFRR/mFRR)."""
    config = DATA_TYPES[data_type]
    product = config['product']
    market = config['market']

    logger.info(f"\n{'='*80}")
    logger.info(f"Updating {config['description']}")
    logger.info(f"Product: {product}, Market: {market}")
    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    logger.info(f"{'='*80}")

    try:
        # Download all days
        all_dfs = client.download_date_range(product, market, start_date, end_date)

        if not all_dfs:
            logger.warning(f"⚠ No data available for {data_type}")
            return False

        # Combine and save
        combined = pd.concat(all_dfs, axis=0, ignore_index=True)

        # Save to CSV
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        filename = f"{product.lower()}_{market.lower()}_de_{start_str}_{end_str}.csv"
        filepath = config['dir'] / filename

        config['dir'].mkdir(parents=True, exist_ok=True)
        combined.to_csv(filepath, index=False)

        logger.info(f"✓ Successfully updated {data_type}")
        logger.info(f"  Saved to: {filepath}")
        logger.info(f"  Records: {len(combined)}")

        return True

    except Exception as e:
        logger.error(f"✗ Failed to update {data_type}: {str(e)}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Update Germany market data with auto-resume',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Update all data types (recommended for daily cron)
  python update_germany_with_resume.py

  # Dry run to see what would be updated
  python update_germany_with_resume.py --dry-run

  # Update only ENTSO-E data
  python update_germany_with_resume.py --data-sources entso_e

  # Update only ancillary services (Regelleistung)
  python update_germany_with_resume.py --data-sources regelleistung

  # Force start from specific date
  python update_germany_with_resume.py --start-date 2024-01-01

  # Update specific data types
  python update_germany_with_resume.py --data-types afrr_capacity afrr_energy

Priority for BESS:
  1. aFRR capacity - HIGHEST opportunity (4-hour blocks, high volatility)
  2. Day-ahead prices - Energy arbitrage
  3. Imbalance prices - Real-time balancing
  4. aFRR energy - Activation prices
  5. FCR, mFRR - Additional opportunities
        """
    )

    # Data source selection
    parser.add_argument('--data-sources', nargs='+',
                       choices=['entso_e', 'regelleistung', 'all'],
                       default=['all'],
                       help='Data sources to update (default: all)')

    # Data type selection
    parser.add_argument('--data-types', nargs='+',
                       choices=list(DATA_TYPES.keys()),
                       help='Specific data types to update (default: all from selected sources)')

    # Date options
    parser.add_argument('--start-date', type=str,
                       help='Force start date (YYYY-MM-DD). Default: auto-resume from last date')
    parser.add_argument('--end-date', type=str,
                       help='End date (YYYY-MM-DD or "yesterday" or "today"). Default: yesterday')
    parser.add_argument('--fallback-days', type=int, default=7,
                       help='Days to look back if no existing data (default: 7)')

    # Options
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be updated without downloading')

    args = parser.parse_args()

    # Determine data types to update
    if args.data_types:
        data_types_to_update = args.data_types
    else:
        # Filter by source
        if 'all' in args.data_sources:
            data_types_to_update = list(DATA_TYPES.keys())
        else:
            data_types_to_update = [
                dt for dt, config in DATA_TYPES.items()
                if config['source'] in args.data_sources
            ]

    # Parse end date
    if args.end_date:
        if args.end_date.lower() == 'yesterday':
            end_date = datetime.now() - timedelta(days=1)
        elif args.end_date.lower() == 'today':
            end_date = datetime.now()
        else:
            try:
                end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
            except ValueError:
                parser.error(f"Invalid end date: {args.end_date}")
    else:
        # Default to yesterday
        end_date = datetime.now() - timedelta(days=1)

    # Parse start date or use auto-resume
    if args.start_date:
        forced_start = True
        try:
            start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
            logger.info(f"Using forced start date: {start_date.date()}")
        except ValueError:
            parser.error(f"Invalid start date: {args.start_date}")
    else:
        forced_start = False
        start_date = None

    # Fallback date
    fallback_date = datetime.now() - timedelta(days=args.fallback_days)

    # Get resume dates if not forcing start
    resume_dates = {}
    if not forced_start:
        logger.info("\nScanning for existing data to determine resume dates...")
        for data_type in data_types_to_update:
            last_date = find_last_date(data_type)
            if last_date:
                resume_dates[data_type] = last_date + timedelta(days=1)
            else:
                resume_dates[data_type] = fallback_date
    else:
        resume_dates = {dt: start_date for dt in data_types_to_update}

    # Print update plan
    print("\n" + "="*80)
    print("GERMANY MARKET DATA UPDATE PLAN")
    print("="*80)
    print(f"End date: {end_date.date()}")
    print(f"Data sources: {', '.join(args.data_sources)}")
    print(f"\nData types and resume dates:")

    for data_type in data_types_to_update:
        resume_date = resume_dates[data_type]
        days = (end_date - resume_date).days + 1
        if days > 0:
            print(f"  {DATA_TYPES[data_type]['description']:45s} {resume_date.date()} -> {end_date.date()} ({days} days)")
        else:
            print(f"  {DATA_TYPES[data_type]['description']:45s} Already up to date")

    if args.dry_run:
        print("\n" + "="*80)
        print("DRY RUN - No data will be downloaded")
        print("="*80)
        return

    print(f"\nTotal updates to perform: {len(data_types_to_update)}")
    print("="*80)

    # Initialize clients
    entso_e_downloaders = {}
    regelleistung_client = None

    if any(DATA_TYPES[dt]['source'] == 'entso_e' for dt in data_types_to_update):
        entso_e_downloaders['da_prices'] = DAMPriceDownloader()
        entso_e_downloaders['imbalance_prices'] = ImbalancePriceDownloader()

    if any(DATA_TYPES[dt]['source'] == 'regelleistung' for dt in data_types_to_update):
        regelleistung_client = RegelleistungAPIClient()

    # Perform updates
    results = {}
    for data_type in data_types_to_update:
        config = DATA_TYPES[data_type]
        resume_date = resume_dates[data_type]

        # Skip if already up to date
        if resume_date > end_date:
            logger.info(f"\nSkipping {data_type} - already up to date")
            results[data_type] = 'skipped'
            continue

        # Update based on source
        if config['source'] == 'entso_e':
            downloader = entso_e_downloaders[data_type]
            success = update_entso_e_data(data_type, resume_date, end_date, downloader)
        else:  # regelleistung
            success = update_regelleistung_data(data_type, resume_date, end_date, regelleistung_client)

        results[data_type] = 'success' if success else 'failed'

    # Print summary
    print("\n" + "="*80)
    print("UPDATE SUMMARY")
    print("="*80)

    successful = sum(1 for v in results.values() if v == 'success')
    failed = sum(1 for v in results.values() if v == 'failed')
    skipped = sum(1 for v in results.values() if v == 'skipped')

    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Skipped (already up to date): {skipped}")

    print("\nDetailed results:")
    for data_type in data_types_to_update:
        result = results.get(data_type, 'unknown')
        icon = {'success': '✓', 'failed': '✗', 'skipped': '⊘'}.get(result, '?')
        print(f"  {icon} {DATA_TYPES[data_type]['description']}: {result}")

    # Exit code
    if failed > 0:
        logger.error(f"\n{failed} updates failed")
        sys.exit(1)
    else:
        logger.info(f"\nAll updates completed successfully")
        sys.exit(0)


if __name__ == '__main__':
    main()

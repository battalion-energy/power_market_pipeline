#!/usr/bin/env python3
"""
ENTSO-E Data Updater with Auto-Resume Capability

Smart updater that:
1. Finds the last downloaded date for each data type and zone
2. Automatically resumes from that point
3. Handles gaps if cron job fails for a few days
4. Perfect for daily cron jobs
5. Supports multiple European zones

Usage:
    # Update Germany (priority focus)
    python update_entso_e_with_resume.py --zones DE_LU

    # Update all priority 1 zones
    python update_entso_e_with_resume.py --priority 1

    # Check what would be updated (dry run)
    python update_entso_e_with_resume.py --zones DE_LU --dry-run

    # Force start date
    python update_entso_e_with_resume.py --zones DE_LU --start-date 2024-01-01

    # Update specific data types only
    python update_entso_e_with_resume.py --zones DE_LU --data-types da_prices imbalance_prices
"""

import os
import sys
import argparse
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import pandas as pd

# Import our modules
from .entso_e_api_client import ENTSOEAPIClient
from .european_zones import BIDDING_ZONES, get_zones_by_priority
from .download_da_prices import DAMPriceDownloader
from .download_imbalance_prices import ImbalancePriceDownloader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Data directory
ENTSO_E_DATA_DIR = Path(os.getenv('ENTSO_E_DATA_DIR', '/pool/ssd8tb/data/iso/ENTSO_E'))

# Data types and their directory structures
DATA_TYPES = {
    "da_prices": {
        "dir": ENTSO_E_DATA_DIR / "csv_files/da_prices",
        "pattern": "da_prices_*_*.csv",  # da_prices_ZONE_YYYY-MM-DD_YYYY-MM-DD.csv
        "description": "Day-Ahead Market Prices",
        "priority": 1,
        "downloader_class": DAMPriceDownloader,
        "chunk_days": 90
    },
    "imbalance_prices": {
        "dir": ENTSO_E_DATA_DIR / "csv_files/imbalance_prices",
        "pattern": "imbalance_prices_*_*.csv",
        "description": "Imbalance Prices (Real-Time)",
        "priority": 2,
        "downloader_class": ImbalancePriceDownloader,
        "chunk_days": 30
    },
    # Future: balancing energy, load, generation, etc.
}


def find_last_date(data_type: str, zone_name: str) -> Optional[datetime]:
    """
    Find the most recent date downloaded for a data type and zone.

    Args:
        data_type: Data type (e.g., 'da_prices', 'imbalance_prices')
        zone_name: Zone name (e.g., 'DE_LU', 'FR')

    Returns:
        Latest date found, or None if no files exist
    """
    config = DATA_TYPES[data_type]
    data_dir = config["dir"]

    if not data_dir.exists():
        logger.warning(f"{zone_name}/{data_type}: Directory not found - {data_dir}")
        return None

    # Get all CSV files for this zone
    # Pattern: da_prices_DE_LU_2024-01-01_2024-01-31.csv
    zone_pattern = f"{data_type.replace('_', '_')}_{zone_name}_*.csv"
    csv_files = list(data_dir.glob(zone_pattern))

    if not csv_files:
        logger.warning(f"{zone_name}/{data_type}: No existing files found")
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
        logger.warning(f"{zone_name}/{data_type}: No valid dates found in filenames")
        return None

    latest = max(dates)
    logger.info(f"{zone_name}/{data_type}: Latest date = {latest.date()} ({len(dates)} files total)")
    return latest


def get_resume_dates(data_types: List[str], zone_names: List[str],
                    fallback_date: datetime) -> Dict[Tuple[str, str], datetime]:
    """
    Get resume dates for each data type and zone combination.

    Args:
        data_types: List of data types to check
        zone_names: List of zones to check
        fallback_date: Date to use if no existing data found

    Returns:
        Dictionary mapping (zone, data_type) -> resume_date
    """
    resume_dates = {}

    for zone_name in zone_names:
        for data_type in data_types:
            last_date = find_last_date(data_type, zone_name)

            if last_date:
                # Resume from next day after last downloaded
                resume_date = last_date + timedelta(days=1)
                resume_dates[(zone_name, data_type)] = resume_date
            else:
                # No existing data, use fallback
                resume_dates[(zone_name, data_type)] = fallback_date

    return resume_dates


def update_zone_data_type(zone_name: str, data_type: str,
                          start_date: datetime, end_date: datetime,
                          downloader) -> bool:
    """
    Update a specific data type for a zone.

    Args:
        zone_name: Zone to update
        data_type: Data type to update
        start_date: Start date
        end_date: End date
        downloader: Downloader instance

    Returns:
        True if successful, False otherwise
    """
    config = DATA_TYPES[data_type]

    logger.info(f"\n{'='*80}")
    logger.info(f"Updating {zone_name} - {config['description']}")
    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    logger.info(f"{'='*80}")

    try:
        filepath = downloader.download_and_save(
            zone_name,
            start_date,
            end_date,
            chunk_days=config['chunk_days']
        )

        if filepath:
            logger.info(f"✓ Successfully updated {zone_name}/{data_type}")
            return True
        else:
            logger.warning(f"⚠ No data available for {zone_name}/{data_type}")
            return False

    except Exception as e:
        logger.error(f"✗ Failed to update {zone_name}/{data_type}: {str(e)}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Update ENTSO-E data with auto-resume capability',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Update Germany with auto-resume
  python update_entso_e_with_resume.py --zones DE_LU

  # Update all priority 1 zones
  python update_entso_e_with_resume.py --priority 1

  # Dry run to see what would be updated
  python update_entso_e_with_resume.py --zones DE_LU FR --dry-run

  # Force start from specific date
  python update_entso_e_with_resume.py --zones DE_LU --start-date 2024-01-01

  # Update only day-ahead prices
  python update_entso_e_with_resume.py --zones DE_LU --data-types da_prices

  # Update until yesterday (good for daily cron)
  python update_entso_e_with_resume.py --priority 1 --end-date yesterday
        """
    )

    # Zone selection
    zone_group = parser.add_mutually_exclusive_group(required=True)
    zone_group.add_argument('--zones', nargs='+', help='Specific zone names (e.g., DE_LU FR NL)')
    zone_group.add_argument('--priority', type=int, choices=[1, 2, 3],
                           help='Update all zones with this priority level')

    # Date options
    parser.add_argument('--start-date', type=str,
                       help='Force start date (YYYY-MM-DD). Default: auto-resume from last date')
    parser.add_argument('--end-date', type=str,
                       help='End date (YYYY-MM-DD or "yesterday" or "today"). Default: yesterday')
    parser.add_argument('--fallback-days', type=int, default=7,
                       help='Days to look back if no existing data (default: 7)')

    # Data type selection
    parser.add_argument('--data-types', nargs='+', choices=list(DATA_TYPES.keys()),
                       help='Specific data types to update (default: all)')

    # Options
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be updated without downloading')
    parser.add_argument('--api-key', type=str,
                       help='ENTSO-E API key (default: $ENTSO_E_API_KEY)')

    args = parser.parse_args()

    # Determine zones
    if args.zones:
        zone_names = args.zones
    else:  # priority
        zone_names = [name for name, zone in BIDDING_ZONES.items() if zone.priority == args.priority]
        logger.info(f"Selected {len(zone_names)} priority {args.priority} zones: {', '.join(zone_names)}")

    # Validate zones
    invalid_zones = [z for z in zone_names if z not in BIDDING_ZONES]
    if invalid_zones:
        parser.error(f"Invalid zone names: {invalid_zones}")

    # Determine data types
    data_types_to_update = args.data_types if args.data_types else list(DATA_TYPES.keys())

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
        # Default to yesterday (market data typically available next day)
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
        start_date = None  # Will be determined per zone/data_type

    # Fallback date for zones with no existing data
    fallback_date = datetime.now() - timedelta(days=args.fallback_days)

    # Get resume dates if not forcing start
    if not forced_start:
        logger.info("\nScanning for existing data to determine resume dates...")
        resume_dates = get_resume_dates(data_types_to_update, zone_names, fallback_date)
    else:
        resume_dates = {(zone, dt): start_date for zone in zone_names for dt in data_types_to_update}

    # Print update plan
    print("\n" + "="*80)
    print("UPDATE PLAN")
    print("="*80)
    print(f"End date: {end_date.date()}")
    print(f"Data types: {', '.join(data_types_to_update)}")
    print(f"\nZones and resume dates:")

    for zone_name in zone_names:
        print(f"\n  {zone_name} ({BIDDING_ZONES[zone_name].name}):")
        for data_type in data_types_to_update:
            resume_date = resume_dates[(zone_name, data_type)]
            days = (end_date - resume_date).days + 1
            print(f"    {DATA_TYPES[data_type]['description']:30s} {resume_date.date()} -> {end_date.date()} ({days} days)")

    if args.dry_run:
        print("\n" + "="*80)
        print("DRY RUN - No data will be downloaded")
        print("="*80)
        return

    # Confirm to proceed
    total_updates = len(zone_names) * len(data_types_to_update)
    print(f"\nTotal updates to perform: {total_updates}")
    print("="*80)

    # Initialize downloaders
    downloaders = {}
    for data_type in data_types_to_update:
        config = DATA_TYPES[data_type]
        downloaders[data_type] = config['downloader_class'](api_key=args.api_key)

    # Perform updates
    results = {}
    for zone_name in zone_names:
        for data_type in data_types_to_update:
            key = (zone_name, data_type)
            resume_date = resume_dates[key]

            # Skip if resume date is after end date
            if resume_date > end_date:
                logger.info(f"Skipping {zone_name}/{data_type} - already up to date")
                results[key] = 'skipped'
                continue

            success = update_zone_data_type(
                zone_name,
                data_type,
                resume_date,
                end_date,
                downloaders[data_type]
            )

            results[key] = 'success' if success else 'failed'

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

    # Detailed results
    print("\nDetailed results:")
    for zone_name in zone_names:
        print(f"\n  {zone_name}:")
        for data_type in data_types_to_update:
            key = (zone_name, data_type)
            result = results.get(key, 'unknown')
            icon = {'success': '✓', 'failed': '✗', 'skipped': '⊘'}.get(result, '?')
            print(f"    {icon} {DATA_TYPES[data_type]['description']}: {result}")

    # Exit code
    if failed > 0:
        logger.error(f"\n{failed} updates failed")
        sys.exit(1)
    else:
        logger.info(f"\nAll updates completed successfully")
        sys.exit(0)


if __name__ == '__main__':
    main()

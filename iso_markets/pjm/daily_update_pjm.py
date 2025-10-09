#!/usr/bin/env python3
"""
PJM Daily Update Script

Downloads yesterday's data for all market types:
- Day-ahead hub LMPs
- Real-time hourly hub LMPs
- Day-ahead ancillary services
- (Future: frequency regulation, RT 5-min)

Run 3x daily via cron: 8am, 2pm, 8pm

Usage:
    python daily_update_pjm.py
    python daily_update_pjm.py --date 2025-10-06
    python daily_update_pjm.py --days-back 7
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))
from pjm_api_client import PJMAPIClient

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Hub list
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


def update_da_hub_lmps(client: PJMAPIClient, target_date: str, data_dir: Path):
    """Update day-ahead hub LMP prices for target date."""
    logger.info(f"Updating DA hub LMPs for {target_date}")

    output_dir = data_dir / 'csv_files' / 'da_hubs'
    output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    for hub_name, pnode_id in PJM_HUBS.items():
        try:
            data = client.get_day_ahead_lmps(
                start_date=target_date,
                end_date=target_date,
                pnode_id=str(pnode_id)
            )

            if not data:
                logger.warning(f"  No data for {hub_name}")
                continue

            # Extract items
            if isinstance(data, dict):
                items = data.get('items', data.get('data', [data]))
            else:
                items = data

            if not items:
                logger.warning(f"  No items for {hub_name}")
                continue

            # Save to daily file
            df = pd.DataFrame(items)
            filename = f"{hub_name}_DA_{target_date}.csv"
            output_path = output_dir / filename
            df.to_csv(output_path, index=False)

            logger.info(f"  ✓ {hub_name}: {len(df)} records")
            success_count += 1

        except Exception as e:
            logger.error(f"  ✗ {hub_name}: {e}")

    logger.info(f"DA LMP: {success_count}/{len(PJM_HUBS)} hubs updated")


def update_rt_hourly_lmps(client: PJMAPIClient, target_date: str, data_dir: Path):
    """Update real-time hourly LMP prices for target date."""
    logger.info(f"Updating RT hourly LMPs for {target_date}")

    output_dir = data_dir / 'csv_files' / 'rt_hourly'
    output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    for hub_name, pnode_id in PJM_HUBS.items():
        try:
            data = client.get_rt_hourly_lmps(
                start_date=target_date,
                end_date=target_date,
                pnode_id=str(pnode_id)
            )

            if not data:
                logger.warning(f"  No data for {hub_name}")
                continue

            # Extract items
            if isinstance(data, dict):
                items = data.get('items', data.get('data', [data]))
            else:
                items = data

            if not items:
                logger.warning(f"  No items for {hub_name}")
                continue

            # Save to daily file
            df = pd.DataFrame(items)
            filename = f"{hub_name}_RT_HOURLY_{target_date}.csv"
            output_path = output_dir / filename
            df.to_csv(output_path, index=False)

            logger.info(f"  ✓ {hub_name}: {len(df)} records")
            success_count += 1

        except Exception as e:
            logger.error(f"  ✗ {hub_name}: {e}")

    logger.info(f"RT Hourly: {success_count}/{len(PJM_HUBS)} hubs updated")


def update_ancillary_services(client: PJMAPIClient, target_date: str, data_dir: Path):
    """Update day-ahead ancillary services for target date."""
    logger.info(f"Updating DA ancillary services for {target_date}")

    output_dir = data_dir / 'csv_files' / 'da_ancillary_services'
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        data = client.get_ancillary_services(
            start_date=target_date,
            end_date=target_date
        )

        if not data:
            logger.warning("  No ancillary services data")
            return

        # Extract items
        if isinstance(data, dict):
            items = data.get('items', data.get('data', [data]))
        else:
            items = data

        if not items:
            logger.warning("  No ancillary services items")
            return

        # Save to daily file
        df = pd.DataFrame(items)
        filename = f"ancillary_services_{target_date}.csv"
        output_path = output_dir / filename
        df.to_csv(output_path, index=False)

        logger.info(f"  ✓ Ancillary services: {len(df)} records")

    except Exception as e:
        logger.error(f"  ✗ Ancillary services: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='PJM Daily Update - Download yesterday\'s data for all markets'
    )
    parser.add_argument(
        '--date',
        type=str,
        help='Specific date to download (YYYY-MM-DD). Default: yesterday'
    )
    parser.add_argument(
        '--days-back',
        type=int,
        help='Number of days back to download (default: 1)'
    )
    parser.add_argument(
        '--skip-da',
        action='store_true',
        help='Skip day-ahead LMP updates'
    )
    parser.add_argument(
        '--skip-rt',
        action='store_true',
        help='Skip real-time hourly updates'
    )
    parser.add_argument(
        '--skip-as',
        action='store_true',
        help='Skip ancillary services updates'
    )

    args = parser.parse_args()

    # Determine target date(s)
    if args.date:
        target_date = datetime.strptime(args.date, '%Y-%m-%d')
        dates_to_process = [target_date]
    elif args.days_back:
        today = datetime.now()
        dates_to_process = [today - timedelta(days=i) for i in range(1, args.days_back + 1)]
    else:
        # Default: yesterday
        yesterday = datetime.now() - timedelta(days=1)
        dates_to_process = [yesterday]

    # Data directory
    data_dir = Path(os.getenv('PJM_DATA_DIR', '/home/enrico/data/PJM_data'))

    # Initialize API client
    try:
        client = PJMAPIClient(requests_per_minute=6)
    except ValueError as e:
        logger.error(f"Failed to initialize API client: {e}")
        logger.error("Please set PJM_API_KEY in your .env file")
        return

    # Process each date
    for target_date in dates_to_process:
        date_str = target_date.strftime('%Y-%m-%d')

        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {date_str}")
        logger.info(f"{'='*60}\n")

        try:
            # Update each market type
            if not args.skip_da:
                update_da_hub_lmps(client, date_str, data_dir)

            if not args.skip_rt:
                update_rt_hourly_lmps(client, date_str, data_dir)

            if not args.skip_as:
                update_ancillary_services(client, date_str, data_dir)

        except Exception as e:
            logger.error(f"Error processing {date_str}: {e}")

    logger.info(f"\n✓ Daily update complete!")
    logger.info(f"Processed {len(dates_to_process)} date(s)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
CAISO Data Updater with Auto-Resume
Downloads and updates CAISO data with automatic resume from last downloaded date.

Data Types:
- Day-Ahead Nodal LMPs (hourly)
- Real-Time 5-Minute Nodal LMPs
- Day-Ahead Ancillary Services (RU, RD, SR, NR)
"""

import os
import sys
import logging
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
from dotenv import load_dotenv

from iso_markets.caiso.caiso_api_client import CAISOAPIClient, format_datetime_for_caiso

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
CAISO_DATA_DIR = Path(os.getenv('CAISO_DATA_DIR', '/pool/ssd8tb/data/iso/CAISO_data'))

# Data type configurations
DATA_TYPES = {
    "da_nodal": {
        "dir": CAISO_DATA_DIR / "csv_files/da_nodal",
        "pattern": "nodal_da_lmp_*.csv",
        "description": "Day-Ahead Nodal LMPs",
        "priority": 1
    },
    "rt_5min_nodal": {
        "dir": CAISO_DATA_DIR / "csv_files/rt_5min_nodal",
        "pattern": "nodal_rt_5min_lmp_*.csv",
        "description": "Real-Time 5-Min Nodal LMPs",
        "priority": 2
    },
    "ancillary_services": {
        "dir": CAISO_DATA_DIR / "csv_files/da_ancillary_services",
        "pattern": "ancillary_services_*.csv",
        "description": "DA Ancillary Services",
        "priority": 3
    }
}

# Retry configuration
MAX_RETRIES = 5
BASE_RETRY_DELAY = 30  # seconds


def get_latest_date(data_type: str) -> Optional[datetime]:
    """Get the latest downloaded date for a data type."""
    config = DATA_TYPES[data_type]
    data_dir = config["dir"]
    pattern = config["pattern"]

    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        return None

    existing_files = list(data_dir.glob(pattern))

    if not existing_files:
        return None

    # Extract dates from filenames
    dates = []
    for f in existing_files:
        try:
            # Remove prefix and .csv suffix
            date_str = f.stem
            for prefix in ["nodal_da_lmp_", "nodal_rt_5min_lmp_", "ancillary_services_"]:
                date_str = date_str.replace(prefix, "")
            dates.append(datetime.strptime(date_str, "%Y-%m-%d"))
        except ValueError:
            continue

    if not dates:
        return None

    return max(dates)


def update_da_nodal(client: CAISOAPIClient, start_date: datetime, end_date: datetime, dry_run: bool = False):
    """Update day-ahead nodal LMP data."""
    data_type = "da_nodal"
    output_dir = DATA_TYPES[data_type]["dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    current_date = start_date
    success_count = 0
    fail_count = 0

    while current_date <= end_date:
        if dry_run:
            current_date += timedelta(days=1)
            continue

        date_str = current_date.strftime("%Y-%m-%d")
        output_file = output_dir / f"nodal_da_lmp_{date_str}.csv"

        # Quick skip check
        if output_file.exists() and output_file.stat().st_size > 1000:
            logger.info(f"✓ {date_str}: DA nodal already exists - skipping")
            success_count += 1
            current_date += timedelta(days=1)
            continue

        logger.info(f"⏳ Downloading DA nodal for {date_str}...")

        start_dt = current_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_dt = current_date.replace(hour=23, minute=59, second=0, microsecond=0)
        start_str = format_datetime_for_caiso(start_dt)
        end_str = format_datetime_for_caiso(end_dt)

        # Retry loop
        day_success = False
        for attempt in range(MAX_RETRIES):
            try:
                df = client.get_day_ahead_lmps(start_str, end_str, node="ALL")
                if df is not None and len(df) > 0:
                    df.to_csv(output_file, index=False)
                    logger.info(f"✓ {date_str}: Saved {len(df):,} rows")
                    day_success = True
                    break
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    wait_time = BASE_RETRY_DELAY * (2 ** attempt)
                    logger.warning(f"⚠️  Attempt {attempt + 1}/{MAX_RETRIES} failed: {str(e)}")
                    logger.warning(f"⏳ Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"❌ Failed: {str(e)}")

        if day_success:
            success_count += 1
        else:
            fail_count += 1

        current_date += timedelta(days=1)

    return success_count, fail_count


def update_rt_5min_nodal(client: CAISOAPIClient, start_date: datetime, end_date: datetime, dry_run: bool = False):
    """Update real-time 5-minute nodal LMP data."""
    data_type = "rt_5min_nodal"
    output_dir = DATA_TYPES[data_type]["dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    current_date = start_date
    success_count = 0
    fail_count = 0

    while current_date <= end_date:
        if dry_run:
            current_date += timedelta(days=1)
            continue

        date_str = current_date.strftime("%Y-%m-%d")
        output_file = output_dir / f"nodal_rt_5min_lmp_{date_str}.csv"

        # Quick skip check
        if output_file.exists() and output_file.stat().st_size > 10000:
            logger.info(f"✓ {date_str}: RT 5-min nodal already exists - skipping")
            success_count += 1
            current_date += timedelta(days=1)
            continue

        logger.info(f"⏳ Downloading RT 5-min nodal for {date_str}...")

        all_dfs = []
        day_success = True

        # Download in 2-hour chunks
        for hour_start in range(0, 24, 2):
            hour_end = min(hour_start + 2, 24)
            start_dt = current_date.replace(hour=hour_start, minute=0, second=0, microsecond=0)
            end_dt = current_date.replace(hour=23 if hour_end == 24 else hour_end, minute=59 if hour_end == 24 else 0, second=0, microsecond=0)

            start_str = format_datetime_for_caiso(start_dt)
            end_str = format_datetime_for_caiso(end_dt)

            # Retry loop for this chunk
            chunk_success = False
            for attempt in range(MAX_RETRIES):
                try:
                    df = client.get_rt_5min_lmps(start_str, end_str, node="ALL")
                    if df is not None and len(df) > 0:
                        all_dfs.append(df)
                    chunk_success = True
                    break
                except Exception as e:
                    if attempt < MAX_RETRIES - 1:
                        wait_time = BASE_RETRY_DELAY * (2 ** attempt)
                        logger.warning(f"⚠️  Chunk {hour_start:02d}:00 attempt {attempt + 1}/{MAX_RETRIES} failed")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"❌ Chunk {hour_start:02d}:00 failed: {str(e)}")
                        chunk_success = False

            if not chunk_success:
                day_success = False
                break

        if day_success and all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            combined_df.to_csv(output_file, index=False)
            logger.info(f"✓ {date_str}: Saved {len(combined_df):,} rows")
            success_count += 1
        else:
            fail_count += 1

        current_date += timedelta(days=1)

    return success_count, fail_count


def update_ancillary_services(client: CAISOAPIClient, start_date: datetime, end_date: datetime, dry_run: bool = False):
    """Update ancillary services data."""
    data_type = "ancillary_services"
    output_dir = DATA_TYPES[data_type]["dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    current_date = start_date
    success_count = 0
    fail_count = 0

    while current_date <= end_date:
        if dry_run:
            current_date += timedelta(days=1)
            continue

        date_str = current_date.strftime("%Y-%m-%d")
        output_file = output_dir / f"ancillary_services_{date_str}.csv"

        # Quick skip check
        if output_file.exists() and output_file.stat().st_size > 500:
            logger.info(f"✓ {date_str}: AS already exists - skipping")
            success_count += 1
            current_date += timedelta(days=1)
            continue

        logger.info(f"⏳ Downloading AS for {date_str}...")

        start_dt = current_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_dt = current_date.replace(hour=23, minute=59, second=0, microsecond=0)
        start_str = format_datetime_for_caiso(start_dt)
        end_str = format_datetime_for_caiso(end_dt)

        # Retry loop
        day_success = False
        for attempt in range(MAX_RETRIES):
            try:
                df = client.get_ancillary_services(start_str, end_str)
                if df is not None and len(df) > 0:
                    df.to_csv(output_file, index=False)
                    logger.info(f"✓ {date_str}: Saved {len(df):,} rows")
                    day_success = True
                    break
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    wait_time = BASE_RETRY_DELAY * (2 ** attempt)
                    logger.warning(f"⚠️  Attempt {attempt + 1}/{MAX_RETRIES} failed: {str(e)}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"❌ Failed: {str(e)}")

        if day_success:
            success_count += 1
        else:
            fail_count += 1

        current_date += timedelta(days=1)

    return success_count, fail_count


def main():
    """Main updater."""
    parser = argparse.ArgumentParser(description="CAISO Data Updater with Auto-Resume")
    parser.add_argument("--start-date", type=str, help="Force start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="Force end date (YYYY-MM-DD)")
    parser.add_argument("--data-types", nargs="+", choices=list(DATA_TYPES.keys()),
                       help="Specific data types to update")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be updated without downloading")

    args = parser.parse_args()

    print("=" * 80)
    print("CAISO Data Updater with Auto-Resume")
    print("=" * 80)

    # Determine which data types to update
    data_types_to_update = args.data_types if args.data_types else list(DATA_TYPES.keys())

    # Check existing data
    logger.info("Checking existing data to determine resume points...")
    print("")

    resume_plan = []
    for dt in data_types_to_update:
        latest = get_latest_date(dt)
        file_count = len(list(DATA_TYPES[dt]["dir"].glob(DATA_TYPES[dt]["pattern"]))) if DATA_TYPES[dt]["dir"].exists() else 0

        if latest:
            logger.info(f"{dt}: Latest date = {latest.strftime('%Y-%m-%d')} ({file_count} files total)")
            start = latest + timedelta(days=1)
        else:
            logger.info(f"{dt}: Latest date = None ({file_count} files total)")
            start = datetime(2019, 1, 1)

        resume_plan.append((dt, start))

    # Determine end date
    if args.end_date:
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    else:
        end_date = datetime.now() - timedelta(days=1)
        end_date = end_date.replace(hour=0, minute=0, second=0, microsecond=0)

    # Show resume plan
    print("\n" + "=" * 80)
    print("Resume Plan:")
    print("=" * 80)

    for dt, start in resume_plan:
        if args.start_date:
            start = datetime.strptime(args.start_date, "%Y-%m-%d")

        days = (end_date - start).days + 1 if start <= end_date else 0
        desc = DATA_TYPES[dt]["description"]
        print(f"{desc:35} {start.strftime('%Y-%m-%d')} -> {end_date.strftime('%Y-%m-%d')} ({days} days)")

    if args.dry_run:
        print("\n[DRY RUN] Would update {} data type(s)".format(len(data_types_to_update)))
        return

    # Create client
    client = CAISOAPIClient(min_delay_between_requests=5.0)

    # Update each data type
    print("\n" + "=" * 80)
    print("Starting Downloads")
    print("=" * 80 + "\n")

    results = {}

    for dt, start in resume_plan:
        if args.start_date:
            start = datetime.strptime(args.start_date, "%Y-%m-%d")

        if start > end_date:
            logger.info(f"✓ {dt}: Already up to date!")
            continue

        logger.info(f"\n{'=' * 80}")
        logger.info(f"Updating {DATA_TYPES[dt]['description']}")
        logger.info(f"{'=' * 80}")

        if dt == "da_nodal":
            success, fail = update_da_nodal(client, start, end_date, args.dry_run)
        elif dt == "rt_5min_nodal":
            success, fail = update_rt_5min_nodal(client, start, end_date, args.dry_run)
        elif dt == "ancillary_services":
            success, fail = update_ancillary_services(client, start, end_date, args.dry_run)

        results[dt] = {"success": success, "fail": fail}

    # Final summary
    print("\n" + "=" * 80)
    print("Update Summary")
    print("=" * 80)

    total_success = sum(r["success"] for r in results.values())
    total_fail = sum(r["fail"] for r in results.values())

    for dt in data_types_to_update:
        if dt in results:
            desc = DATA_TYPES[dt]["description"]
            r = results[dt]
            print(f"{desc:35} Success: {r['success']:4d}  Failed: {r['fail']:4d}")

    print("=" * 80)
    print(f"Total:                              Success: {total_success:4d}  Failed: {total_fail:4d}")
    print("=" * 80)

    if total_fail > 0:
        sys.exit(1)
    else:
        logger.info("✓ All data updated successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()

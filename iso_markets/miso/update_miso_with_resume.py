#!/usr/bin/env python3
"""
MISO Data Updater with Auto-Resume Capability

Smart updater that:
1. Finds the last downloaded date for each data type
2. Automatically resumes from that point
3. Handles gaps if cron job fails for a few days
4. Perfect for daily cron jobs

Usage:
    python update_miso_with_resume.py                    # Auto-resume all data types
    python update_miso_with_resume.py --dry-run          # Check what would be updated
    python update_miso_with_resume.py --start-date 2024-01-01  # Force start date
"""

import asyncio
import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Data directory
MISO_DATA_DIR = Path("/pool/ssd8tb/data/iso/MISO")

# Data types and their directory structures
DATA_TYPES = {
    "lmp_da_expost": {
        "dir": MISO_DATA_DIR / "csv_files/da_expost",
        "pattern": "*_da_expost_lmp.csv",
        "description": "Day-Ahead Ex-Post LMP"
    },
    "lmp_rt_final": {
        "dir": MISO_DATA_DIR / "csv_files/rt_final",
        "pattern": "*_rt_lmp_final.csv",
        "description": "Real-Time Final LMP"
    },
    "rt_5min": {
        "dir": MISO_DATA_DIR / "csv_files/rt_5min",
        "pattern": "*.csv",
        "description": "Real-Time 5-min LMP"
    },
    "ancillary_da": {
        "dir": MISO_DATA_DIR / "csv_files/asm_da_expost",
        "pattern": "*_asm_expost_damcp.csv",
        "description": "Ancillary Services DA"
    },
    "ancillary_rt": {
        "dir": MISO_DATA_DIR / "csv_files/asm_rt_final",
        "pattern": "*_asm_rtmcp_final.csv",
        "description": "Ancillary Services RT"
    },
    "load": {
        "dir": MISO_DATA_DIR / "csv_files/load",
        "pattern": "*.csv",
        "description": "Load Data"
    },
    "fuel_mix": {
        "dir": MISO_DATA_DIR / "eia_fuel_mix",
        "pattern": "eia_fuel_mix_*_*_pivot.csv",
        "description": "EIA Fuel Mix"
    }
}


def find_last_date(data_type: str) -> Optional[datetime]:
    """Find the most recent date downloaded for a data type."""
    config = DATA_TYPES[data_type]
    data_dir = config["dir"]

    if not data_dir.exists():
        logger.warning(f"{data_type}: Directory not found - {data_dir}")
        return None

    # Get all CSV files
    csv_files = list(data_dir.glob(config["pattern"]))

    if not csv_files:
        logger.warning(f"{data_type}: No existing files found")
        return None

    # Extract dates from filenames (YYYYMMDD pattern)
    dates = []
    for csv_file in csv_files:
        filename = csv_file.stem
        # Try to find YYYYMMDD pattern in filename
        import re
        date_match = re.search(r'(\d{8})', filename)
        if date_match:
            try:
                date_str = date_match.group(1)
                date = datetime.strptime(date_str, '%Y%m%d')
                dates.append(date)
            except ValueError:
                continue
        else:
            # Try YYYY-MM-DD pattern (for some files)
            date_match = re.search(r'(\d{4})-(\d{2})-(\d{2})', filename)
            if date_match:
                try:
                    date_str = f"{date_match.group(1)}{date_match.group(2)}{date_match.group(3)}"
                    date = datetime.strptime(date_str, '%Y%m%d')
                    dates.append(date)
                except ValueError:
                    continue

    if not dates:
        logger.warning(f"{data_type}: No valid dates found in filenames")
        return None

    latest = max(dates)
    logger.info(f"{data_type}: Latest date = {latest.date()} ({len(dates)} files total)")
    return latest


def get_resume_date(data_types: List[str], fallback_date: datetime) -> Dict[str, datetime]:
    """Get resume dates for each data type."""
    resume_dates = {}

    for data_type in data_types:
        last_date = find_last_date(data_type)

        if last_date:
            # Resume from next day after last downloaded
            resume_date = last_date + timedelta(days=1)
            resume_dates[data_type] = resume_date
        else:
            # No existing data, use fallback
            resume_dates[data_type] = fallback_date

    return resume_dates


async def update_lmp_data(start_date: datetime, end_date: datetime, market_types: List[str]):
    """Update LMP data."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Updating LMP Data: {start_date.date()} to {end_date.date()}")
    logger.info(f"{'='*80}")

    # Import here to avoid circular dependencies
    from .download_historical_lmp import MISOLMPDownloader

    async with MISOLMPDownloader() as downloader:
        await downloader.download_date_range(
            start_date=start_date,
            end_date=end_date,
            market_types=market_types,
            filter_hubs=True,
            max_concurrent=5
        )


async def update_rt_5min_data(start_date: datetime, end_date: datetime):
    """Update RT 5-min data."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Updating RT 5-min Data: {start_date.date()} to {end_date.date()}")
    logger.info(f"{'='*80}")

    from .download_rt_5min_lmp import MISO5MinLMPDownloader

    async with MISO5MinLMPDownloader() as downloader:
        await downloader.download_date_range(
            start_date=start_date,
            end_date=end_date,
            filter_hubs=True,
            max_concurrent=3
        )


async def update_ancillary_data(start_date: datetime, end_date: datetime):
    """Update ancillary services data."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Updating Ancillary Services: {start_date.date()} to {end_date.date()}")
    logger.info(f"{'='*80}")

    from .download_ancillary_services import MISOAncillaryServicesDownloader

    async with MISOAncillaryServicesDownloader() as downloader:
        await downloader.download_date_range(
            start_date=start_date,
            end_date=end_date,
            report_types=["da_expost_asm", "rt_final_asm"]
        )


async def update_load_data(start_date: datetime, end_date: datetime):
    """Update load data."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Updating Load Data: {start_date.date()} to {end_date.date()}")
    logger.info(f"{'='*80}")

    from .download_load_data import MISOLoadDownloader

    async with MISOLoadDownloader() as downloader:
        await downloader.download_date_range(
            start_date=start_date,
            end_date=end_date
        )


async def update_fuel_mix_data(start_date: datetime, end_date: datetime):
    """Update EIA fuel mix data."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Updating EIA Fuel Mix: {start_date.date()} to {end_date.date()}")
    logger.info(f"{'='*80}")

    from .download_eia_fuel_mix import EIAFuelMixDownloader

    downloader = EIAFuelMixDownloader()
    async with downloader:
        df = await downloader.download_date_range(start_date, end_date)
        if not df.empty:
            await downloader.save_data(df, start_date, end_date)


async def main():
    parser = argparse.ArgumentParser(
        description="MISO data updater with auto-resume capability"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Force start date (YYYY-MM-DD), otherwise auto-resume from last date"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date (YYYY-MM-DD), defaults to today"
    )
    parser.add_argument(
        "--data-types",
        nargs="+",
        choices=list(DATA_TYPES.keys()),
        help="Specific data types to update (default: all)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without downloading"
    )
    parser.add_argument(
        "--fallback-days",
        type=int,
        default=7,
        help="If no existing data, download last N days (default: 7)"
    )

    args = parser.parse_args()

    # Determine data types to update
    if args.data_types:
        data_types = args.data_types
    else:
        data_types = list(DATA_TYPES.keys())

    # Parse end date
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d") if args.end_date else datetime.now()

    # Get resume dates
    logger.info(f"\n{'='*80}")
    logger.info(f"MISO Data Updater with Auto-Resume")
    logger.info(f"{'='*80}")
    logger.info(f"Checking existing data to determine resume points...\n")

    fallback_date = end_date - timedelta(days=args.fallback_days)

    if args.start_date:
        # Force start date for all types
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        resume_dates = {dt: start_date for dt in data_types}
        logger.info(f"Using forced start date: {start_date.date()}")
    else:
        # Auto-resume
        resume_dates = get_resume_date(data_types, fallback_date)

    # Show resume plan
    logger.info(f"\n{'='*80}")
    logger.info(f"Resume Plan:")
    logger.info(f"{'='*80}")
    for data_type in data_types:
        start = resume_dates[data_type]
        days = (end_date - start).days + 1
        logger.info(f"{DATA_TYPES[data_type]['description']:30} {start.date()} -> {end_date.date()} ({days} days)")

    if args.dry_run:
        logger.info(f"\n[DRY RUN] Would update {len(data_types)} data types")
        return 0

    # Update each data type
    logger.info(f"\n{'='*80}")
    logger.info(f"Starting Updates...")
    logger.info(f"{'='*80}\n")

    success_count = 0

    try:
        # Update LMP data (DA and RT)
        lmp_types_to_update = [dt for dt in data_types if dt.startswith("lmp_")]
        if lmp_types_to_update:
            earliest_lmp = min(resume_dates[dt] for dt in lmp_types_to_update)
            market_types = []
            if "lmp_da_expost" in lmp_types_to_update:
                market_types.append("da_expost")
            if "lmp_rt_final" in lmp_types_to_update:
                market_types.append("rt_final")

            if market_types:
                await update_lmp_data(earliest_lmp, end_date, market_types)
                success_count += len(lmp_types_to_update)

        # Update RT 5-min
        if "rt_5min" in data_types:
            await update_rt_5min_data(resume_dates["rt_5min"], end_date)
            success_count += 1

        # Update Ancillary Services
        as_types = [dt for dt in data_types if dt.startswith("ancillary_")]
        if as_types:
            earliest_as = min(resume_dates[dt] for dt in as_types)
            await update_ancillary_data(earliest_as, end_date)
            success_count += len(as_types)

        # Update Load
        if "load" in data_types:
            await update_load_data(resume_dates["load"], end_date)
            success_count += 1

        # Update Fuel Mix
        if "fuel_mix" in data_types:
            await update_fuel_mix_data(resume_dates["fuel_mix"], end_date)
            success_count += 1

    except Exception as e:
        logger.error(f"Update failed: {e}", exc_info=True)
        return 1

    # Summary
    logger.info(f"\n{'='*80}")
    logger.info(f"Update Summary: {success_count}/{len(data_types)} data types updated successfully")
    logger.info(f"{'='*80}")

    return 0 if success_count == len(data_types) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

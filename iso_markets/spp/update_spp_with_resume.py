#!/usr/bin/env python3
"""
SPP Data Updater with Auto-Resume Capability

Smart updater that:
1. Finds the last downloaded date for each data type
2. Automatically resumes from that point
3. Handles gaps if cron job fails for a few days
4. Perfect for daily cron jobs

Usage:
    python update_spp_with_resume.py                    # Auto-resume all data types
    python update_spp_with_resume.py --dry-run          # Check what would be updated
    python update_spp_with_resume.py --start-date 2024-01-01  # Force start date
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
SPP_DATA_DIR = Path("/pool/ssd8tb/data/iso/SPP")

# Data types and their directory structures
DATA_TYPES = {
    "lmp_da": {
        "dir": SPP_DATA_DIR / "csv_files/da_lmp",
        "pattern": "SPP_da_lmp_*.csv",
        "description": "Day-Ahead LMP"
    },
    "lmp_rt_daily": {
        "dir": SPP_DATA_DIR / "csv_files/rt_lmp_daily",
        "pattern": "SPP_rt_lmp_daily_*.csv",
        "description": "Real-Time LMP (daily)"
    },
    "as_da_mcp": {
        "dir": SPP_DATA_DIR / "csv_files/da_mcp",
        "pattern": "SPP_da_mcp_*.csv",
        "description": "Ancillary Services DA MCP"
    },
    "as_rt_mcp_daily": {
        "dir": SPP_DATA_DIR / "csv_files/rt_mcp_daily",
        "pattern": "SPP_rt_mcp_daily_*.csv",
        "description": "Ancillary Services RT MCP (daily)"
    },
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
    from download_lmp import SPPLMPDownloader

    async with SPPLMPDownloader() as downloader:
        await downloader.download_date_range(
            start_date=start_date,
            end_date=end_date,
            market_types=market_types,
            filter_hubs=True,
            max_concurrent=5
        )


async def update_ancillary_data(start_date: datetime, end_date: datetime, market_types: List[str]):
    """Update ancillary services data."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Updating Ancillary Services: {start_date.date()} to {end_date.date()}")
    logger.info(f"{'='*80}")

    from download_ancillary_services import SPPAncillaryServicesDownloader

    async with SPPAncillaryServicesDownloader() as downloader:
        await downloader.download_date_range(
            start_date=start_date,
            end_date=end_date,
            market_types=market_types,
            max_concurrent=5
        )


async def main():
    parser = argparse.ArgumentParser(
        description="SPP data updater with auto-resume capability"
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
    logger.info(f"SPP Data Updater with Auto-Resume")
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
            if "lmp_da" in lmp_types_to_update:
                market_types.append("da")
            if "lmp_rt_daily" in lmp_types_to_update:
                market_types.append("rt_daily")

            if market_types:
                await update_lmp_data(earliest_lmp, end_date, market_types)
                success_count += len(lmp_types_to_update)

        # Update Ancillary Services
        as_types = [dt for dt in data_types if dt.startswith("as_")]
        if as_types:
            earliest_as = min(resume_dates[dt] for dt in as_types)
            market_types = []
            if "as_da_mcp" in as_types:
                market_types.append("da_mcp")
            if "as_rt_mcp_daily" in as_types:
                market_types.append("rt_mcp_daily")

            if market_types:
                await update_ancillary_data(earliest_as, end_date, market_types)
                success_count += len(as_types)

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

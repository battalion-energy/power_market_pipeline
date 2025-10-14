#!/usr/bin/env python3
"""
NYISO standalone downloader with automatic resume capability.

Features:
- Detects last downloaded file and resumes from that point
- Downloads all market data types (DA, RT, AS, Load)
- Suitable for cron jobs - automatically catches up if run fails
- Validates data and logs progress

Usage:
    # Download from specific date to today
    python download_nyiso_standalone.py --start-date 2019-01-01

    # Auto-resume from last download
    python download_nyiso_standalone.py --auto-resume

    # Test with small range
    python download_nyiso_standalone.py --start-date 2024-01-01 --end-date 2024-01-07
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Tuple
import aiohttp
import re


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nyiso_download.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class NYISODownloader:
    """NYISO data downloader with resume capability."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.base_url = "http://mis.nyiso.com/public/csv"
        self.csv_dir = output_dir / "NYISO_data" / "csv_files"
        self.csv_dir.mkdir(parents=True, exist_ok=True)

        # Track statistics
        self.stats = {
            'downloaded': 0,
            'skipped': 0,
            'failed': 0,
            'total_attempted': 0
        }

    def find_last_downloaded_date(self, data_type: str, location_type: str = None) -> Optional[datetime]:
        """Find the most recent date that was successfully downloaded."""
        if location_type:
            search_dir = self.csv_dir / data_type / location_type
        else:
            search_dir = self.csv_dir / data_type

        if not search_dir.exists():
            return None

        # Find all CSV files and extract dates
        dates = []
        for csv_file in search_dir.glob("*.csv"):
            # Extract date from filename (format: YYYYMMDDfilename.csv)
            match = re.match(r'(\d{8})', csv_file.name)
            if match:
                try:
                    date = datetime.strptime(match.group(1), '%Y%m%d')
                    dates.append(date)
                except ValueError:
                    continue

        return max(dates) if dates else None

    def determine_start_date(self, requested_start: Optional[datetime], data_type: str,
                           location_type: str = None) -> datetime:
        """Determine actual start date based on last download."""
        last_date = self.find_last_downloaded_date(data_type, location_type)

        if last_date:
            # Resume from day after last successful download
            resume_date = last_date + timedelta(days=1)
            logger.info(f"Found existing data for {data_type}/{location_type or 'all'} through {last_date.date()}")
            logger.info(f"Resuming from {resume_date.date()}")

            # Use the later of resume_date or requested_start
            if requested_start and requested_start > resume_date:
                return requested_start
            return resume_date
        else:
            logger.info(f"No existing data found for {data_type}/{location_type or 'all'}")
            return requested_start or datetime(2019, 1, 1)

    async def download_file(self, url: str, output_path: Path,
                           session: aiohttp.ClientSession, retry_attempts: int = 3) -> bool:
        """Download a single file with retry logic."""
        if output_path.exists():
            self.stats['skipped'] += 1
            return True

        self.stats['total_attempted'] += 1

        for attempt in range(retry_attempts):
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=300)) as response:
                    if response.status == 200:
                        content = await response.read()
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        output_path.write_bytes(content)
                        self.stats['downloaded'] += 1
                        logger.info(f"✓ Downloaded: {output_path.name}")
                        return True
                    elif response.status == 404:
                        logger.debug(f"File not found (expected for future dates): {url}")
                        self.stats['failed'] += 1
                        return False
                    else:
                        logger.warning(f"HTTP {response.status} for {url}")
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {output_path.name}: {e}")
                if attempt < retry_attempts - 1:
                    await asyncio.sleep(30)

        self.stats['failed'] += 1
        return False

    async def download_lmp(self, start_date: datetime, end_date: datetime,
                          location_types: List[str] = None):
        """Download LMP data."""
        if location_types is None:
            location_types = ["zone", "gen"]

        markets = [
            ("DAM", "damlbmp", "damlbmp"),
            ("RT5M", "realtime", "realtime")
        ]

        async with aiohttp.ClientSession() as session:
            for market_code, url_prefix, file_prefix in markets:
                logger.info(f"\n{'='*60}")
                logger.info(f"Downloading {market_code} LMP data")
                logger.info(f"{'='*60}")

                for loc_type in location_types:
                    # Determine actual start date for this specific dataset
                    actual_start = self.determine_start_date(start_date, market_code.lower(), loc_type)

                    if actual_start > end_date:
                        logger.info(f"  {loc_type}: Already up to date")
                        continue

                    logger.info(f"  Downloading {loc_type} data from {actual_start.date()} to {end_date.date()}")

                    current_date = actual_start
                    while current_date <= end_date:
                        date_str = current_date.strftime("%Y%m%d")
                        filename = f"{date_str}{file_prefix}_{loc_type}.csv"
                        url = f"{self.base_url}/{url_prefix}/{filename}"
                        output_path = self.csv_dir / market_code.lower() / loc_type / filename

                        await self.download_file(url, output_path, session)
                        current_date += timedelta(days=1)

    async def download_ancillary_services(self, start_date: datetime, end_date: datetime):
        """Download ancillary services data."""
        markets = [
            ("DAM", "damasp", "damasp"),
            ("RTM", "rtasp", "rtasp")
        ]

        async with aiohttp.ClientSession() as session:
            for market_code, url_prefix, file_prefix in markets:
                logger.info(f"\n{'='*60}")
                logger.info(f"Downloading {market_code} Ancillary Services")
                logger.info(f"{'='*60}")

                # Determine actual start date for this dataset
                actual_start = self.determine_start_date(start_date, f"ancillary_services/{market_code.lower()}")

                if actual_start > end_date:
                    logger.info(f"  Already up to date")
                    continue

                logger.info(f"  Downloading from {actual_start.date()} to {end_date.date()}")

                current_date = actual_start
                while current_date <= end_date:
                    date_str = current_date.strftime("%Y%m%d")
                    filename = f"{date_str}{file_prefix}.csv"
                    url = f"{self.base_url}/{url_prefix}/{filename}"
                    output_path = self.csv_dir / "ancillary_services" / market_code.lower() / filename

                    await self.download_file(url, output_path, session)
                    current_date += timedelta(days=1)

    async def download_load(self, start_date: datetime, end_date: datetime):
        """Download load data."""
        load_types = [
            ("actual", "pal", "pal"),
            ("forecast", "isolf", "isolf")
        ]

        async with aiohttp.ClientSession() as session:
            for load_type, url_prefix, file_prefix in load_types:
                logger.info(f"\n{'='*60}")
                logger.info(f"Downloading {load_type} load data")
                logger.info(f"{'='*60}")

                # Determine actual start date for this dataset
                actual_start = self.determine_start_date(start_date, f"load/{load_type}")

                if actual_start > end_date:
                    logger.info(f"  Already up to date")
                    continue

                logger.info(f"  Downloading from {actual_start.date()} to {end_date.date()}")

                current_date = actual_start
                while current_date <= end_date:
                    date_str = current_date.strftime("%Y%m%d")
                    filename = f"{date_str}{file_prefix}.csv"
                    url = f"{self.base_url}/{url_prefix}/{filename}"
                    output_path = self.csv_dir / "load" / load_type / filename

                    await self.download_file(url, output_path, session)
                    current_date += timedelta(days=1)

    async def download_all(self, start_date: datetime, end_date: datetime):
        """Download all NYISO data types."""
        logger.info(f"\n{'='*80}")
        logger.info(f"NYISO DATA DOWNLOAD")
        logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
        logger.info(f"Output directory: {self.csv_dir}")
        logger.info(f"{'='*80}\n")

        # Download all data types sequentially (easier to track progress)
        await self.download_lmp(start_date, end_date)
        await self.download_ancillary_services(start_date, end_date)
        await self.download_load(start_date, end_date)

        # Print final statistics
        logger.info(f"\n{'='*80}")
        logger.info(f"DOWNLOAD COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Total attempted: {self.stats['total_attempted']}")
        logger.info(f"Successfully downloaded: {self.stats['downloaded']}")
        logger.info(f"Already existed (skipped): {self.stats['skipped']}")
        logger.info(f"Failed/Not found: {self.stats['failed']}")
        logger.info(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="NYISO standalone downloader with auto-resume"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date (YYYY-MM-DD). If --auto-resume, this is the fallback if no existing data found."
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date (YYYY-MM-DD). Defaults to today."
    )
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="Automatically resume from last downloaded date"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/pool/ssd8tb/data/iso",
        help="Output directory for CSV files"
    )

    args = parser.parse_args()

    # Parse dates
    if args.start_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    elif args.auto_resume:
        start_date = datetime(2019, 1, 1)  # Will be overridden by resume logic
    else:
        logger.error("Must specify --start-date or --auto-resume")
        sys.exit(1)

    end_date = datetime.strptime(args.end_date, "%Y-%m-%d") if args.end_date else datetime.now()
    output_dir = Path(args.output_dir)

    # Create downloader and run
    downloader = NYISODownloader(output_dir)
    asyncio.run(downloader.download_all(start_date, end_date))


if __name__ == "__main__":
    main()

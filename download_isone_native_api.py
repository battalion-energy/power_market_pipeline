#!/usr/bin/env python3
"""
ISO-NE data downloader using native ISO-NE Web Services API.

Bypasses gridstatus to get full native data directly from ISO-NE.

Features:
- Direct ISO-NE API calls (no gridstatus filtering)
- Automatically resumes from last downloaded date
- Handles RT LMP 5-min (requires 24 hourly API calls per day)
- Handles Load data
- Auto-resume capability for cron jobs

API Documentation: https://www.iso-ne.com/participate/support/web-services-data

Usage:
    python download_isone_native_api.py --start-date 2019-01-01
    python download_isone_native_api.py --auto-resume
"""

import argparse
import logging
import sys
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
import requests
from requests.auth import HTTPBasicAuth
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('isone_native_api_download.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class ISONENativeAPIDownloader:
    """ISO-NE downloader using native Web Services API."""

    def __init__(self, output_dir: Path, username: str, password: str):
        self.output_dir = output_dir
        self.csv_dir = output_dir / "ISONE_data" / "csv_files"
        self.csv_dir.mkdir(parents=True, exist_ok=True)

        self.base_url = "https://webservices.iso-ne.com/api/v1.1"
        self.auth = HTTPBasicAuth(username, password)

        self.stats = {
            'downloaded': 0,
            'skipped': 0,
            'failed': 0
        }

    def make_api_call(self, url: str, retries: int = 3) -> dict:
        """Make API call with retries."""
        for attempt in range(retries):
            try:
                # Add Accept header to request JSON format
                headers = {'Accept': 'application/json'}
                response = requests.get(url, auth=self.auth, headers=headers, timeout=30)
                response.raise_for_status()
                return response.json()
            except Exception as e:
                if attempt < retries - 1:
                    logger.warning(f"  Retry {attempt + 1}/{retries} for {url}: {e}")
                    continue
                else:
                    raise

    def find_last_date(self, data_type: str) -> datetime:
        """Find the most recent date downloaded for a data type."""
        type_dir = self.csv_dir / data_type
        if not type_dir.exists():
            return None

        dates = []
        for csv_file in type_dir.glob("*.csv"):
            try:
                date_str = csv_file.stem.split('_')[0]
                date = datetime.strptime(date_str, '%Y-%m-%d')
                dates.append(date)
            except:
                continue

        return max(dates) if dates else None

    def save_dataframe(self, df: pd.DataFrame, data_type: str, date: datetime):
        """Save DataFrame to CSV."""
        if df is None or len(df) == 0:
            return False

        date_str = date.strftime('%Y-%m-%d')
        filename = f"{date_str}_{data_type}.csv"
        output_path = self.csv_dir / data_type / filename

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        return True

    def download_rt_lmp_5min_day(self, date: datetime):
        """Download RT LMP 5-min data for one day.

        ISO-NE requires calling hour-by-hour (24 API calls per day).
        Using 'final' data which is more complete than 'prelim'.
        """
        data_type = "lmp_real_time_5_min"
        output_path = self.csv_dir / data_type / f"{date.strftime('%Y-%m-%d')}_{data_type}.csv"

        if output_path.exists():
            self.stats['skipped'] += 1
            logger.debug(f"Skipping {data_type} {date.date()} (already exists)")
            return True

        try:
            logger.info(f"Downloading {data_type} for {date.date()}...")

            all_data = []
            date_str = date.strftime('%Y%m%d')

            # Need to call each hour (0-23) separately
            for hour in range(24):
                url = f"{self.base_url}/fiveminutelmp/final/day/{date_str}/starthour/{hour:02d}"

                try:
                    # Add delay to respect rate limits (1 second between calls)
                    if hour > 0:
                        time.sleep(1)
                    response = self.make_api_call(url)

                    # Extract data from response
                    if "FiveMinLmps" in response and "FiveMinLmp" in response["FiveMinLmps"]:
                        hour_data = response["FiveMinLmps"]["FiveMinLmp"]
                        if isinstance(hour_data, list):
                            all_data.extend(hour_data)
                        else:
                            all_data.append(hour_data)
                except Exception as e:
                    logger.warning(f"  Failed hour {hour}: {e}")
                    continue

            if all_data:
                df = pd.DataFrame(all_data)

                if self.save_dataframe(df, data_type, date):
                    self.stats['downloaded'] += 1
                    logger.info(f"  ✓ Saved {len(df)} rows to {output_path.name}")
                    return True
            else:
                logger.warning(f"  No data returned for {date.date()}")
                self.stats['failed'] += 1

        except Exception as e:
            logger.error(f"  ✗ Failed to download {data_type} for {date.date()}: {e}")
            self.stats['failed'] += 1

        return False

    def download_load_day(self, date: datetime):
        """Download load data for one day."""
        data_type = "load"
        output_path = self.csv_dir / data_type / f"{date.strftime('%Y-%m-%d')}_{data_type}.csv"

        if output_path.exists():
            self.stats['skipped'] += 1
            logger.debug(f"Skipping load {date.date()} (already exists)")
            return True

        try:
            logger.info(f"Downloading load for {date.date()}...")

            date_str = date.strftime('%Y%m%d')
            url = f"{self.base_url}/hourlysysload/day/{date_str}"

            response = self.make_api_call(url)

            # Extract data from response
            if "HourlySysLoads" in response and "HourlySysLoad" in response["HourlySysLoads"]:
                data = response["HourlySysLoads"]["HourlySysLoad"]

                if isinstance(data, list):
                    df = pd.DataFrame(data)
                else:
                    df = pd.DataFrame([data])

                if self.save_dataframe(df, data_type, date):
                    self.stats['downloaded'] += 1
                    logger.info(f"  ✓ Saved {len(df)} rows to {output_path.name}")
                    return True
            else:
                logger.warning(f"  No load data returned for {date.date()}")
                self.stats['failed'] += 1

        except Exception as e:
            logger.error(f"  ✗ Failed to download load for {date.date()}: {e}")
            self.stats['failed'] += 1

        return False

    def download_date_range(self, start_date: datetime, end_date: datetime):
        """Download all data types for a date range."""
        logger.info(f"\n{'='*80}")
        logger.info(f"ISO-NE DATA DOWNLOAD (Native API)")
        logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
        logger.info(f"Output directory: {self.csv_dir}")
        logger.info(f"{'='*80}\n")

        current_date = start_date
        while current_date <= end_date:
            logger.info(f"\n--- Processing {current_date.date()} ---")

            # Download RT LMP 5-min (requires 24 API calls)
            self.download_rt_lmp_5min_day(current_date)

            # Download Load
            self.download_load_day(current_date)

            current_date += timedelta(days=1)

        # Print statistics
        logger.info(f"\n{'='*80}")
        logger.info(f"DOWNLOAD COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Downloaded: {self.stats['downloaded']} files")
        logger.info(f"Skipped (existing): {self.stats['skipped']} files")
        logger.info(f"Failed: {self.stats['failed']} files")
        logger.info(f"{'='*80}\n")

    def auto_resume(self, fallback_start: datetime, end_date: datetime):
        """Auto-resume from the last downloaded date."""
        data_types = ["lmp_real_time_5_min", "load"]

        last_dates = []
        for data_type in data_types:
            last_date = self.find_last_date(data_type)
            if last_date:
                last_dates.append((data_type, last_date))
                logger.info(f"Found {data_type} data through {last_date.date()}")

        if last_dates:
            earliest = min(last_dates, key=lambda x: x[1])
            resume_date = earliest[1] + timedelta(days=1)
            logger.info(f"\n✓ Resuming from {resume_date.date()} (earliest incomplete: {earliest[0]})")
        else:
            resume_date = fallback_start
            logger.info(f"\n✓ No existing data found, starting from {resume_date.date()}")

        if resume_date > end_date:
            logger.info(f"✓ Already up to date! Last data: {resume_date - timedelta(days=1)}")
            return

        self.download_date_range(resume_date, end_date)


def main():
    parser = argparse.ArgumentParser(
        description="ISO-NE downloader using native Web Services API"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date (YYYY-MM-DD). Fallback if no existing data found."
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

    # Get credentials from environment
    username = os.getenv("ISONE_USERNAME")
    password = os.getenv("ISONE_PASSWORD")

    if not username or not password:
        logger.error("ISONE_USERNAME and ISONE_PASSWORD must be set in environment or .env file")
        sys.exit(1)

    # Parse dates
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d") if args.start_date else datetime(2019, 1, 1)
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d") if args.end_date else datetime.now()
    output_dir = Path(args.output_dir)

    # Create downloader
    downloader = ISONENativeAPIDownloader(output_dir, username, password)

    if args.auto_resume:
        downloader.auto_resume(start_date, end_date)
    else:
        downloader.download_date_range(start_date, end_date)


if __name__ == "__main__":
    main()

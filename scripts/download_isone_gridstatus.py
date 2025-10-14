#!/usr/bin/env python3
"""
ISO-NE (ISONE) data downloader using gridstatus library with auto-resume.

Features:
- Uses gridstatus library (handles all API complexity)
- Automatically resumes from last downloaded date
- Saves data to CSV files organized by date
- Suitable for cron jobs
- Logs all operations

Usage:
    # Download from 2019 to today
    python download_isone_gridstatus.py --start-date 2019-01-01

    # Auto-resume (starts from last downloaded date or 2019-01-01)
    python download_isone_gridstatus.py --auto-resume

    # Test with small range
    python download_isone_gridstatus.py --start-date 2024-01-01 --end-date 2024-01-03
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
import gridstatus
import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('isone_gridstatus_download.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class ISONEGridstatusDownloader:
    """ISO-NE downloader using gridstatus with resume capability."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.csv_dir = output_dir / "ISONE_data" / "csv_files"
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        self.isone = gridstatus.ISONE()

        self.stats = {
            'downloaded': 0,
            'skipped': 0,
            'failed': 0
        }

    def find_last_date(self, data_type: str) -> datetime:
        """Find the most recent date downloaded for a data type."""
        type_dir = self.csv_dir / data_type
        if not type_dir.exists():
            return None

        dates = []
        for csv_file in type_dir.glob("*.csv"):
            # Extract date from filename (format: YYYY-MM-DD_*.csv)
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

    def download_lmp_day(self, date: datetime, market: str):
        """Download LMP data for one day."""
        data_type = f"lmp_{market.lower()}"
        output_path = self.csv_dir / data_type / f"{date.strftime('%Y-%m-%d')}_{data_type}.csv"

        if output_path.exists():
            self.stats['skipped'] += 1
            logger.debug(f"Skipping {data_type} {date.date()} (already exists)")
            return True

        try:
            logger.info(f"Downloading {data_type} for {date.date()}...")
            df = self.isone.get_lmp(date=date, market=market, locations="ALL")

            if self.save_dataframe(df, data_type, date):
                self.stats['downloaded'] += 1
                logger.info(f"  ✓ Saved {len(df)} rows to {output_path.name}")
                return True
        except Exception as e:
            logger.error(f"  ✗ Failed to download {data_type} for {date.date()}: {e}")
            self.stats['failed'] += 1

        return False

    def download_as_day(self, date: datetime, market: str):
        """Download ancillary services for one day."""
        data_type = f"as_{market.lower()}"
        output_path = self.csv_dir / data_type / f"{date.strftime('%Y-%m-%d')}_{data_type}.csv"

        if output_path.exists():
            self.stats['skipped'] += 1
            logger.debug(f"Skipping {data_type} {date.date()} (already exists)")
            return True

        try:
            logger.info(f"Downloading {data_type} for {date.date()}...")

            # ISO-NE AS prices
            if market == "DAY_AHEAD_HOURLY":
                df = self.isone.get_as_prices(date=date, market="DAY_AHEAD_HOURLY")
            else:  # REAL_TIME_5_MIN
                df = self.isone.get_as_prices(date=date, market="REAL_TIME_5_MIN")

            if self.save_dataframe(df, data_type, date):
                self.stats['downloaded'] += 1
                logger.info(f"  ✓ Saved {len(df)} rows to {output_path.name}")
                return True
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
            end_date = date + timedelta(days=1)
            df = self.isone.get_load(start=date, end=end_date)

            if self.save_dataframe(df, data_type, date):
                self.stats['downloaded'] += 1
                logger.info(f"  ✓ Saved {len(df)} rows to {output_path.name}")
                return True
        except Exception as e:
            logger.error(f"  ✗ Failed to download load for {date.date()}: {e}")
            self.stats['failed'] += 1

        return False

    def download_fuel_mix_day(self, date: datetime):
        """Download fuel mix data for one day."""
        data_type = "fuel_mix"
        output_path = self.csv_dir / data_type / f"{date.strftime('%Y-%m-%d')}_{data_type}.csv"

        if output_path.exists():
            self.stats['skipped'] += 1
            logger.debug(f"Skipping fuel mix {date.date()} (already exists)")
            return True

        try:
            logger.info(f"Downloading fuel mix for {date.date()}...")
            end_date = date + timedelta(days=1)
            df = self.isone.get_fuel_mix(start=date, end=end_date)

            if self.save_dataframe(df, data_type, date):
                self.stats['downloaded'] += 1
                logger.info(f"  ✓ Saved {len(df)} rows to {output_path.name}")
                return True
        except Exception as e:
            logger.error(f"  ✗ Failed to download fuel mix for {date.date()}: {e}")
            self.stats['failed'] += 1

        return False

    def download_date_range(self, start_date: datetime, end_date: datetime):
        """Download all data types for a date range."""
        logger.info(f"\n{'='*80}")
        logger.info(f"ISO-NE DATA DOWNLOAD (gridstatus)")
        logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
        logger.info(f"Output directory: {self.csv_dir}")
        logger.info(f"{'='*80}\n")

        current_date = start_date
        while current_date <= end_date:
            logger.info(f"\n--- Processing {current_date.date()} ---")

            # Download data types (skip AS - not available for ISO-NE)
            self.download_lmp_day(current_date, "DAY_AHEAD_HOURLY")
            self.download_lmp_day(current_date, "REAL_TIME_5_MIN")
            self.download_load_day(current_date)
            self.download_fuel_mix_day(current_date)

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
        # Find the earliest last date across all data types
        data_types = ["lmp_day_ahead_hourly", "lmp_real_time_5_min",
                     "as_day_ahead_hourly", "as_real_time_5_min", "load", "fuel_mix"]

        last_dates = []
        for data_type in data_types:
            last_date = self.find_last_date(data_type)
            if last_date:
                last_dates.append((data_type, last_date))
                logger.info(f"Found {data_type} data through {last_date.date()}")

        if last_dates:
            # Use the earliest last date to ensure all data types are in sync
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
        description="ISO-NE downloader using gridstatus with auto-resume"
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

    # Parse dates
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d") if args.start_date else datetime(2019, 1, 1)
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d") if args.end_date else datetime.now()
    output_dir = Path(args.output_dir)

    # Create downloader
    downloader = ISONEGridstatusDownloader(output_dir)

    if args.auto_resume:
        downloader.auto_resume(start_date, end_date)
    else:
        downloader.download_date_range(start_date, end_date)


if __name__ == "__main__":
    main()

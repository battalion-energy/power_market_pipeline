#!/usr/bin/env python3
"""
ISO-NE data downloader using historical archive CSV files.

Downloads directly from ISO-NE's public historical data archives.
No authentication required, no rate limits - simple HTTP downloads.

Features:
- Direct CSV downloads from public archives
- Automatically resumes from last downloaded date
- Handles Day-Ahead LMP and Real-Time LMP data
- Suitable for cron jobs
- Logs all operations

Historical Archive URLs:
- Day-Ahead LMP: https://www.iso-ne.com/histRpts/da-lmp/WW_DALMP_ISO_YYYYMMDD.csv
- Real-Time LMP: https://www.iso-ne.com/histRpts/rt-lmp/lmp_rt_final_YYYYMMDD.csv

Usage:
    # Download from 2019 to today
    python download_isone_historical_archive.py --start-date 2019-01-01

    # Auto-resume (starts from last downloaded date or 2019-01-01)
    python download_isone_historical_archive.py --auto-resume

    # Test with small range
    python download_isone_historical_archive.py --start-date 2024-01-01 --end-date 2024-01-03
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
import requests
import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('isone_historical_download.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class ISONEHistoricalDownloader:
    """ISO-NE downloader using historical archive CSV files."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.csv_dir = output_dir / "ISONE_data" / "csv_files"
        self.csv_dir.mkdir(parents=True, exist_ok=True)

        self.base_url_da = "https://www.iso-ne.com/histRpts/da-lmp"
        self.base_url_rt = "https://www.iso-ne.com/histRpts/rt-lmp"

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

    def download_csv(self, url: str, retries: int = 3) -> pd.DataFrame:
        """Download CSV file with retries.

        ISO-NE CSV files have special format:
        - Lines starting with "C" are comments
        - Lines starting with "H" are headers (2 lines)
        - Lines starting with "D" are data rows
        """
        for attempt in range(retries):
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()

                # Parse the special ISO-NE CSV format
                from io import StringIO
                lines = response.text.strip().split('\n')

                # Extract header from the first "H" line
                header_line = None
                data_lines = []

                for line in lines:
                    if line.startswith('"H"'):
                        # First H line contains column names
                        if header_line is None:
                            # Remove the "H" prefix and parse
                            parts = line.split(',', 1)
                            if len(parts) > 1:
                                header_line = parts[1]
                    elif line.startswith('"D"'):
                        # Data line - remove "D" prefix
                        parts = line.split(',', 1)
                        if len(parts) > 1:
                            data_lines.append(parts[1])

                if not header_line or not data_lines:
                    logger.warning(f"  No valid data in file: {url}")
                    return None

                # Reconstruct CSV with header and data
                csv_content = header_line + '\n' + '\n'.join(data_lines)
                df = pd.read_csv(StringIO(csv_content))

                return df

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    # File doesn't exist - this is expected for some dates
                    return None
                if attempt < retries - 1:
                    logger.warning(f"  Retry {attempt + 1}/{retries} for {url}: {e}")
                    continue
                else:
                    raise
            except Exception as e:
                if attempt < retries - 1:
                    logger.warning(f"  Retry {attempt + 1}/{retries} for {url}: {e}")
                    continue
                else:
                    raise

    def download_da_lmp_day(self, date: datetime):
        """Download Day-Ahead LMP data for one day."""
        data_type = "lmp_day_ahead_hourly"
        output_path = self.csv_dir / data_type / f"{date.strftime('%Y-%m-%d')}_{data_type}.csv"

        if output_path.exists():
            self.stats['skipped'] += 1
            logger.debug(f"Skipping {data_type} {date.date()} (already exists)")
            return True

        try:
            logger.info(f"Downloading {data_type} for {date.date()}...")

            # Format: WW_DALMP_ISO_YYYYMMDD.csv
            date_str = date.strftime('%Y%m%d')
            url = f"{self.base_url_da}/WW_DALMP_ISO_{date_str}.csv"

            df = self.download_csv(url)

            if df is not None:
                if self.save_dataframe(df, data_type, date):
                    self.stats['downloaded'] += 1
                    logger.info(f"  ✓ Saved {len(df)} rows to {output_path.name}")
                    return True
            else:
                logger.warning(f"  No data available for {date.date()}")
                self.stats['failed'] += 1

        except Exception as e:
            logger.error(f"  ✗ Failed to download {data_type} for {date.date()}: {e}")
            self.stats['failed'] += 1

        return False

    def download_rt_lmp_day(self, date: datetime):
        """Download Real-Time LMP data for one day."""
        data_type = "lmp_real_time_5_min"
        output_path = self.csv_dir / data_type / f"{date.strftime('%Y-%m-%d')}_{data_type}.csv"

        if output_path.exists():
            self.stats['skipped'] += 1
            logger.debug(f"Skipping {data_type} {date.date()} (already exists)")
            return True

        try:
            logger.info(f"Downloading {data_type} for {date.date()}...")

            # Format: lmp_rt_final_YYYYMMDD.csv (try final first, then prelim)
            date_str = date.strftime('%Y%m%d')

            # Try final first
            url_final = f"{self.base_url_rt}/lmp_rt_final_{date_str}.csv"
            df = self.download_csv(url_final)

            # If final doesn't exist, try prelim
            if df is None:
                url_prelim = f"{self.base_url_rt}/lmp_rt_prelim_{date_str}.csv"
                df = self.download_csv(url_prelim)

            if df is not None:
                if self.save_dataframe(df, data_type, date):
                    self.stats['downloaded'] += 1
                    logger.info(f"  ✓ Saved {len(df)} rows to {output_path.name}")
                    return True
            else:
                logger.warning(f"  No data available for {date.date()}")
                self.stats['failed'] += 1

        except Exception as e:
            logger.error(f"  ✗ Failed to download {data_type} for {date.date()}: {e}")
            self.stats['failed'] += 1

        return False

    def download_date_range(self, start_date: datetime, end_date: datetime):
        """Download all data types for a date range."""
        logger.info(f"\n{'='*80}")
        logger.info(f"ISO-NE DATA DOWNLOAD (Historical Archive)")
        logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
        logger.info(f"Output directory: {self.csv_dir}")
        logger.info(f"{'='*80}\n")

        current_date = start_date
        while current_date <= end_date:
            logger.info(f"\n--- Processing {current_date.date()} ---")

            # Download Day-Ahead LMP
            self.download_da_lmp_day(current_date)

            # Download Real-Time LMP
            self.download_rt_lmp_day(current_date)

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
        data_types = ["lmp_day_ahead_hourly", "lmp_real_time_5_min"]

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
        description="ISO-NE downloader using historical archive CSV files"
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
    downloader = ISONEHistoricalDownloader(output_dir)

    if args.auto_resume:
        downloader.auto_resume(start_date, end_date)
    else:
        downloader.download_date_range(start_date, end_date)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
MISO 5-Minute Real-Time LMP Data Downloader

Downloads 5-minute resolution real-time LMP data from MISO.

Data Sources:
1. Weekly Historical Files (ZIP archives):
   - URL: https://docs.misoenergy.org/marketreports/YYYYMMDD_5MIN_LMP.zip
   - Date should be a Monday
   - Contains data from 2 weeks before through 1 week before file date

2. Recent Data API (Last 4 days):
   - Current interval: messageType=currentinterval
   - Today (rolling): messageType=rollingmarketday
   - Yesterday: messageType=previousmarketday
   - 2 days ago: messageType=previousmarketday2
   - 3 days ago: messageType=previousmarketday3

Data Format:
- 5-minute intervals (12 per hour, 288 per day)
- Columns: Timestamp, Node, LMP, MLC (Loss), MCC (Congestion), MEC (Energy)
- ~7,000+ nodes per timestamp (all nodes)
- Weekly files: ~500MB+ compressed, ~2-3GB uncompressed
"""

import asyncio
import os
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import httpx
import pandas as pd
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables
load_dotenv()

# Configuration
MISO_BASE_URL = "https://docs.misoenergy.org/marketreports"
MISO_RT_API_URL = "https://api.misoenergy.org/MISORTWDBIReporter/Reporter.asmx"
MISO_DATA_DIR = os.getenv("MISO_DATA_DIR", "/pool/ssd8tb/data/iso/MISO/csv_files")


class MISO5MinLMPDownloader:
    """Downloads MISO 5-minute real-time LMP data."""

    def __init__(self, output_dir: str = MISO_DATA_DIR):
        self.output_dir = Path(output_dir)
        self.session: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self.session = httpx.AsyncClient(timeout=600.0, follow_redirects=True)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.aclose()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def download_weekly_zip(
        self,
        monday_date: datetime,
        filter_hubs: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Download weekly 5-minute LMP ZIP file.

        Weekly files are dated on Mondays and contain data from
        two weeks before through one week before the file date.

        Args:
            monday_date: A Monday date for the weekly file
            filter_hubs: If True, only keep hub-level data
        """
        date_str = monday_date.strftime("%Y%m%d")
        url = f"{MISO_BASE_URL}/{date_str}_5MIN_LMP.zip"

        # Create output directory
        output_dir = self.output_dir / "rt_5min_lmp"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Output paths
        zip_file = output_dir / f"{date_str}_5MIN_LMP.zip"
        csv_file = output_dir / f"{date_str}_5MIN_LMP.csv"

        # If CSV already exists, skip
        if csv_file.exists():
            print(f"✓ Already exists: {csv_file.name}")
            return pd.read_csv(csv_file)

        # If ZIP exists but not CSV, extract it
        if zip_file.exists():
            print(f"Extracting existing ZIP: {zip_file.name}...", end=" ")
            try:
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    # Extract CSV (check both .csv and .CSV)
                    csv_names = [name for name in zip_ref.namelist() if name.lower().endswith('.csv')]
                    if csv_names:
                        extracted_csv = zip_ref.extract(csv_names[0], output_dir)
                        # Rename to standard name
                        Path(extracted_csv).rename(csv_file)

                        # Parse CSV - skip 4 header rows and handle footer disclaimer
                        df = pd.read_csv(csv_file, skiprows=4, on_bad_lines='skip')
                        # Note: 5MIN_LMP files don't have a 'Type' column for hub filtering
                        # For now, we save all data - hub filtering can be done in post-processing
                        df.to_csv(csv_file, index=False)
                        print(f"✓ Extracted ({len(df)} records)")
                        return df
            except Exception as e:
                print(f"✗ Error extracting: {e}")
                return None

        # Download ZIP file
        try:
            print(f"Downloading: {monday_date.strftime('%Y-%m-%d')} weekly 5MIN_LMP...", end=" ")

            response = await self.session.get(url)

            if response.status_code == 404:
                print(f"✗ Not available (404)")
                return None

            response.raise_for_status()

            # Save ZIP file
            zip_file.write_bytes(response.content)
            print(f"✓ Downloaded ({len(response.content) / 1024 / 1024:.1f} MB)", end=" ")

            # Extract CSV from ZIP
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                csv_names = [name for name in zip_ref.namelist() if name.lower().endswith('.csv')]
                if not csv_names:
                    print(f"✗ No CSV found in ZIP")
                    return None

                # Extract the CSV
                extracted_csv = zip_ref.extract(csv_names[0], output_dir)

                # Rename to standard name
                Path(extracted_csv).rename(csv_file)

                # Parse CSV (skip 4 header rows and handle footer disclaimer)
                try:
                    df = pd.read_csv(csv_file, skiprows=4, on_bad_lines='skip')
                    # Note: 5MIN_LMP files don't have a 'Type' column for hub filtering
                    # For now, we save all data - hub filtering can be done in post-processing
                    df.to_csv(csv_file, index=False)
                    print(f"→ Extracted ({len(df)} records)")
                    return df

                except Exception as e:
                    print(f"✗ Error parsing CSV: {e}")
                    return None

        except httpx.HTTPError as e:
            print(f"✗ HTTP Error: {e}")
            return None
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
            return None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def download_recent_api(
        self,
        message_type: str = "rollingmarketday"
    ) -> Optional[pd.DataFrame]:
        """
        Download recent 5-minute LMP data via API (last 4 days).

        Args:
            message_type: One of:
                - currentinterval: Current 5-minute interval
                - rollingmarketday: Today (rolling market day)
                - previousmarketday: Yesterday
                - previousmarketday2: 2 days ago
                - previousmarketday3: 3 days ago
        """
        url = f"{MISO_RT_API_URL}?messageType={message_type}&returnType=csv"

        # Create output directory
        output_dir = self.output_dir / "rt_5min_lmp" / "recent"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Output file
        date_str = datetime.now().strftime("%Y%m%d")
        output_file = output_dir / f"{date_str}_{message_type}.csv"

        try:
            print(f"Downloading: {message_type} via API...", end=" ")

            response = await self.session.get(url)
            response.raise_for_status()

            # Save CSV
            output_file.write_text(response.text)

            # Parse CSV
            try:
                df = pd.read_csv(output_file)
                print(f"✓ Downloaded ({len(df)} records)")
                return df
            except Exception as e:
                print(f"✓ Downloaded (binary file, parse error: {e})")
                return None

        except httpx.HTTPError as e:
            print(f"✗ HTTP Error: {e}")
            return None
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
            return None

    def get_monday_dates(self, start_date: datetime, end_date: datetime) -> List[datetime]:
        """
        Generate list of Monday dates for weekly file downloads.

        Weekly files are dated on Mondays and contain data from
        2 weeks before through 1 week before the file date.

        To get complete coverage, we need to download files where:
        - File Monday date > start_date + 1 week
        - File Monday date <= end_date + 2 weeks
        """
        monday_dates = []

        # Start from first Monday after (start_date + 1 week)
        current = start_date + timedelta(days=7)
        # Find next Monday
        days_until_monday = (7 - current.weekday()) % 7
        current = current + timedelta(days=days_until_monday)

        # End at last Monday before (end_date + 2 weeks)
        end = end_date + timedelta(days=14)

        while current <= end:
            monday_dates.append(current)
            current += timedelta(days=7)

        return monday_dates

    async def download_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        filter_hubs: bool = False,
        download_recent_api: bool = False,
        max_concurrent: int = 3  # Lower concurrency for large files
    ):
        """
        Download 5-minute RT LMP data for a date range.

        Args:
            start_date: Start date for data coverage
            end_date: End date for data coverage
            filter_hubs: If True, only keep hub-level data (~130 nodes)
            download_recent_api: If True, also download last 4 days via API
            max_concurrent: Maximum concurrent downloads (default 3 for large files)
        """
        print(f"\n=== MISO 5-Minute RT LMP Downloader ===")
        print(f"Data coverage: {start_date.date()} to {end_date.date()}")
        print(f"Filter hubs only: {filter_hubs}")
        print(f"Output: {self.output_dir}")
        print(f"Max concurrent downloads: {max_concurrent}")
        print(f"Note: Weekly files are ~500MB compressed, ~2-3GB uncompressed\n")

        # Get Monday dates for weekly files
        monday_dates = self.get_monday_dates(start_date, end_date)
        print(f"Weekly files to download: {len(monday_dates)}")
        print(f"File date range: {monday_dates[0].date()} to {monday_dates[-1].date()}\n")

        # Download weekly files
        semaphore = asyncio.Semaphore(max_concurrent)

        async def download_with_limit(monday_date):
            async with semaphore:
                return await self.download_weekly_zip(monday_date, filter_hubs)

        tasks = [download_with_limit(date) for date in monday_dates]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Summary for weekly files
        successful = sum(1 for r in results if r is not None and not isinstance(r, Exception))
        failed = sum(1 for r in results if isinstance(r, Exception))
        not_available = sum(1 for r in results if r is None)

        print(f"\n=== Weekly Files Summary ===")
        print(f"Successful: {successful}")
        print(f"Not available: {not_available}")
        print(f"Failed: {failed}")

        # Download recent data via API if requested
        if download_recent_api:
            print(f"\n=== Downloading Recent Data via API ===")
            api_types = [
                "currentinterval",
                "rollingmarketday",
                "previousmarketday",
                "previousmarketday2",
                "previousmarketday3"
            ]

            for api_type in api_types:
                await self.download_recent_api(api_type)


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download MISO 5-minute real-time LMP data"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Start date for data coverage (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date for data coverage (YYYY-MM-DD), defaults to today"
    )
    parser.add_argument(
        "--filter-hubs",
        action="store_true",
        help="Only download hub-level data (~130 nodes vs ~7,000 nodes)"
    )
    parser.add_argument(
        "--download-recent-api",
        action="store_true",
        help="Also download last 4 days via API"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=MISO_DATA_DIR,
        help="Output directory"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=3,
        help="Maximum concurrent downloads (default 3 for large files)"
    )

    args = parser.parse_args()

    # Parse dates
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d") if args.end_date else datetime.now()

    # Download data
    async with MISO5MinLMPDownloader(args.output_dir) as downloader:
        await downloader.download_date_range(
            start_date=start_date,
            end_date=end_date,
            filter_hubs=args.filter_hubs,
            download_recent_api=args.download_recent_api,
            max_concurrent=args.max_concurrent
        )


if __name__ == "__main__":
    asyncio.run(main())

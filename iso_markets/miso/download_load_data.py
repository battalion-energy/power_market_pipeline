#!/usr/bin/env python3
"""
MISO Load Data Downloader

Downloads load data (actual and forecast) from MISO's Market Reports portal.

Available Data:
- Daily Forecast and Actual Load by Local Resource Zone (df_al)
- Daily Regional Forecast and Actual Load (rf_al)

URL Pattern: https://docs.misoenergy.org/marketreports/YYYYMMDD_{report_type}.xls
"""

import asyncio
import os
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
MISO_DATA_DIR = os.getenv("MISO_DATA_DIR", "/pool/ssd8tb/data/iso/MISO/csv_files")

# Load Report Types
LOAD_REPORT_TYPES = {
    "local_resource_zone": "df_al",  # Daily Forecast and Actual Load by LRZ
    "regional": "rf_al",              # Daily Regional Forecast and Actual Load
}


class MISOLoadDownloader:
    """Downloads MISO load data from market reports portal."""

    def __init__(self, output_dir: str = MISO_DATA_DIR):
        self.output_dir = Path(output_dir)
        self.session: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self.session = httpx.AsyncClient(timeout=300.0, follow_redirects=True)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.aclose()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def download_day(
        self,
        date: datetime,
        report_type: str
    ) -> Optional[pd.DataFrame]:
        """Download load data for a single day."""

        date_str = date.strftime("%Y%m%d")
        report_file = LOAD_REPORT_TYPES[report_type]
        url = f"{MISO_BASE_URL}/{date_str}_{report_file}.xls"

        # Create output directory structure
        report_dir = self.output_dir / "load" / report_type
        report_dir.mkdir(parents=True, exist_ok=True)

        # Output file path
        output_file = report_dir / f"{date_str}_{report_file}.xls"

        # Skip if already downloaded
        if output_file.exists():
            print(f"✓ Already exists: {output_file.name}")
            return pd.read_excel(output_file)

        try:
            print(f"Downloading: {date.strftime('%Y-%m-%d')} {report_type}...", end=" ")

            response = await self.session.get(url)

            if response.status_code == 404:
                print(f"✗ Not available (404)")
                return None

            response.raise_for_status()
            content = response.content

            # Save raw file
            output_file.write_bytes(content)

            # Try to parse
            try:
                df = pd.read_excel(output_file)
                print(f"✓ Downloaded ({len(df)} records)")
                return df
            except Exception as e:
                print(f"✓ Downloaded (binary file, parse error: {e})")
                return None

        except httpx.HTTPError as e:
            print(f"✗ Error: {e}")
            return None
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
            return None

    async def download_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        report_types: List[str] = None,
        max_concurrent: int = 5
    ):
        """Download load data for a date range."""

        if report_types is None:
            # Default: Both LRZ and Regional
            report_types = list(LOAD_REPORT_TYPES.keys())

        # Generate all dates in range
        dates = []
        current = start_date
        while current <= end_date:
            dates.append(current)
            current += timedelta(days=1)

        print(f"\n=== MISO Load Data Downloader ===")
        print(f"Date range: {start_date.date()} to {end_date.date()}")
        print(f"Total days: {len(dates)}")
        print(f"Report types: {', '.join(report_types)}")
        print(f"Output: {self.output_dir}")
        print(f"Max concurrent downloads: {max_concurrent}\n")

        # Download data
        semaphore = asyncio.Semaphore(max_concurrent)

        async def download_with_limit(date, report_type):
            async with semaphore:
                return await self.download_day(date, report_type)

        tasks = []
        for report_type in report_types:
            for date in dates:
                tasks.append(download_with_limit(date, report_type))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Summary
        successful = sum(1 for r in results if r is not None and not isinstance(r, Exception))
        failed = sum(1 for r in results if isinstance(r, Exception))
        not_available = sum(1 for r in results if r is None)

        print(f"\n=== Download Summary ===")
        print(f"Successful: {successful}")
        print(f"Not available: {not_available}")
        print(f"Failed: {failed}")


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Download MISO load data")
    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD), defaults to today"
    )
    parser.add_argument(
        "--report-types",
        type=str,
        nargs="+",
        choices=list(LOAD_REPORT_TYPES.keys()),
        default=list(LOAD_REPORT_TYPES.keys()),
        help="Report types to download"
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
        default=5,
        help="Maximum concurrent downloads"
    )

    args = parser.parse_args()

    # Parse dates
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d") if args.end_date else datetime.now()

    # Download data
    async with MISOLoadDownloader(args.output_dir) as downloader:
        await downloader.download_date_range(
            start_date=start_date,
            end_date=end_date,
            report_types=args.report_types,
            max_concurrent=args.max_concurrent
        )


if __name__ == "__main__":
    asyncio.run(main())

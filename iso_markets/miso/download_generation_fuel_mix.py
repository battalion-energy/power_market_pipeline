#!/usr/bin/env python3
"""
MISO Generation and Fuel Mix Data Downloader

Downloads generation and fuel mix data from MISO's Market Reports portal.

Available Data:
- Real-Time 5-Minute Generation Fuel Mix (rt_gfm)
- Historical Generation Fuel Mix (Annual aggregation)
- ACE Data (Area Control Error)

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

# Generation Report Types
GEN_REPORT_TYPES = {
    "fuel_mix_5min": "sr_gfm",     # Short-Range Generation Fuel Mix (5-minute)
    "ace_data": "ace",              # ACE (Area Control Error) Data
}


class MISOGenerationDownloader:
    """Downloads MISO generation and fuel mix data from market reports portal."""

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
        """Download generation data for a single day."""

        date_str = date.strftime("%Y%m%d")
        report_file = GEN_REPORT_TYPES[report_type]

        # Special handling for ACE data format
        if report_type == "ace_data":
            month_day = date.strftime("%m%d")
            url = f"{MISO_BASE_URL}/{date_str}_ace_{month_day}.csv"
            file_ext = "csv"
        else:
            url = f"{MISO_BASE_URL}/{date_str}_{report_file}.xls"
            file_ext = "xls"

        # Create output directory structure
        report_dir = self.output_dir / "generation" / report_type
        report_dir.mkdir(parents=True, exist_ok=True)

        # Output file path
        output_file = report_dir / f"{date_str}_{report_file}.{file_ext}"

        # Skip if already downloaded
        if output_file.exists():
            print(f"✓ Already exists: {output_file.name}")
            if file_ext == "csv":
                return pd.read_csv(output_file)
            else:
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
            output_file.write_bytes(content) if file_ext != "csv" else output_file.write_text(response.text)

            # Try to parse
            try:
                if file_ext == "csv":
                    df = pd.read_csv(output_file)
                else:
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

    async def download_annual_fuel_mix(self, year: int) -> Optional[pd.DataFrame]:
        """Download annual historical generation fuel mix."""

        url = f"{MISO_BASE_URL}/historical_gen_fuel_mix_{year}.xls"

        # Create output directory
        report_dir = self.output_dir / "generation" / "fuel_mix_annual"
        report_dir.mkdir(parents=True, exist_ok=True)

        # Output file path
        output_file = report_dir / f"historical_gen_fuel_mix_{year}.xls"

        # Skip if already downloaded
        if output_file.exists():
            print(f"✓ Already exists: {output_file.name}")
            return pd.read_excel(output_file)

        try:
            print(f"Downloading: {year} annual fuel mix...", end=" ")

            response = await self.session.get(url)

            if response.status_code == 404:
                print(f"✗ Not available (404)")
                return None

            response.raise_for_status()

            # Save raw file
            output_file.write_bytes(response.content)

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
        download_annual: bool = False,
        max_concurrent: int = 5
    ):
        """Download generation data for a date range."""

        if report_types is None:
            # Default: Fuel mix 5-min data
            report_types = ["fuel_mix_5min"]

        # Generate all dates in range
        dates = []
        current = start_date
        while current <= end_date:
            dates.append(current)
            current += timedelta(days=1)

        print(f"\n=== MISO Generation Data Downloader ===")
        print(f"Date range: {start_date.date()} to {end_date.date()}")
        print(f"Total days: {len(dates)}")
        print(f"Report types: {', '.join(report_types)}")
        print(f"Output: {self.output_dir}")
        print(f"Max concurrent downloads: {max_concurrent}\n")

        # Download annual files if requested
        if download_annual:
            years = set(d.year for d in dates)
            print(f"Downloading annual fuel mix files for years: {sorted(years)}")
            for year in sorted(years):
                await self.download_annual_fuel_mix(year)
            print()

        # Download daily data
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

    parser = argparse.ArgumentParser(
        description="Download MISO generation and fuel mix data"
    )
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
        choices=list(GEN_REPORT_TYPES.keys()),
        default=["fuel_mix_5min"],
        help="Report types to download"
    )
    parser.add_argument(
        "--download-annual",
        action="store_true",
        help="Also download annual historical fuel mix files"
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
    async with MISOGenerationDownloader(args.output_dir) as downloader:
        await downloader.download_date_range(
            start_date=start_date,
            end_date=end_date,
            report_types=args.report_types,
            download_annual=args.download_annual,
            max_concurrent=args.max_concurrent
        )


if __name__ == "__main__":
    asyncio.run(main())

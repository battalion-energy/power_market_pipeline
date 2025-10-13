#!/usr/bin/env python3
"""
SPP Historical LMP Data Downloader

Downloads day-ahead and real-time LMP data from SPP's Marketplace portal.
Data is downloaded as CSV files and stored in the SPP_data directory.

URL Patterns (from portal.spp.org):
- Day-Ahead: https://portal.spp.org/file-browser-api/download/da-lmp-by-location?path=/YYYY/MM/By_Day/DA-LMP-SL-YYYYMMDD0100.csv
- Real-Time (daily): https://portal.spp.org/file-browser-api/download/rtbm-lmp-by-location?path=/YYYY/MM/By_Day/RTBM-LMP-DAILY-SL-YYYYMMDD.csv
- Real-Time (5-min): https://portal.spp.org/file-browser-api/download/rtbm-lmp-by-location?path=/YYYY/MM/By_Interval/DD/RTBM-LMP-SL-YYYYMMDDHHMM.csv

Available markets:
- da: Day-Ahead hourly LMP
- rt_daily: Real-Time daily aggregated (preferred for bulk downloads)
- rt_5min: Real-Time 5-minute interval data

Data structure:
- Location types: Hub, Interface, Settlement Location
- Columns: GMTIntervalEnd, Settlement Location, Pnode, LMP, MLC (loss), MCC (congestion), MEC (energy)
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
SPP_BASE_URL = "https://portal.spp.org/file-browser-api/download"
SPP_DATA_DIR = os.getenv("SPP_DATA_DIR", "/pool/ssd8tb/data/iso/SPP")

# Market endpoints
ENDPOINTS = {
    "da": "da-lmp-by-location",
    "rt_daily": "rtbm-lmp-by-location",
    "rt_5min": "rtbm-lmp-by-location",
}

# SPP Hub and Interface names (from gridstatus)
HUBS_AND_INTERFACES = {
    "AECI", "ALTW", "AMRN", "BLKW", "CLEC", "DPC", "EDDY", "EES",
    "ERCOTE", "ERCOTN", "GRE", "LAM345", "MCWEST", "MDU", "MEC",
    "MISO", "NSP", "OTP", "RCEAST", "SCSE", "SGE", "SPA", "SPC",
    "SPPNORTH_HUB", "SPPSOUTH_HUB"
}


class SPPLMPDownloader:
    """Downloads SPP LMP data from Marketplace portal."""

    def __init__(self, output_dir: str = SPP_DATA_DIR):
        self.output_dir = Path(output_dir)
        self.session: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self.session = httpx.AsyncClient(timeout=300.0)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.aclose()

    def _construct_da_url(self, date: datetime) -> str:
        """Construct URL for Day-Ahead LMP data."""
        year = date.strftime("%Y")
        month = date.strftime("%m")
        date_str = date.strftime("%Y%m%d")

        # DA files are named: DA-LMP-SL-YYYYMMDD0100.csv
        # Path structure: /YYYY/MM/By_Day/
        path = f"/{year}/{month}/By_Day/DA-LMP-SL-{date_str}0100.csv"
        return f"{SPP_BASE_URL}/{ENDPOINTS['da']}?path={path}"

    def _construct_rt_daily_url(self, date: datetime) -> str:
        """Construct URL for Real-Time daily LMP data."""
        year = date.strftime("%Y")
        month = date.strftime("%m")
        date_str = date.strftime("%Y%m%d")

        # RT daily files: RTBM-LMP-DAILY-SL-YYYYMMDD.csv
        # Path structure: /YYYY/MM/By_Day/
        path = f"/{year}/{month}/By_Day/RTBM-LMP-DAILY-SL-{date_str}.csv"
        return f"{SPP_BASE_URL}/{ENDPOINTS['rt_daily']}?path={path}"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def download_day(
        self,
        date: datetime,
        market_type: str,
        filter_hubs: bool = True
    ) -> Optional[pd.DataFrame]:
        """Download LMP data for a single day."""

        if market_type == "da":
            url = self._construct_da_url(date)
            market_name = "da_lmp"
        elif market_type == "rt_daily":
            url = self._construct_rt_daily_url(date)
            market_name = "rt_lmp_daily"
        else:
            raise ValueError(f"Unsupported market type: {market_type}")

        date_str = date.strftime("%Y%m%d")

        # Create output directory structure
        # Save nodal data to separate directory to avoid collision with hub files
        if filter_hubs:
            market_dir = self.output_dir / "csv_files" / market_name
            output_file = market_dir / f"SPP_{market_name}_{date_str}.csv"
        else:
            # Nodal data goes to separate directory with "_nodal" suffix
            market_dir = self.output_dir / "csv_files" / f"{market_name}_nodal"
            output_file = market_dir / f"SPP_{market_name}_nodal_{date_str}.csv"

        market_dir.mkdir(parents=True, exist_ok=True)

        # Skip if already downloaded
        if output_file.exists():
            print(f"✓ Already exists: {output_file.name}")
            return pd.read_csv(output_file)

        try:
            print(f"Downloading: {date_str} {market_type}...", end=" ")

            response = await self.session.get(url)

            if response.status_code == 404:
                print(f"✗ Not available (404)")
                return None

            response.raise_for_status()
            content = response.text

            # Parse CSV
            df = pd.read_csv(pd.io.common.StringIO(content))

            # Strip whitespace from column names
            df.columns = df.columns.str.strip()

            # RT daily files use different column names - standardize them
            if market_type == "rt_daily":
                df = df.rename(columns={
                    "GMT Interval": "GMTIntervalEnd",
                    "Settlement Location Name": "Settlement Location",
                    "PNODE Name": "PNode",
                })

            if filter_hubs:
                # Filter for hubs and interfaces only
                df = df[df['Settlement Location'].isin(HUBS_AND_INTERFACES)].copy()
                print(f"✓ Downloaded ({len(df)} hub/interface rows)")
            else:
                print(f"✓ Downloaded ({len(df)} rows)")

            # Save to file
            df.to_csv(output_file, index=False)

            return df

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
        market_types: List[str] = None,
        filter_hubs: bool = True,
        max_concurrent: int = 5
    ):
        """Download LMP data for a date range."""

        if market_types is None:
            market_types = ["da", "rt_daily"]

        # Generate all dates in range
        dates = []
        current = start_date
        while current <= end_date:
            dates.append(current)
            current += timedelta(days=1)

        print(f"\n=== SPP LMP Downloader ===")
        print(f"Date range: {start_date.date()} to {end_date.date()}")
        print(f"Total days: {len(dates)}")
        print(f"Markets: {', '.join(market_types)}")
        print(f"Output: {self.output_dir}")
        print(f"Filter hubs/interfaces only: {filter_hubs}")
        print(f"Max concurrent downloads: {max_concurrent}\n")

        # Download data
        semaphore = asyncio.Semaphore(max_concurrent)

        async def download_with_limit(date, market_type):
            async with semaphore:
                return await self.download_day(date, market_type, filter_hubs)

        tasks = []
        for market_type in market_types:
            for date in dates:
                tasks.append(download_with_limit(date, market_type))

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

    parser = argparse.ArgumentParser(description="Download SPP historical LMP data")
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
        "--markets",
        type=str,
        nargs="+",
        choices=["da", "rt_daily", "rt_5min"],
        default=["da", "rt_daily"],
        help="Market types to download"
    )
    parser.add_argument(
        "--all-nodes",
        action="store_true",
        help="Download all settlement locations (default: hubs/interfaces only)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=SPP_DATA_DIR,
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
    async with SPPLMPDownloader(args.output_dir) as downloader:
        await downloader.download_date_range(
            start_date=start_date,
            end_date=end_date,
            market_types=args.markets,
            filter_hubs=not args.all_nodes,
            max_concurrent=args.max_concurrent
        )


if __name__ == "__main__":
    asyncio.run(main())

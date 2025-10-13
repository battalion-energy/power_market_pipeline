#!/usr/bin/env python3
"""
SPP Ancillary Services (Operating Reserve Prices) Downloader

Downloads day-ahead and real-time ancillary service prices from SPP's Marketplace portal.
Data is downloaded as CSV files and stored in the SPP_data directory.

URL Patterns (from portal.spp.org):
- Day-Ahead MCP: https://portal.spp.org/file-browser-api/download/da-mcp?path=/YYYY/MM/DA-MCP-YYYYMMDD0100.csv
- Real-Time MCP (daily): https://portal.spp.org/file-browser-api/download/rtbm-mcp?path=/YYYY/MM/By_Day/RTBM-MCP-DAILY-YYYYMMDD.csv

Available markets:
- da_mcp: Day-Ahead Marginal Clearing Prices (hourly)
- rt_mcp_daily: Real-Time Marginal Clearing Prices (5-min, daily file)

Reserve Types:
- Reg Up/Dn: Regulation Up/Down
- Ramp Up/Dn: Ramp Up/Down
- Spin: Spinning Reserve
- Supp: Supplemental Reserve
- Unc Up: Uncertainty Reserve Up

Data structure:
- Columns: GMTIntervalEnd, Reserve Zone, Reg Up, Reg DN, Ramp Up, Ramp DN, Spin, Supp, Unc Up
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
    "da_mcp": "da-mcp",
    "rt_mcp_daily": "rtbm-mcp",
}


class SPPAncillaryServicesDownloader:
    """Downloads SPP Ancillary Services (MCP) data from Marketplace portal."""

    def __init__(self, output_dir: str = SPP_DATA_DIR):
        self.output_dir = Path(output_dir)
        self.session: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self.session = httpx.AsyncClient(timeout=300.0)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.aclose()

    def _construct_da_mcp_url(self, date: datetime) -> str:
        """Construct URL for Day-Ahead MCP data."""
        year = date.strftime("%Y")
        month = date.strftime("%m")
        date_str = date.strftime("%Y%m%d")

        # DA MCP files: DA-MCP-YYYYMMDD0100.csv
        # Path structure: /YYYY/MM/
        path = f"/{year}/{month}/DA-MCP-{date_str}0100.csv"
        return f"{SPP_BASE_URL}/{ENDPOINTS['da_mcp']}?path={path}"

    def _construct_rt_mcp_daily_url(self, date: datetime) -> str:
        """Construct URL for Real-Time MCP daily data."""
        year = date.strftime("%Y")
        month = date.strftime("%m")
        date_str = date.strftime("%Y%m%d")

        # RT MCP daily files: RTBM-MCP-DAILY-YYYYMMDD.csv
        # Path structure: /YYYY/MM/By_Day/
        path = f"/{year}/{month}/By_Day/RTBM-MCP-DAILY-{date_str}.csv"
        return f"{SPP_BASE_URL}/{ENDPOINTS['rt_mcp_daily']}?path={path}"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def download_day(
        self,
        date: datetime,
        market_type: str
    ) -> Optional[pd.DataFrame]:
        """Download Ancillary Services MCP data for a single day."""

        if market_type == "da_mcp":
            url = self._construct_da_mcp_url(date)
            market_name = "da_mcp"
        elif market_type == "rt_mcp_daily":
            url = self._construct_rt_mcp_daily_url(date)
            market_name = "rt_mcp_daily"
        else:
            raise ValueError(f"Unsupported market type: {market_type}")

        date_str = date.strftime("%Y%m%d")

        # Create output directory structure
        market_dir = self.output_dir / "csv_files" / market_name
        market_dir.mkdir(parents=True, exist_ok=True)

        # Output file path
        output_file = market_dir / f"SPP_{market_name}_{date_str}.csv"

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

            # Standardize column names
            if market_type == "da_mcp":
                # DA MCP columns are already mostly standard
                df = df.rename(columns={
                    "RegUP": "Reg Up",
                    "RegDN": "Reg DN",
                    "RampUP": "Ramp Up",
                    "RampDN": "Ramp DN",
                    "UncUP": "Unc Up",
                })
            elif market_type == "rt_mcp_daily":
                # RT MCP daily files have different column names
                df = df.rename(columns={
                    "RegUPService": "Reg Up Service",
                    "RegDNService": "Reg DN Service",
                    "RegUpMile": "Reg Up Mile",
                    "RegDNMile": "Reg DN Mile",
                    "RampUP": "Ramp Up",
                    "RampDN": "Ramp DN",
                    "UncUP": "Unc Up",
                })

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
        max_concurrent: int = 5
    ):
        """Download Ancillary Services MCP data for a date range."""

        if market_types is None:
            market_types = ["da_mcp", "rt_mcp_daily"]

        # Generate all dates in range
        dates = []
        current = start_date
        while current <= end_date:
            dates.append(current)
            current += timedelta(days=1)

        print(f"\n=== SPP Ancillary Services Downloader ===")
        print(f"Date range: {start_date.date()} to {end_date.date()}")
        print(f"Total days: {len(dates)}")
        print(f"Markets: {', '.join(market_types)}")
        print(f"Output: {self.output_dir}")
        print(f"Max concurrent downloads: {max_concurrent}\n")

        # Download data
        semaphore = asyncio.Semaphore(max_concurrent)

        async def download_with_limit(date, market_type):
            async with semaphore:
                return await self.download_day(date, market_type)

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

    parser = argparse.ArgumentParser(description="Download SPP Ancillary Services MCP data")
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
        choices=["da_mcp", "rt_mcp_daily"],
        default=["da_mcp", "rt_mcp_daily"],
        help="Market types to download"
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
    async with SPPAncillaryServicesDownloader(args.output_dir) as downloader:
        await downloader.download_date_range(
            start_date=start_date,
            end_date=end_date,
            market_types=args.markets,
            max_concurrent=args.max_concurrent
        )


if __name__ == "__main__":
    asyncio.run(main())

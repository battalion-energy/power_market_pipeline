#!/usr/bin/env python3
"""
MISO Historical LMP Data Downloader

Downloads day-ahead and real-time LMP data from MISO's market reports portal.
Data is downloaded as CSV files and stored in the MISO_data directory.

URL Pattern: https://docs.misoenergy.org/marketreports/{YYYYMMDD}_{market_type}_lmp.csv

Available markets:
- da_expost_lmp: Day-Ahead Ex-Post (actual) LMP
- da_exante_lmp: Day-Ahead Ex-Ante (forecasted) LMP
- rt_lmp_final: Real-Time Final LMP (hourly aggregated)

Data structure:
- Node types: Hub (aggregated hubs), Loadzone, Interface, Gennode
- Columns: Node, Type, Value, HE 1 through HE 24 (Hour Ending in EST)
- Each row contains LMP, MCC (congestion), MLC (loss) for each node
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

# Market types to download
MARKET_TYPES = {
    "da_expost": "da_expost_lmp",  # Day-Ahead Ex-Post (actual)
    "da_exante": "da_exante_lmp",  # Day-Ahead Ex-Ante (forecasted)
    "rt_final": "rt_lmp_final",    # Real-Time Final (hourly)
}


class MISOLMPDownloader:
    """Downloads MISO LMP data from market reports portal."""

    def __init__(self, output_dir: str = MISO_DATA_DIR):
        self.output_dir = Path(output_dir)
        self.session: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self.session = httpx.AsyncClient(timeout=300.0)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.aclose()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def download_day(
        self,
        date: datetime,
        market_type: str,
        filter_hubs: bool = True
    ) -> Optional[pd.DataFrame]:
        """Download LMP data for a single day."""

        date_str = date.strftime("%Y%m%d")
        market_file = MARKET_TYPES[market_type]
        url = f"{MISO_BASE_URL}/{date_str}_{market_file}.csv"

        # Create output directory structure
        market_dir = self.output_dir / market_type
        market_dir.mkdir(parents=True, exist_ok=True)

        # Output file path
        output_file = market_dir / f"{date_str}_{market_file}.csv"

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

            # Save raw file
            output_file.write_text(content)

            # Parse and optionally filter
            # Skip first 4 header rows: title, date, empty line, EST note line
            df = pd.read_csv(output_file, skiprows=4)

            if filter_hubs:
                # Filter for hub-level data only
                df = df[df['Type'] == 'Hub'].copy()

                # Save filtered version
                hub_file = market_dir / f"{date_str}_{market_file}_hubs_only.csv"
                df.to_csv(hub_file, index=False)
                print(f"✓ Downloaded ({len(df)} hubs)")
            else:
                print(f"✓ Downloaded (all nodes)")

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
            market_types = list(MARKET_TYPES.keys())

        # Generate all dates in range
        dates = []
        current = start_date
        while current <= end_date:
            dates.append(current)
            current += timedelta(days=1)

        print(f"\n=== MISO LMP Downloader ===")
        print(f"Date range: {start_date.date()} to {end_date.date()}")
        print(f"Total days: {len(dates)}")
        print(f"Markets: {', '.join(market_types)}")
        print(f"Output: {self.output_dir}")
        print(f"Filter hubs only: {filter_hubs}")
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

    parser = argparse.ArgumentParser(description="Download MISO historical LMP data")
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
        choices=list(MARKET_TYPES.keys()),
        default=list(MARKET_TYPES.keys()),
        help="Market types to download"
    )
    parser.add_argument(
        "--all-nodes",
        action="store_true",
        help="Download all nodes (default: hubs only)"
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
    async with MISOLMPDownloader(args.output_dir) as downloader:
        await downloader.download_date_range(
            start_date=start_date,
            end_date=end_date,
            market_types=args.markets,
            filter_hubs=not args.all_nodes,
            max_concurrent=args.max_concurrent
        )


if __name__ == "__main__":
    asyncio.run(main())

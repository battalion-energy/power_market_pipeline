#!/usr/bin/env python3
"""
MISO Data Exchange API Historical Data Downloader

Downloads historical LMP and ancillary services data using MISO's Data Exchange API.
This complements the CSV downloader by accessing data from 2019-2023 that's not
available via the public CSV downloads.

Authentication: Bearer token using API keys
Rate Limits: 100 calls/minute, 24,000 calls/day
"""

import asyncio
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any

import httpx
import pandas as pd
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables
load_dotenv()

# Configuration
PRICING_API_BASE = "https://apim.misoenergy.org/pricing/v1"
LOAD_GEN_API_BASE = "https://apim.misoenergy.org/lgi/v1"

PRICING_API_KEY = os.getenv("MISO_PRICING_API_KEY")
LOAD_GEN_API_KEY = os.getenv("MISO_LOAD_AND_GEN_API_KEY")
MISO_DATA_DIR = os.getenv("MISO_DATA_DIR", "/pool/ssd8tb/data/iso/MISO/api_data")

# Market types and endpoints
PRICING_ENDPOINTS = {
    "da_exante_lmp": "/day-ahead/{date}/lmp-exante",
    "da_expost_lmp": "/day-ahead/{date}/lmp-expost",
    "rt_exante_lmp": "/real-time/{date}/lmp-exante",
    "rt_expost_lmp": "/real-time/{date}/lmp-expost",
    "da_exante_mcp": "/day-ahead/{date}/mcp-exante",
    "da_expost_mcp": "/day-ahead/{date}/mcp-expost",
    "rt_exante_mcp": "/real-time/{date}/mcp-exante",
    "rt_expost_mcp": "/real-time/{date}/mcp-expost",
}


class MISOAPIDownloader:
    """Downloads MISO data from Data Exchange API."""

    def __init__(
        self,
        pricing_api_key: str,
        load_gen_api_key: str,
        output_dir: str = MISO_DATA_DIR,
        max_concurrent: int = 5  # Conservative to respect rate limits
    ):
        self.pricing_api_key = pricing_api_key
        self.load_gen_api_key = load_gen_api_key
        self.output_dir = Path(output_dir)
        self.max_concurrent = max_concurrent
        self.session: Optional[httpx.AsyncClient] = None

        # Rate limiting
        self.calls_per_minute = 0
        self.calls_per_day = 0
        self.minute_start = datetime.now()
        self.day_start = datetime.now()

    async def __aenter__(self):
        self.session = httpx.AsyncClient(timeout=300.0)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.aclose()

    async def _check_rate_limit(self):
        """Enforce rate limits: 100/min, 24000/day"""
        now = datetime.now()

        # Reset minute counter
        if (now - self.minute_start).total_seconds() >= 60:
            self.calls_per_minute = 0
            self.minute_start = now

        # Reset day counter
        if (now - self.day_start).total_seconds() >= 86400:
            self.calls_per_day = 0
            self.day_start = now

        # Wait if approaching limits
        if self.calls_per_minute >= 95:
            wait_time = 60 - (now - self.minute_start).total_seconds()
            if wait_time > 0:
                print(f"Rate limit: waiting {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)
                self.calls_per_minute = 0
                self.minute_start = datetime.now()

        if self.calls_per_day >= 23900:
            print("Daily rate limit reached! Stopping.")
            raise Exception("Daily rate limit reached")

        self.calls_per_minute += 1
        self.calls_per_day += 1

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def _fetch_page(
        self,
        url: str,
        api_key: str,
        params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Fetch a single page from the API."""
        await self._check_rate_limit()

        headers = {
            "Ocp-Apim-Subscription-Key": api_key,
            "Accept": "application/json"
        }

        response = await self.session.get(url, headers=headers, params=params)

        if response.status_code == 404:
            return None

        response.raise_for_status()
        return response.json()

    async def _fetch_all_pages(
        self,
        url: str,
        api_key: str,
        params: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Fetch all pages for a given endpoint."""
        all_data = []
        page_number = 0

        while True:
            page_params = {**(params or {}), "pageNumber": page_number}

            result = await self._fetch_page(url, api_key, page_params)

            if result is None:
                break

            data = result.get("data", [])
            if not data:
                break

            all_data.extend(data)

            # Check if last page
            page_info = result.get("page", {})
            if page_info.get("lastPage", True):
                break

            page_number += 1

        return all_data

    async def download_pricing_data(
        self,
        market_type: str,
        date: datetime,
        filter_hubs: bool = True
    ) -> Optional[pd.DataFrame]:
        """Download pricing data for a single day."""

        if market_type not in PRICING_ENDPOINTS:
            raise ValueError(f"Unknown market type: {market_type}")

        date_str = date.strftime("%Y-%m-%d")
        endpoint = PRICING_ENDPOINTS[market_type].format(date=date_str)
        url = f"{PRICING_API_BASE}{endpoint}"

        # Create output directory
        market_dir = self.output_dir / "pricing" / market_type
        market_dir.mkdir(parents=True, exist_ok=True)

        # Output file
        output_file = market_dir / f"{date.strftime('%Y%m%d')}_{market_type}.json"

        # Skip if exists
        if output_file.exists():
            print(f"✓ Already exists: {output_file.name}")
            return pd.read_json(output_file)

        try:
            print(f"Downloading: {date_str} {market_type}...", end=" ")

            # Fetch data (no pagination needed - API returns all data in one response)
            result = await self._fetch_page(url, self.pricing_api_key)

            if result is None:
                print(f"✗ No data available")
                return None

            # Extract data array
            data = result.get("data", [])
            if not data:
                print(f"✗ No data available")
                return None

            # Save raw JSON
            output_file.write_text(json.dumps(data, indent=2))

            # Convert to DataFrame
            df = pd.DataFrame(data)

            # Filter for hubs if requested (node name contains common hub patterns)
            if filter_hubs and "node" in df.columns:
                hub_patterns = [".HUB", ".AZ", ".ARR"]
                mask = df["node"].str.contains("|".join(hub_patterns), case=False, na=False)
                df = df[mask].copy()

            # Save CSV version
            csv_file = market_dir / f"{date.strftime('%Y%m%d')}_{market_type}.csv"
            df.to_csv(csv_file, index=False)

            print(f"✓ Downloaded ({len(df)} records)")
            return df

        except httpx.HTTPError as e:
            print(f"✗ HTTP Error: {e}")
            return None
        except Exception as e:
            print(f"✗ Error: {e}")
            return None

    async def download_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        market_types: List[str] = None,
        filter_hubs: bool = True
    ):
        """Download data for a date range."""

        if market_types is None:
            # Default: DA and RT ex-post LMP + MCP
            market_types = [
                "da_expost_lmp",
                "da_expost_mcp",
                "rt_expost_lmp",
                "rt_expost_mcp"
            ]

        # Generate dates
        dates = []
        current = start_date
        while current <= end_date:
            dates.append(current)
            current += timedelta(days=1)

        print(f"\n=== MISO API Downloader ===")
        print(f"Date range: {start_date.date()} to {end_date.date()}")
        print(f"Total days: {len(dates)}")
        print(f"Markets: {', '.join(market_types)}")
        print(f"Output: {self.output_dir}")
        print(f"Filter hubs only: {filter_hubs}")
        print(f"Max concurrent: {self.max_concurrent}")
        print(f"Rate limits: 100/min, 24,000/day\n")

        # Download with concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def download_with_limit(market_type, date):
            async with semaphore:
                return await self.download_pricing_data(market_type, date, filter_hubs)

        tasks = []
        for market_type in market_types:
            for date in dates:
                tasks.append(download_with_limit(market_type, date))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Summary
        successful = sum(1 for r in results if r is not None and not isinstance(r, Exception))
        failed = sum(1 for r in results if isinstance(r, Exception))
        not_available = sum(1 for r in results if r is None)

        print(f"\n=== Download Summary ===")
        print(f"Successful: {successful}")
        print(f"Not available: {not_available}")
        print(f"Failed: {failed}")
        print(f"\nAPI calls used today: {self.calls_per_day}")


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download MISO historical data using Data Exchange API"
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
        "--markets",
        type=str,
        nargs="+",
        choices=list(PRICING_ENDPOINTS.keys()),
        default=["da_expost_lmp", "da_expost_mcp", "rt_expost_lmp", "rt_expost_mcp"],
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
        help="Maximum concurrent requests (careful with rate limits!)"
    )

    args = parser.parse_args()

    # Check API keys
    if not PRICING_API_KEY:
        print("ERROR: MISO_PRICING_API_KEY not found in .env file!")
        return

    if not LOAD_GEN_API_KEY:
        print("ERROR: MISO_LOAD_AND_GEN_API_KEY not found in .env file!")
        return

    # Parse dates
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d") if args.end_date else datetime.now()

    # Download data
    async with MISOAPIDownloader(
        PRICING_API_KEY,
        LOAD_GEN_API_KEY,
        args.output_dir,
        args.max_concurrent
    ) as downloader:
        await downloader.download_date_range(
            start_date=start_date,
            end_date=end_date,
            market_types=args.markets,
            filter_hubs=not args.all_nodes
        )


if __name__ == "__main__":
    asyncio.run(main())

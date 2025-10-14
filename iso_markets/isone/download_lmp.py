#!/usr/bin/env python3
"""
ISO-NE LMP Data Downloader

Downloads day-ahead and real-time LMP data from ISO-NE Web Services API:
- Day-Ahead LMP: Hourly prices for next operating day
- Real-Time LMP: 5-minute actual prices

API Documentation: https://webservices.iso-ne.com/docs/v1.1/

Adapted from: /home/enrico/projects/solar_sim/data_scripts/isone/ISONE-data-fetch.py
"""

import asyncio
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict

import httpx
import pandas as pd
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables
load_dotenv()

# Configuration
ISONE_API_BASE = "https://webservices.iso-ne.com/api/v1.1"
ISONE_USERNAME = os.getenv("ISONE_USERNAME")
ISONE_PASSWORD = os.getenv("ISONE_PASSWORD")
ISONE_DATA_DIR = os.getenv("ISONE_DATA_DIR", "/pool/ssd8tb/data/iso/ISONE")

# Hub locations (major pricing points)
HUB_LOCATIONS = [
    4000,  # .H.INTERNAL_HUB
    4001,  # .Z.MAINE
    4002,  # .Z.NEWHAMPSHIRE
    4003,  # .Z.VERMONT
    4004,  # .Z.CONNECTICUT
    4005,  # .Z.RHODEISLAND
    4006,  # .Z.SEMASS
    4007,  # .Z.WCMASS
    4008,  # .Z.NEMASSBOST
]


class ISONELMPDownloader:
    """Downloads ISO-NE LMP data from Web Services API."""

    def __init__(
        self,
        username: str,
        password: str,
        output_dir: str = ISONE_DATA_DIR,
        max_concurrent: int = 3,
        request_delay: float = 2.0
    ):
        self.username = username
        self.password = password
        self.output_dir = Path(output_dir)
        self.max_concurrent = max_concurrent
        self.request_delay = request_delay
        self.session: Optional[httpx.AsyncClient] = None
        self.all_locations: List[int] = []

    async def __aenter__(self):
        self.session = httpx.AsyncClient(
            auth=(self.username, self.password),
            timeout=180.0
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.aclose()

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=300))
    async def fetch_all_locations(self) -> List[int]:
        """
        Fetch all location IDs from the API.

        Returns list of location IDs for nodal downloads.
        """
        if self.all_locations:
            return self.all_locations

        url = f"{ISONE_API_BASE}/locations/all.json"

        try:
            print(f"Fetching all location IDs...", end=" ")

            response = await self.session.get(url)

            if response.status_code == 429:
                print(f"✗ Rate limited (429), retrying with backoff...")
                raise httpx.HTTPStatusError("Rate limited", request=response.request, response=response)

            response.raise_for_status()
            data = response.json()

            # Extract location IDs
            locations = data.get('Locations', {}).get('Location', [])
            self.all_locations = [loc['LocationID'] for loc in locations if 'LocationID' in loc]

            print(f"✓ Found {len(self.all_locations)} locations")
            return self.all_locations

        except httpx.HTTPError as e:
            print(f"✗ HTTP Error: {e}")
            return []
        except Exception as e:
            print(f"✗ Error: {e}")
            return []

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=300))
    async def download_da_lmp(
        self,
        date: datetime,
        location_id: int
    ) -> Optional[pd.DataFrame]:
        """
        Download day-ahead hourly LMP for a single day and location.

        Returns:
        - BeginDate: Timestamp
        - LocId: Location ID
        - Location: Location name
        - LmpTotal: Total LMP ($/MWh)
        - EnergyComponent: Energy component
        - CongestionComponent: Congestion component
        - LossComponent: Loss component
        """
        date_str = date.strftime("%Y%m%d")
        url = f"{ISONE_API_BASE}/hourlylmp/da/final/day/{date_str}/location/{location_id}.json"

        # Create output directory
        da_dir = self.output_dir / "csv_files" / "da_lmp"
        da_dir.mkdir(parents=True, exist_ok=True)

        # Output file
        output_file = da_dir / f"{date_str}_da_lmp_loc{location_id}.csv"

        # Skip if exists
        if output_file.exists():
            return pd.read_csv(output_file)

        try:
            print(f"Downloading: {date_str} DA LMP location {location_id}...", end=" ")

            response = await self.session.get(url)

            if response.status_code == 404:
                print(f"✗ Not available (404)")
                return None

            if response.status_code == 429:
                print(f"✗ Rate limited (429), retrying with backoff...")
                raise httpx.HTTPStatusError("Rate limited", request=response.request, response=response)

            response.raise_for_status()
            data = response.json()

            # Extract data
            lmp_data = data.get('HourlyLmps', {}).get('HourlyLmp', [])

            if not lmp_data:
                print(f"✗ No data available")
                return None

            # Convert to DataFrame
            records = []
            for item in lmp_data:
                # Parse location object if it's a dict
                location = item.get("Location", {})
                if isinstance(location, dict):
                    loc_id = location.get("@LocId")
                    loc_type = location.get("@LocType")
                    loc_name = location.get("$")
                else:
                    loc_id = None
                    loc_type = None
                    loc_name = str(location) if location else None

                records.append({
                    "BeginDate": item["BeginDate"],
                    "LocId": loc_id,
                    "LocType": loc_type,
                    "LocationName": loc_name,
                    "LmpTotal": item.get("LmpTotal"),
                    "EnergyComponent": item.get("EnergyComponent"),
                    "CongestionComponent": item.get("CongestionComponent"),
                    "LossComponent": item.get("LossComponent")
                })

            df = pd.DataFrame(records)

            # Save to CSV
            df.to_csv(output_file, index=False)

            print(f"✓ Downloaded ({len(df)} records)")
            return df

        except httpx.HTTPError as e:
            print(f"✗ HTTP Error: {e}")
            return None
        except Exception as e:
            print(f"✗ Error: {e}")
            return None

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=300))
    async def download_rt_lmp(
        self,
        date: datetime,
        location_id: int
    ) -> Optional[pd.DataFrame]:
        """
        Download real-time 5-minute LMP for a single day and location.

        Returns:
        - BeginDate: Timestamp
        - LocId: Location ID
        - Location: Location name
        - LmpTotal: Total LMP ($/MWh)
        - EnergyComponent: Energy component
        - CongestionComponent: Congestion component (if available)
        - LossComponent: Loss component
        """
        date_str = date.strftime("%Y%m%d")
        url = f"{ISONE_API_BASE}/fiveminutelmp/final/day/{date_str}/location/{location_id}.json"

        # Create output directory
        rt_dir = self.output_dir / "csv_files" / "rt_lmp"
        rt_dir.mkdir(parents=True, exist_ok=True)

        # Output file
        output_file = rt_dir / f"{date_str}_rt_lmp_loc{location_id}.csv"

        # Skip if exists
        if output_file.exists():
            return pd.read_csv(output_file)

        try:
            print(f"Downloading: {date_str} RT LMP location {location_id}...", end=" ")

            response = await self.session.get(url)

            if response.status_code == 404:
                print(f"✗ Not available (404)")
                return None

            if response.status_code == 429:
                print(f"✗ Rate limited (429), retrying with backoff...")
                raise httpx.HTTPStatusError("Rate limited", request=response.request, response=response)

            response.raise_for_status()
            data = response.json()

            # Extract data
            lmp_data = data.get('FiveMinLmps', {}).get('FiveMinLmp', [])

            if not lmp_data:
                print(f"✗ No data available")
                return None

            # Convert to DataFrame
            records = []
            for item in lmp_data:
                # Parse location object if it's a dict
                location = item.get("Location", {})
                if isinstance(location, dict):
                    loc_id = location.get("@LocId")
                    loc_type = location.get("@LocType")
                    loc_name = location.get("$")
                else:
                    loc_id = None
                    loc_type = None
                    loc_name = str(location) if location else None

                records.append({
                    "BeginDate": item["BeginDate"],
                    "LocId": loc_id,
                    "LocType": loc_type,
                    "LocationName": loc_name,
                    "LmpTotal": item.get("LmpTotal"),
                    "EnergyComponent": item.get("EnergyComponent"),
                    "CongestionComponent": item.get("CongestionComponent"),
                    "LossComponent": item.get("LossComponent")
                })

            df = pd.DataFrame(records)

            # Save to CSV
            df.to_csv(output_file, index=False)

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
        filter_hubs: bool = False,
        reverse: bool = False
    ):
        """
        Download LMP data for a date range.

        Args:
            start_date: Start date
            end_date: End date
            market_types: List of market types ['da', 'rt']
            filter_hubs: If True, download only hub locations; if False, download all nodes
            reverse: If True, download in reverse chronological order (newest first)
        """

        if market_types is None:
            market_types = ["da", "rt"]

        # Get locations to download
        if filter_hubs:
            locations = HUB_LOCATIONS
            print(f"Downloading hub locations only: {len(locations)} hubs")
        else:
            # Fetch all locations from API
            locations = await self.fetch_all_locations()
            if not locations:
                print("ERROR: Could not fetch location IDs")
                return
            print(f"Downloading all nodal locations: {len(locations)} nodes")

        # Generate dates
        dates = []
        current = start_date
        while current <= end_date:
            dates.append(current)
            current += timedelta(days=1)

        # Reverse dates if requested (download newest first)
        if reverse:
            dates.reverse()

        print(f"\n=== ISO-NE LMP Downloader ===")
        print(f"Date range: {start_date.date()} to {end_date.date()}")
        print(f"Total days: {len(dates)}")
        print(f"Market types: {', '.join(market_types)}")
        print(f"Locations: {len(locations)} ({'hubs only' if filter_hubs else 'all nodes'})")
        print(f"Output: {self.output_dir}")
        print(f"Max concurrent: {self.max_concurrent}\n")

        # Download with concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def download_with_limit(task_fn):
            async with semaphore:
                return await task_fn()

        tasks = []

        # Add DA LMP tasks
        if "da" in market_types:
            for location_id in locations:
                for date in dates:
                    tasks.append(
                        download_with_limit(
                            lambda d=date, loc=location_id: self.download_da_lmp(d, loc)
                        )
                    )

        # Add RT LMP tasks
        if "rt" in market_types:
            for location_id in locations:
                for date in dates:
                    tasks.append(
                        download_with_limit(
                            lambda d=date, loc=location_id: self.download_rt_lmp(d, loc)
                        )
                    )

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
        description="Download ISO-NE historical LMP data"
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
        "--market-types",
        type=str,
        nargs="+",
        choices=["da", "rt"],
        default=["da", "rt"],
        help="Market types to download"
    )
    parser.add_argument(
        "--hubs-only",
        action="store_true",
        help="Download only hub locations (not all nodes)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=ISONE_DATA_DIR,
        help="Output directory"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=3,
        help="Maximum concurrent downloads"
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=2.0,
        help="Delay between requests in seconds (default: 2.0)"
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Download in reverse chronological order (newest first)"
    )

    args = parser.parse_args()

    # Check credentials
    if not ISONE_USERNAME or not ISONE_PASSWORD:
        print("ERROR: ISONE_USERNAME and ISONE_PASSWORD must be set in .env file!")
        return

    # Parse dates
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d") if args.end_date else datetime.now()

    # Download data
    async with ISONELMPDownloader(
        ISONE_USERNAME,
        ISONE_PASSWORD,
        args.output_dir,
        args.max_concurrent,
        args.request_delay
    ) as downloader:
        await downloader.download_date_range(
            start_date=start_date,
            end_date=end_date,
            market_types=args.market_types,
            filter_hubs=args.hubs_only,
            reverse=args.reverse
        )


if __name__ == "__main__":
    asyncio.run(main())

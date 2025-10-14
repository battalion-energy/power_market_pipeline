#!/usr/bin/env python3
"""
ISO-NE Ancillary Services Data Downloader

Downloads ancillary services data from ISO-NE Web Services API:
- Frequency Regulation (FR): RegServiceClearingPrice, RegCapacityClearingPrice
- Ten-Minute Reserves: Spinning (TMSR), Non-Spinning (TMNSR), 30-min Operating (TMOR)

API Documentation: https://www.iso-ne.com/isoexpress/web/reports/operations/-/tree/energy-reserve-and-regulation

Reserve Zones:
- 7000: ROS (Rest of System)
- 7001: SEMA (Southeastern Massachusetts)
- 7002: WCMA (Western/Central Massachusetts)
- 7003: CMA (Central Massachusetts)
- 7004: NEMA (Northeastern Massachusetts)
- 7005: CT (Connecticut)
- 7006: RI (Rhode Island)
- 7007: SWCT (Southwestern Connecticut)
- 7008: NOR (Northern New England)
- 7009: ME (Maine)
- 7010: NH (New Hampshire)
- 7011: VT (Vermont)

Adapted from: /home/enrico/projects/solar_sim/data_scripts/isone/ISONE-data-fetch.py
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
ISONE_API_BASE = "https://webservices.iso-ne.com/api/v1.1"
ISONE_USERNAME = os.getenv("ISONE_USERNAME")
ISONE_PASSWORD = os.getenv("ISONE_PASSWORD")
ISONE_DATA_DIR = os.getenv("ISONE_DATA_DIR", "/pool/ssd8tb/data/iso/ISONE")

# Reserve zones (documented in ISO-NE API)
RESERVE_ZONES = {
    7000: "ROS",      # Rest of System
    7001: "SEMA",     # Southeastern Massachusetts
    7002: "WCMA",     # Western/Central Massachusetts
    7003: "CMA",      # Central Massachusetts
    7004: "NEMA",     # Northeastern Massachusetts
    7005: "CT",       # Connecticut
    7006: "RI",       # Rhode Island
    7007: "SWCT",     # Southwestern Connecticut
    7008: "NOR",      # Northern New England
    7009: "ME",       # Maine
    7010: "NH",       # New Hampshire
    7011: "VT",       # Vermont
}


class ISONEASDownloader:
    """Downloads ISO-NE Ancillary Services data from Web Services API."""

    def __init__(
        self,
        username: str,
        password: str,
        output_dir: str = ISONE_DATA_DIR,
        max_concurrent: int = 3,
        request_delay: float = 1.0
    ):
        self.username = username
        self.password = password
        self.output_dir = Path(output_dir)
        self.max_concurrent = max_concurrent
        self.request_delay = request_delay
        self.session: Optional[httpx.AsyncClient] = None

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
    async def download_frequency_regulation(
        self,
        date: datetime
    ) -> Optional[pd.DataFrame]:
        """
        Download frequency regulation clearing prices for a single day.

        Returns:
        - RegServiceClearingPrice: Price for regulation service ($/MW)
        - RegCapacityClearingPrice: Price for regulation capacity ($/MW)
        """
        date_str = date.strftime("%Y%m%d")
        url = f"{ISONE_API_BASE}/fiveminutercp/final/day/{date_str}.json"

        # Create output directory
        fr_dir = self.output_dir / "csv_files" / "frequency_regulation"
        fr_dir.mkdir(parents=True, exist_ok=True)

        # Output file
        output_file = fr_dir / f"{date_str}_freq_reg.csv"

        # Skip if exists
        if output_file.exists():
            # print(f"✓ Already exists: {output_file.name}")
            return pd.read_csv(output_file)

        try:
            print(f"Downloading: {date_str} frequency regulation...", end=" ")

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
            reserve_prices = data.get('FiveMinRcps', {}).get('FiveMinRcp', [])

            if not reserve_prices:
                print(f"✗ No data available")
                return None

            # Convert to DataFrame
            records = []
            for item in reserve_prices:
                records.append({
                    "BeginDate": item["BeginDate"],
                    "RegServiceClearingPrice": item["RegServiceClearingPrice"],
                    "RegCapacityClearingPrice": item["RegCapacityClearingPrice"]
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
    async def download_reserve_prices(
        self,
        date: datetime,
        zone_id: int
    ) -> Optional[pd.DataFrame]:
        """
        Download 10-minute and 30-minute reserve prices for a single day and zone.

        Returns prices for:
        - TMSR (Ten-Minute Spinning Reserve)
        - TMNSR (Ten-Minute Non-Spinning Reserve)
        - TMOR (Thirty-Minute Operating Reserve)
        """
        date_str = date.strftime("%Y%m%d")
        zone_name = RESERVE_ZONES.get(zone_id, f"Zone{zone_id}")
        url = f"{ISONE_API_BASE}/fiveminutereserveprice/final/day/{date_str}/reserveZone/{zone_id}.json"

        # Create output directory
        reserve_dir = self.output_dir / "csv_files" / "reserve_prices"
        reserve_dir.mkdir(parents=True, exist_ok=True)

        # Output file
        output_file = reserve_dir / f"{date_str}_reserve_zone{zone_id}_{zone_name}.csv"

        # Skip if exists
        if output_file.exists():
            # print(f"✓ Already exists: {output_file.name}")
            return pd.read_csv(output_file)

        try:
            print(f"Downloading: {date_str} reserve zone {zone_id} ({zone_name})...", end=" ")

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
            reserve_prices = data.get('FiveMinReservePrices', {}).get('FiveMinReservePrice', [])

            if not reserve_prices:
                print(f"✗ No data available")
                return None

            # Convert to DataFrame
            records = []
            for item in reserve_prices:
                records.append({
                    "BeginDate": item["BeginDate"],
                    "ReserveZoneId": item["ReserveZoneId"],
                    "ReserveZoneName": item["ReserveZoneName"],
                    "TenMinSpinRequirement": item.get("TenMinSpinRequirement"),
                    "Total10MinRequirement": item.get("Total10MinRequirement"),
                    "Total30MinRequirement": item.get("Total30MinRequirement"),
                    "TmsrDesignatedMw": item.get("TmsrDesignatedMw"),
                    "TmnsrDesignatedMw": item.get("TmnsrDesignatedMw"),
                    "TmorDesignatedMw": item.get("TmorDesignatedMw"),
                    "TmsrClearingPrice": item.get("TmsrClearingPrice"),
                    "TmnsrClearingPrice": item.get("TmnsrClearingPrice"),
                    "TmorClearingPrice": item.get("TmorClearingPrice")
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
        data_types: List[str] = None,
        reserve_zones: List[int] = None,
        reverse: bool = False
    ):
        """
        Download ancillary services data for a date range.

        Args:
            start_date: Start date
            end_date: End date
            data_types: List of data types to download ['freq_reg', 'reserves']
            reserve_zones: List of reserve zone IDs (only for 'reserves' type)
            reverse: If True, download in reverse chronological order (newest first)
        """

        if data_types is None:
            data_types = ["freq_reg", "reserves"]

        if reserve_zones is None:
            # Default to main zones
            reserve_zones = [7000, 7001, 7002, 7003]

        # Generate dates
        dates = []
        current = start_date
        while current <= end_date:
            dates.append(current)
            current += timedelta(days=1)

        # Reverse dates if requested (download newest first)
        if reverse:
            dates.reverse()

        print(f"\n=== ISO-NE Ancillary Services Downloader ===")
        print(f"Date range: {start_date.date()} to {end_date.date()}")
        print(f"Total days: {len(dates)}")
        print(f"Data types: {', '.join(data_types)}")
        if "reserves" in data_types:
            zone_names = [f"{z} ({RESERVE_ZONES.get(z, 'Unknown')})" for z in reserve_zones]
            print(f"Reserve zones: {', '.join(zone_names)}")
        print(f"Output: {self.output_dir}")
        print(f"Max concurrent: {self.max_concurrent}\n")

        # Download with concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def download_with_limit(task_fn):
            async with semaphore:
                return await task_fn()

        tasks = []

        # Add frequency regulation tasks
        if "freq_reg" in data_types:
            for date in dates:
                tasks.append(
                    download_with_limit(
                        lambda d=date: self.download_frequency_regulation(d)
                    )
                )

        # Add reserve price tasks
        if "reserves" in data_types:
            for zone_id in reserve_zones:
                for date in dates:
                    tasks.append(
                        download_with_limit(
                            lambda d=date, z=zone_id: self.download_reserve_prices(d, z)
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
        description="Download ISO-NE historical ancillary services data"
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
        "--data-types",
        type=str,
        nargs="+",
        choices=["freq_reg", "reserves"],
        default=["freq_reg", "reserves"],
        help="Data types to download"
    )
    parser.add_argument(
        "--reserve-zones",
        type=int,
        nargs="+",
        default=[7000, 7001, 7002, 7003],
        help="Reserve zone IDs (default: 7000 7001 7002 7003)"
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
    async with ISONEASDownloader(
        ISONE_USERNAME,
        ISONE_PASSWORD,
        args.output_dir,
        args.max_concurrent,
        args.request_delay
    ) as downloader:
        await downloader.download_date_range(
            start_date=start_date,
            end_date=end_date,
            data_types=args.data_types,
            reserve_zones=args.reserve_zones,
            reverse=args.reverse
        )


if __name__ == "__main__":
    asyncio.run(main())

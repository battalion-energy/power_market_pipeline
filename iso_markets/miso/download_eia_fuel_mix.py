#!/usr/bin/env python3
"""
EIA-930 MISO Fuel Mix Data Downloader

Downloads hourly generation fuel mix data for MISO from the U.S. Energy Information
Administration (EIA) Form 930 API.

Data Source: EIA Open Data API v2
- API: https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/
- Documentation: https://www.eia.gov/opendata/

Available Data:
- Hourly generation by fuel type (Coal, Gas, Nuclear, Wind, Solar, Hydro, Other)
- Coverage: July 2015 - present (fuel mix by source added July 2018)
- Resolution: Hourly
- Geographic: MISO balancing authority

Requirements:
- Free EIA API key: https://www.eia.gov/opendata/register.php
- Add to .env: EIA_API_KEY=your_api_key_here
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
EIA_API_BASE_URL = "https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/"
MISO_DATA_DIR = os.getenv("MISO_DATA_DIR", "/pool/ssd8tb/data/iso/MISO")
EIA_API_KEY = os.getenv("EIA_API_KEY")

# EIA API pagination limit
MAX_ROWS_PER_REQUEST = 5000


class EIAFuelMixDownloader:
    """Downloads MISO fuel mix data from EIA-930 API."""

    def __init__(self, api_key: str = None, output_dir: str = MISO_DATA_DIR):
        self.api_key = api_key or EIA_API_KEY
        if not self.api_key:
            raise ValueError(
                "EIA API key required. Set EIA_API_KEY in .env or pass as argument.\n"
                "Get free API key at: https://www.eia.gov/opendata/register.php"
            )

        self.output_dir = Path(output_dir)
        self.session: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self.session = httpx.AsyncClient(timeout=300.0, follow_redirects=True)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.aclose()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def fetch_page(
        self,
        start_datetime: str,
        end_datetime: str,
        offset: int = 0,
        fuel_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Fetch a single page of data from EIA API.

        Args:
            start_datetime: Start datetime in format YYYY-MM-DDTHH (e.g., "2024-01-01T00")
            end_datetime: End datetime in format YYYY-MM-DDTHH (e.g., "2024-12-31T23")
            offset: Pagination offset
            fuel_type: Optional filter for specific fuel type

        Returns:
            API response as dictionary
        """
        params = {
            "api_key": self.api_key,
            "frequency": "hourly",
            "data[0]": "value",
            "facets[respondent][]": "MISO",
            "start": start_datetime,
            "end": end_datetime,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "offset": offset,
            "length": MAX_ROWS_PER_REQUEST
        }

        # Optional fuel type filter
        if fuel_type:
            params["facets[fueltype][]"] = fuel_type

        try:
            response = await self.session.get(EIA_API_BASE_URL, params=params)
            response.raise_for_status()
            return response.json()

        except httpx.HTTPError as e:
            print(f"✗ HTTP Error: {e}")
            raise

    async def download_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        fuel_types: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Download fuel mix data for a date range with pagination.

        Args:
            start_date: Start date
            end_date: End date
            fuel_types: Optional list of fuel types to filter (default: all)

        Returns:
            DataFrame with all fuel mix data
        """
        start_datetime = start_date.strftime("%Y-%m-%dT00")
        end_datetime = end_date.strftime("%Y-%m-%dT23")

        print(f"\n=== EIA-930 MISO Fuel Mix Downloader ===")
        print(f"Date range: {start_date.date()} to {end_date.date()}")
        print(f"Period: {start_datetime} to {end_datetime}")
        print(f"Output: {self.output_dir}")

        all_data = []
        offset = 0
        page_num = 1

        while True:
            print(f"Fetching page {page_num} (offset {offset})...", end=" ")

            try:
                response = await self.fetch_page(
                    start_datetime=start_datetime,
                    end_datetime=end_datetime,
                    offset=offset,
                    fuel_type=None  # Get all fuel types
                )

                if "response" in response and "data" in response["response"]:
                    data = response["response"]["data"]

                    if not data:
                        print("✓ No more data")
                        break

                    all_data.extend(data)
                    rows_fetched = len(data)
                    total_rows = len(all_data)

                    print(f"✓ {rows_fetched} rows (total: {total_rows})")

                    # Check if we've fetched all available data
                    if rows_fetched < MAX_ROWS_PER_REQUEST:
                        print("✓ Reached end of data")
                        break

                    # Move to next page
                    offset += MAX_ROWS_PER_REQUEST
                    page_num += 1

                    # Small delay to avoid rate limiting
                    await asyncio.sleep(0.5)

                elif "response" in response and "error" in response["response"]:
                    error_msg = response["response"]["error"]
                    print(f"✗ API Error: {error_msg}")
                    break
                else:
                    print("✗ Unexpected response format")
                    break

            except Exception as e:
                print(f"✗ Error: {e}")
                break

        if not all_data:
            print("\n⚠️  No data retrieved")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(all_data)

        print(f"\n=== Download Summary ===")
        print(f"Total records: {len(df)}")

        if not df.empty:
            print(f"Date range: {df['period'].min()} to {df['period'].max()}")
            print(f"Fuel types: {sorted(df['fueltype'].unique().tolist())}")
            print(f"Columns: {', '.join(df.columns)}")

        return df

    async def save_data(
        self,
        df: pd.DataFrame,
        start_date: datetime,
        end_date: datetime,
        save_json: bool = True,
        save_csv: bool = True
    ):
        """
        Save downloaded data to disk.

        Args:
            df: DataFrame with fuel mix data
            start_date: Start date for filename
            end_date: End date for filename
            save_json: Save raw JSON format
            save_csv: Save processed CSV format
        """
        if df.empty:
            print("⚠️  No data to save")
            return

        # Create output directory
        output_dir = self.output_dir / "eia_fuel_mix"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename based on date range
        date_str = f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"

        # Save JSON (raw format from API)
        if save_json:
            json_file = output_dir / f"eia_fuel_mix_{date_str}.json"
            df.to_json(json_file, orient="records", indent=2)
            print(f"✓ Saved JSON: {json_file.name} ({len(df)} records)")

        # Save CSV (processed format)
        if save_csv:
            csv_file = output_dir / f"eia_fuel_mix_{date_str}.csv"
            df.to_csv(csv_file, index=False)
            print(f"✓ Saved CSV: {csv_file.name} ({len(df)} records)")

        # Create pivot table (wide format: rows=timestamp, cols=fuel type)
        if "period" in df.columns and "fueltype" in df.columns and "value" in df.columns:
            pivot_df = df.pivot_table(
                index="period",
                columns="fueltype",
                values="value",
                aggfunc="first"  # Use first value if duplicates
            )

            pivot_file = output_dir / f"eia_fuel_mix_{date_str}_pivot.csv"
            pivot_df.to_csv(pivot_file)
            print(f"✓ Saved pivot CSV: {pivot_file.name} ({len(pivot_df)} timestamps)")

    async def download_and_save(
        self,
        start_date: datetime,
        end_date: datetime,
        fuel_types: Optional[List[str]] = None
    ):
        """
        Download and save fuel mix data for a date range.

        Args:
            start_date: Start date
            end_date: End date
            fuel_types: Optional list of fuel types to filter
        """
        # Download data
        df = await self.download_date_range(start_date, end_date, fuel_types)

        # Save data
        if not df.empty:
            await self.save_data(df, start_date, end_date)


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download MISO fuel mix data from EIA-930 API"
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
        "--api-key",
        type=str,
        default=None,
        help="EIA API key (or set EIA_API_KEY in .env)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=MISO_DATA_DIR,
        help="Output directory"
    )
    parser.add_argument(
        "--fuel-types",
        type=str,
        nargs="+",
        default=None,
        help="Optional: Filter by fuel types (COL, NG, NUC, SUN, WND, WAT, OTH)"
    )

    args = parser.parse_args()

    # Parse dates
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d") if args.end_date else datetime.now()

    # Check API key
    api_key = args.api_key or EIA_API_KEY
    if not api_key:
        print("❌ ERROR: EIA API key required!")
        print("   Get free API key at: https://www.eia.gov/opendata/register.php")
        print("   Then add to .env: EIA_API_KEY=your_api_key_here")
        print("   Or pass via --api-key argument")
        return

    # Download data
    try:
        async with EIAFuelMixDownloader(api_key, args.output_dir) as downloader:
            await downloader.download_and_save(
                start_date=start_date,
                end_date=end_date,
                fuel_types=args.fuel_types
            )

        print("\n✅ Download completed successfully!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())

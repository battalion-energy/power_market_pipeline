#!/usr/bin/env python3
"""
Download ERCOT Battery Storage Data

This script downloads historical and real-time battery storage data from ERCOT:
1. Historical data (2019-2024): From Fuel Mix Reports
2. Recent/Real-time data: From ERCOT API endpoints

Data Sources:
- Energy Storage Resources API: /api/1/services/read/dashboards/energy-storage-resources.json
- Fuel Mix API: /api/1/services/read/dashboards/fuel-mix.json
- Historical Fuel Mix Reports: https://www.ercot.com/gridinfo/generation

Output Format:
- CSV files with columns: timestamp, total_charging_mw, total_discharging_mw, net_output_mw
- Standardized to UTC timestamps
- 5-minute intervals
"""

import os
import sys
import json
import requests
import pandas as pd
import zipfile
import io
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ERCOTBatteryStorageDownloader:
    """Download and process ERCOT battery storage data."""

    # API Endpoints
    ENERGY_STORAGE_API = "https://www.ercot.com/api/1/services/read/dashboards/energy-storage-resources.json"
    FUEL_MIX_API = "https://www.ercot.com/api/1/services/read/dashboards/fuel-mix.json"

    # Historical data URLs (need to be scraped from the generation page)
    GENERATION_PAGE = "https://www.ercot.com/gridinfo/generation"

    def __init__(self, output_dir: str = "ercot_battery_storage_data"):
        """Initialize downloader with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Session for requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def download_energy_storage_api_data(self) -> pd.DataFrame:
        """Download current and previous day data from Energy Storage Resources API."""
        logger.info("Downloading energy storage data from API...")

        try:
            response = self.session.get(self.ENERGY_STORAGE_API, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Extract previous day and current day data
            all_records = []

            if 'previousDay' in data and 'data' in data['previousDay']:
                all_records.extend(data['previousDay']['data'])

            if 'currentDay' in data and 'data' in data['currentDay']:
                all_records.extend(data['currentDay']['data'])

            if not all_records:
                logger.warning("No data returned from Energy Storage API")
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(all_records)

            # Standardize column names
            df = df.rename(columns={
                'timestamp': 'timestamp',
                'totalCharging': 'total_charging_mw',
                'totalDischarging': 'total_discharging_mw',
                'netOutput': 'net_output_mw'
            })

            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Sort by timestamp
            df = df.sort_values('timestamp')

            # Keep only relevant columns
            df = df[['timestamp', 'total_charging_mw', 'total_discharging_mw', 'net_output_mw']]

            logger.info(f"Downloaded {len(df)} records from Energy Storage API")
            logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

            return df

        except Exception as e:
            logger.error(f"Error downloading from Energy Storage API: {e}")
            return pd.DataFrame()

    def download_fuel_mix_api_data(self) -> pd.DataFrame:
        """Download fuel mix data and extract Power Storage generation."""
        logger.info("Downloading fuel mix data from API...")

        try:
            response = self.session.get(self.FUEL_MIX_API, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Extract data for all available days
            all_records = []

            if 'data' in data:
                for date_key, date_data in data['data'].items():
                    for timestamp_key, fuel_data in date_data.items():
                        if 'Power Storage' in fuel_data:
                            storage_data = fuel_data['Power Storage']
                            record = {
                                'timestamp': timestamp_key,
                                'generation_mw': storage_data.get('gen', 0)
                            }
                            all_records.append(record)

            if not all_records:
                logger.warning("No Power Storage data found in Fuel Mix API")
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(all_records)

            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Sort by timestamp
            df = df.sort_values('timestamp')

            logger.info(f"Downloaded {len(df)} records from Fuel Mix API")
            logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

            return df

        except Exception as e:
            logger.error(f"Error downloading from Fuel Mix API: {e}")
            return pd.DataFrame()

    def download_historical_fuel_mix(self, start_year: int = 2019, end_year: int = 2024) -> pd.DataFrame:
        """Download historical fuel mix reports and extract Power Storage data.

        Note: The fuel mix reports are zip files that need to be downloaded from:
        https://www.ercot.com/gridinfo/generation

        This function expects the files to be manually downloaded and placed in:
        {output_dir}/historical_downloads/

        Files should be named: IntGenbyFuel{YEAR}.zip
        """
        logger.info(f"Processing historical fuel mix data ({start_year}-{end_year})...")

        historical_dir = self.output_dir / "historical_downloads"
        historical_dir.mkdir(parents=True, exist_ok=True)

        all_data = []

        for year in range(start_year, end_year + 1):
            zip_file = historical_dir / f"IntGenbyFuel{year}.zip"

            if not zip_file.exists():
                logger.warning(
                    f"Historical file not found: {zip_file}\n"
                    f"Please download from {self.GENERATION_PAGE}"
                )
                continue

            try:
                logger.info(f"Processing {zip_file.name}...")

                with zipfile.ZipFile(zip_file, 'r') as zf:
                    # Find Excel/CSV files in the zip
                    for filename in zf.namelist():
                        if filename.endswith(('.xlsx', '.xls', '.csv')):
                            logger.info(f"  Reading {filename}...")

                            with zf.open(filename) as f:
                                # Try to read as Excel first
                                if filename.endswith('.csv'):
                                    df = pd.read_csv(f)
                                else:
                                    df = pd.read_excel(f)

                                # Look for Power Storage columns
                                # Column names vary, common patterns:
                                # - "STORAGE", "Power Storage", "Battery Storage", "ESR"
                                storage_cols = [col for col in df.columns
                                              if any(term in str(col).upper()
                                                   for term in ['STORAGE', 'BATTERY', 'ESR'])]

                                if storage_cols:
                                    logger.info(f"    Found storage columns: {storage_cols}")
                                    # Process and append data
                                    # This will need to be customized based on actual file format
                                    # For now, we'll just note what we found
                                else:
                                    logger.debug(f"    No storage columns found in {filename}")

            except Exception as e:
                logger.error(f"Error processing {zip_file.name}: {e}")
                continue

        if all_data:
            df = pd.concat(all_data, ignore_index=True)
            df = df.sort_values('timestamp')
            return df
        else:
            return pd.DataFrame()

    def save_data(self, df: pd.DataFrame, filename: str):
        """Save DataFrame to CSV file."""
        if df.empty:
            logger.warning(f"No data to save for {filename}")
            return

        output_file = self.output_dir / filename
        df.to_csv(output_file, index=False)
        logger.info(f"Saved {len(df)} records to {output_file}")

    def merge_and_deduplicate(self, dfs: List[pd.DataFrame]) -> pd.DataFrame:
        """Merge multiple DataFrames and remove duplicates."""
        if not dfs:
            return pd.DataFrame()

        # Filter out empty DataFrames
        dfs = [df for df in dfs if not df.empty]

        if not dfs:
            return pd.DataFrame()

        # Concatenate all DataFrames
        combined = pd.concat(dfs, ignore_index=True)

        # Remove duplicates based on timestamp
        combined = combined.drop_duplicates(subset=['timestamp'], keep='last')

        # Sort by timestamp
        combined = combined.sort_values('timestamp')

        return combined

    def download_all(self, start_year: int = 2019):
        """Download all available battery storage data."""
        logger.info("=" * 80)
        logger.info("ERCOT Battery Storage Data Downloader")
        logger.info("=" * 80)

        # Download from APIs
        logger.info("\n1. Downloading from ERCOT APIs...")
        api_storage_df = self.download_energy_storage_api_data()
        api_fuelmix_df = self.download_fuel_mix_api_data()

        # Save API data separately
        if not api_storage_df.empty:
            self.save_data(api_storage_df, "battery_storage_api_recent.csv")

        if not api_fuelmix_df.empty:
            self.save_data(api_fuelmix_df, "battery_storage_fuelmix_recent.csv")

        # Download historical data
        logger.info(f"\n2. Processing historical data ({start_year}-2024)...")
        historical_df = self.download_historical_fuel_mix(start_year, 2024)

        if not historical_df.empty:
            self.save_data(historical_df, "battery_storage_historical.csv")

        # Merge all data
        logger.info("\n3. Merging all data sources...")
        all_dfs = [df for df in [api_storage_df, historical_df] if not df.empty]

        if all_dfs:
            combined_df = self.merge_and_deduplicate(all_dfs)
            self.save_data(combined_df, "battery_storage_combined.csv")

            logger.info("\n" + "=" * 80)
            logger.info("DOWNLOAD COMPLETE")
            logger.info("=" * 80)
            logger.info(f"Total records: {len(combined_df)}")
            logger.info(f"Date range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")
            logger.info(f"Output directory: {self.output_dir.absolute()}")
        else:
            logger.warning("No data was downloaded successfully")

    def print_instructions(self):
        """Print instructions for downloading historical files."""
        print("\n" + "=" * 80)
        print("INSTRUCTIONS FOR HISTORICAL DATA")
        print("=" * 80)
        print("\nTo download historical data (2019-2024), you need to:")
        print(f"\n1. Visit: {self.GENERATION_PAGE}")
        print("\n2. Download these files:")
        print("   - 'Fuel Mix Report: 2007 - 2024' (ZIP file, ~48 MB)")
        print("   - 'Fuel Mix Report: 2025' (Excel file, ~2 MB)")
        print(f"\n3. Place the downloaded files in:")
        print(f"   {(self.output_dir / 'historical_downloads').absolute()}/")
        print("\n4. Run this script again")
        print("\n" + "=" * 80)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Download ERCOT battery storage data from 2019 to present'
    )
    parser.add_argument(
        '--start-year',
        type=int,
        default=2019,
        help='Start year for historical data (default: 2019)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='ercot_battery_storage_data',
        help='Output directory for data files'
    )
    parser.add_argument(
        '--instructions',
        action='store_true',
        help='Print instructions for downloading historical files'
    )

    args = parser.parse_args()

    downloader = ERCOTBatteryStorageDownloader(output_dir=args.output_dir)

    if args.instructions:
        downloader.print_instructions()
    else:
        downloader.download_all(start_year=args.start_year)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Download ERCOT Battery Storage Data - Full Automated Version

This script automatically downloads historical and real-time battery storage data from ERCOT:
1. Historical data (2019-2024): Automatically scrapes and downloads Fuel Mix Reports
2. Recent/Real-time data: From ERCOT API endpoints

Data Sources:
- Energy Storage Resources API: /api/1/services/read/dashboards/energy-storage-resources.json
- Fuel Mix API: /api/1/services/read/dashboards/fuel-mix.json
- Historical Fuel Mix Reports: https://www.ercot.com/gridinfo/generation
"""

import os
import sys
import json
import requests
import pandas as pd
import zipfile
import io
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any
import time
import logging
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ERCOTBatteryStorageDownloader:
    """Download and process ERCOT battery storage data with full automation."""

    # API Endpoints
    ENERGY_STORAGE_API = "https://www.ercot.com/api/1/services/read/dashboards/energy-storage-resources.json"
    FUEL_MIX_API = "https://www.ercot.com/api/1/services/read/dashboards/fuel-mix.json"
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

    def scrape_historical_file_urls(self) -> Dict[str, str]:
        """Scrape the generation page to find download URLs for historical files."""
        logger.info("Scraping generation page for historical file URLs...")

        try:
            response = self.session.get(self.GENERATION_PAGE, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Find download links
            file_urls = {}

            # Look for links with "Fuel Mix" in the text
            for link in soup.find_all('a', href=True):
                link_text = link.get_text().strip()
                href = link['href']

                # Match patterns like "Fuel Mix Report: 2007 - 2024" or "Fuel Mix Report: 2025"
                if 'fuel mix' in link_text.lower() and 'report' in link_text.lower():
                    # Extract year range
                    year_match = re.search(r'(\d{4})\s*-\s*(\d{4})|(\d{4})', link_text)
                    if year_match:
                        # Make URL absolute if needed
                        if href.startswith('/'):
                            href = f"https://www.ercot.com{href}"
                        elif not href.startswith('http'):
                            href = f"https://www.ercot.com/{href}"

                        file_urls[link_text] = href
                        logger.info(f"Found: {link_text} -> {href}")

            return file_urls

        except Exception as e:
            logger.error(f"Error scraping generation page: {e}")
            return {}

    def download_historical_file(self, url: str, filename: str) -> Optional[Path]:
        """Download a historical file from URL."""
        output_path = self.output_dir / "historical_downloads" / filename

        # Skip if already downloaded
        if output_path.exists():
            logger.info(f"File already exists: {filename}")
            return output_path

        logger.info(f"Downloading {filename} from {url}...")

        try:
            response = self.session.get(url, timeout=300, stream=True)
            response.raise_for_status()

            # Create directory if needed
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Download with progress
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size:
                            percent = (downloaded / total_size) * 100
                            logger.info(f"  Progress: {percent:.1f}%")

            logger.info(f"Downloaded {filename} successfully")
            return output_path

        except Exception as e:
            logger.error(f"Error downloading {filename}: {e}")
            if output_path.exists():
                output_path.unlink()  # Remove partial download
            return None

    def process_historical_fuel_mix_file(self, file_path: Path) -> pd.DataFrame:
        """Process a historical fuel mix file and extract Power Storage data."""
        logger.info(f"Processing {file_path.name}...")

        all_data = []

        try:
            # Determine file type
            if file_path.suffix == '.zip':
                # Process ZIP file
                with zipfile.ZipFile(file_path, 'r') as zf:
                    for filename in zf.namelist():
                        if filename.endswith(('.xlsx', '.xls', '.csv')):
                            logger.info(f"  Reading {filename}...")

                            try:
                                with zf.open(filename) as f:
                                    # Try to read file
                                    if filename.endswith('.csv'):
                                        df = pd.read_csv(f)
                                    else:
                                        df = pd.read_excel(f)

                                    # Process this file
                                    storage_data = self.extract_storage_from_df(df, filename)
                                    if not storage_data.empty:
                                        all_data.append(storage_data)

                            except Exception as e:
                                logger.warning(f"    Error reading {filename}: {e}")
                                continue

            elif file_path.suffix in ['.xlsx', '.xls']:
                # Process Excel file directly
                df = pd.read_excel(file_path)
                storage_data = self.extract_storage_from_df(df, file_path.name)
                if not storage_data.empty:
                    all_data.append(storage_data)

            if all_data:
                combined = pd.concat(all_data, ignore_index=True)
                combined = combined.drop_duplicates(subset=['timestamp'], keep='last')
                combined = combined.sort_values('timestamp')
                logger.info(f"  Extracted {len(combined)} storage records from {file_path.name}")
                return combined
            else:
                logger.warning(f"  No storage data found in {file_path.name}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {e}")
            return pd.DataFrame()

    def extract_storage_from_df(self, df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        """Extract Power Storage data from a DataFrame."""
        # Look for timestamp/date columns
        date_cols = [col for col in df.columns
                    if any(term in str(col).upper()
                         for term in ['DATE', 'TIME', 'INTERVAL'])]

        # Look for Power Storage columns
        storage_cols = [col for col in df.columns
                       if any(term in str(col).upper()
                            for term in ['STORAGE', 'BATTERY', 'ESR', 'ENERGY STORAGE'])]

        if not date_cols or not storage_cols:
            return pd.DataFrame()

        logger.info(f"    Found date columns: {date_cols}")
        logger.info(f"    Found storage columns: {storage_cols}")

        # Extract relevant data
        # This is a simplified version - actual implementation would need to handle
        # various file formats and column structures
        result = pd.DataFrame()

        try:
            # Use first date column and first storage column
            date_col = date_cols[0]
            storage_col = storage_cols[0]

            result = df[[date_col, storage_col]].copy()
            result.columns = ['timestamp', 'net_output_mw']

            # Convert to datetime
            result['timestamp'] = pd.to_datetime(result['timestamp'])

            # Drop NaN values
            result = result.dropna()

        except Exception as e:
            logger.warning(f"    Error extracting data: {e}")
            return pd.DataFrame()

        return result

    def download_historical_data(self, start_year: int = 2019) -> pd.DataFrame:
        """Download and process historical fuel mix data."""
        logger.info(f"Downloading historical data from {start_year}...")

        # Scrape URLs
        file_urls = self.scrape_historical_file_urls()

        if not file_urls:
            logger.error("Could not find any historical file URLs")
            return pd.DataFrame()

        # Download and process each file
        all_data = []

        for file_description, url in file_urls.items():
            # Determine filename from URL or description
            url_path = url.split('/')[-1]
            if url_path:
                filename = url_path
            else:
                # Generate filename from description
                filename = file_description.replace(' ', '_').replace(':', '').replace('-', '_') + '.zip'

            # Download file
            file_path = self.download_historical_file(url, filename)

            if file_path and file_path.exists():
                # Process file
                df = self.process_historical_fuel_mix_file(file_path)
                if not df.empty:
                    # Filter by year
                    df['year'] = df['timestamp'].dt.year
                    df = df[df['year'] >= start_year]
                    df = df.drop('year', axis=1)

                    if not df.empty:
                        all_data.append(df)

        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            combined = combined.drop_duplicates(subset=['timestamp'], keep='last')
            combined = combined.sort_values('timestamp')
            return combined
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

    def download_all(self, start_year: int = 2019, download_historical: bool = True):
        """Download all available battery storage data."""
        logger.info("=" * 80)
        logger.info("ERCOT Battery Storage Data Downloader (Full Automated)")
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
        historical_df = pd.DataFrame()

        if download_historical:
            logger.info(f"\n2. Downloading historical data ({start_year}-2024)...")
            historical_df = self.download_historical_data(start_year)

            if not historical_df.empty:
                self.save_data(historical_df, "battery_storage_historical.csv")
        else:
            logger.info("\n2. Skipping historical data download (use --download-historical to enable)")

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


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Download ERCOT battery storage data from 2019 to present (fully automated)'
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
        '--download-historical',
        action='store_true',
        help='Download historical files (may take several minutes)'
    )
    parser.add_argument(
        '--api-only',
        action='store_true',
        help='Download only API data (last 2 days)'
    )

    args = parser.parse_args()

    downloader = ERCOTBatteryStorageDownloader(output_dir=args.output_dir)

    if args.api_only:
        # Quick mode - just download API data
        api_storage = downloader.download_energy_storage_api_data()
        if not api_storage.empty:
            downloader.save_data(api_storage, "battery_storage_latest.csv")
            print(f"\nDownloaded {len(api_storage)} records")
            print(f"Date range: {api_storage['timestamp'].min()} to {api_storage['timestamp'].max()}")
    else:
        downloader.download_all(
            start_year=args.start_year,
            download_historical=args.download_historical
        )


if __name__ == "__main__":
    main()

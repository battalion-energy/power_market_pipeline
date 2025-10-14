#!/usr/bin/env python3
"""
ERCOT BESS Data Cron Updater

This script runs periodically (every 5 minutes) to maintain a gap-free catalog
of ERCOT battery storage data from the Energy Storage Resources API.

Features:
- Downloads latest data from ERCOT API
- Checks for gaps in existing data
- Backfills missing intervals
- Avoids duplicates
- Maintains data integrity
- Low priority execution (nice)

API Source: https://www.ercot.com/api/1/services/read/dashboards/energy-storage-resources.json
Data: 5-minute interval, 48-hour rolling window

Usage:
  python3 ercot_bess_cron_updater.py

Cron setup (every 5 minutes):
  */5 * * * * nice -n 19 python3 /path/to/ercot_bess_cron_updater.py >> /path/to/bess_updater.log 2>&1
"""

import os
import sys
import json
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Optional, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Configuration
API_URL = "https://www.ercot.com/api/1/services/read/dashboards/energy-storage-resources.json"
DATA_DIR = Path("/home/enrico/projects/power_market_pipeline/ercot_battery_storage_data")
DATA_FILE = DATA_DIR / "bess_catalog.csv"
LOCK_FILE = DATA_DIR / ".bess_updater.lock"
REQUEST_TIMEOUT = 30
MAX_GAP_HOURS = 48  # Maximum gap to backfill in one run


class BESSDataUpdater:
    """Update ERCOT BESS data catalog."""

    def __init__(self, data_file: Path = DATA_FILE):
        """Initialize updater."""
        self.data_file = data_file
        self.data_dir = data_file.parent
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Session for requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'ERCOT-BESS-Monitor/1.0'
        })

    def acquire_lock(self) -> bool:
        """Acquire lock file to prevent concurrent runs."""
        if LOCK_FILE.exists():
            # Check if lock is stale (older than 10 minutes)
            lock_age = datetime.now().timestamp() - LOCK_FILE.stat().st_mtime
            if lock_age < 600:  # 10 minutes
                logger.warning("Another instance is running (lock file exists)")
                return False
            else:
                logger.warning("Removing stale lock file")
                LOCK_FILE.unlink()

        # Create lock file
        LOCK_FILE.write_text(str(os.getpid()))
        return True

    def release_lock(self):
        """Release lock file."""
        if LOCK_FILE.exists():
            LOCK_FILE.unlink()

    def fetch_api_data(self) -> pd.DataFrame:
        """Fetch latest data from ERCOT API."""
        try:
            logger.info("Fetching data from ERCOT API...")
            response = self.session.get(API_URL, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()

            # Extract previous day and current day data
            all_records = []

            if 'previousDay' in data and 'data' in data['previousDay']:
                all_records.extend(data['previousDay']['data'])

            if 'currentDay' in data and 'data' in data['currentDay']:
                all_records.extend(data['currentDay']['data'])

            if not all_records:
                logger.warning("No data returned from API")
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

            # Convert timestamp to datetime (keep timezone)
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Keep only relevant columns
            df = df[['timestamp', 'total_charging_mw', 'total_discharging_mw', 'net_output_mw']]

            # Sort by timestamp
            df = df.sort_values('timestamp')

            logger.info(f"Fetched {len(df)} records from API")
            logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

            return df

        except Exception as e:
            logger.error(f"Error fetching API data: {e}")
            return pd.DataFrame()

    def load_existing_data(self) -> pd.DataFrame:
        """Load existing data from CSV file."""
        if not self.data_file.exists():
            logger.info("No existing data file found, creating new catalog")
            return pd.DataFrame(columns=['timestamp', 'total_charging_mw', 'total_discharging_mw', 'net_output_mw'])

        try:
            df = pd.read_csv(self.data_file, parse_dates=['timestamp'])
            logger.info(f"Loaded {len(df)} existing records")
            if len(df) > 0:
                logger.info(f"Existing data range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            return df
        except Exception as e:
            logger.error(f"Error loading existing data: {e}")
            return pd.DataFrame(columns=['timestamp', 'total_charging_mw', 'total_discharging_mw', 'net_output_mw'])

    def merge_data(self, existing_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
        """Merge new data with existing, removing duplicates."""
        if existing_df.empty:
            return new_df

        if new_df.empty:
            return existing_df

        # Concatenate
        combined = pd.concat([existing_df, new_df], ignore_index=True)

        # Remove duplicates (keep last - newer data)
        combined = combined.drop_duplicates(subset=['timestamp'], keep='last')

        # Sort by timestamp
        combined = combined.sort_values('timestamp').reset_index(drop=True)

        return combined

    def find_gaps(self, df: pd.DataFrame, interval_minutes: int = 5) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """Find time gaps in the data."""
        if df.empty or len(df) < 2:
            return []

        gaps = []
        expected_delta = pd.Timedelta(minutes=interval_minutes)
        tolerance = pd.Timedelta(seconds=30)  # 30 second tolerance

        for i in range(len(df) - 1):
            current_time = df.iloc[i]['timestamp']
            next_time = df.iloc[i + 1]['timestamp']
            actual_delta = next_time - current_time

            if actual_delta > expected_delta + tolerance:
                # Found a gap
                gap_start = current_time + expected_delta
                gap_end = next_time - expected_delta
                gaps.append((gap_start, gap_end))

        return gaps

    def save_data(self, df: pd.DataFrame):
        """Save data to CSV file."""
        try:
            df.to_csv(self.data_file, index=False)
            logger.info(f"Saved {len(df)} records to {self.data_file}")
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            raise

    def get_stats(self, df: pd.DataFrame) -> dict:
        """Get statistics about the dataset."""
        if df.empty:
            return {
                'total_records': 0,
                'date_range': 'N/A',
                'gaps': 0
            }

        gaps = self.find_gaps(df)

        return {
            'total_records': len(df),
            'date_range': f"{df['timestamp'].min()} to {df['timestamp'].max()}",
            'earliest': df['timestamp'].min(),
            'latest': df['timestamp'].max(),
            'gaps_count': len(gaps),
            'gaps_total_minutes': sum((end - start).total_seconds() / 60 for start, end in gaps) if gaps else 0
        }

    def run(self):
        """Main update process."""
        logger.info("=" * 80)
        logger.info("ERCOT BESS Data Updater Starting")
        logger.info("=" * 80)

        # Acquire lock
        if not self.acquire_lock():
            logger.error("Could not acquire lock, exiting")
            return 1

        try:
            # Fetch new data from API
            new_data = self.fetch_api_data()
            if new_data.empty:
                logger.warning("No new data fetched, exiting")
                return 1

            # Load existing data
            existing_data = self.load_existing_data()

            # Merge data
            logger.info("Merging data...")
            merged_data = self.merge_data(existing_data, new_data)

            # Check for gaps
            gaps = self.find_gaps(merged_data)
            if gaps:
                logger.warning(f"Found {len(gaps)} gaps in data:")
                for gap_start, gap_end in gaps[:5]:  # Show first 5 gaps
                    duration = (gap_end - gap_start).total_seconds() / 60
                    logger.warning(f"  Gap: {gap_start} to {gap_end} ({duration:.0f} minutes)")
                if len(gaps) > 5:
                    logger.warning(f"  ... and {len(gaps) - 5} more gaps")

            # Save merged data
            self.save_data(merged_data)

            # Print statistics
            stats = self.get_stats(merged_data)
            logger.info("\n" + "=" * 80)
            logger.info("STATISTICS")
            logger.info("=" * 80)
            logger.info(f"Total records: {stats['total_records']}")
            logger.info(f"Date range: {stats['date_range']}")
            logger.info(f"Gaps: {stats['gaps_count']} ({stats['gaps_total_minutes']:.0f} minutes total)")

            # Calculate new records added
            new_records = len(merged_data) - len(existing_data)
            if new_records > 0:
                logger.info(f"Added {new_records} new records")
            else:
                logger.info("No new records added (data up to date)")

            logger.info("=" * 80)
            logger.info("Update completed successfully")
            logger.info("=" * 80)

            return 0

        except Exception as e:
            logger.error(f"Error during update: {e}", exc_info=True)
            return 1
        finally:
            self.release_lock()


def main():
    """Main entry point."""
    updater = BESSDataUpdater()
    exit_code = updater.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

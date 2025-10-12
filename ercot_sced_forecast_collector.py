#!/usr/bin/env python3
"""
ERCOT SCED LMP Forecast Collector with Vintage Preservation

Downloads RTD Indicative LMPs (SCED forecasts) and preserves forecast vintages.
Each RTD run produces forecasts for the next 12 intervals. We track:
  - rtd_timestamp: When the forecast was published
  - interval_ending: Which future interval is being forecasted
  - settlement_point: Location (hub, zone, or resource node)
  - lmp: Forecasted price

This allows analysis of forecast accuracy over time and forecast evolution.

Usage:
    # Download maximum available history
    python3 ercot_sced_forecast_collector.py --harvest

    # Continuous mode (for cron)
    python3 ercot_sced_forecast_collector.py --continuous

Environment Variables Required:
    ERCOT_USERNAME
    ERCOT_PASSWORD
    ERCOT_SUBSCRIPTION_KEY
"""

import asyncio
import sys
import argparse
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from ercot_ws_downloader.client import ERCOTWebServiceClient
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
DATA_DIR = Path(__file__).parent / "ercot_battery_storage_data" / "sced_forecasts"
CATALOG_FILE = DATA_DIR / "forecast_catalog.csv"
ENDPOINT = "np6-970-cd/rtd_lmp_node_zone_hub"


class SCEDForecastCollector:
    """Collect ERCOT SCED LMP forecasts with vintage preservation."""

    def __init__(self, output_dir: Path = DATA_DIR):
        """Initialize collector."""
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.client = ERCOTWebServiceClient()

    async def fetch_forecasts(
        self,
        hours_back: int = 48,
        settlement_point_type: str = None
    ) -> pd.DataFrame:
        """
        Fetch SCED forecast data from ERCOT API.

        Args:
            hours_back: Hours of historical data to fetch
            settlement_point_type: Filter by type (HU, LZ, RN) or None for all

        Returns:
            DataFrame with forecast vintages
        """
        logger.info(f"Fetching SCED forecasts (last {hours_back} hours)...")

        # Calculate time range
        start_time = (datetime.now() - timedelta(hours=hours_back)).strftime("%Y-%m-%dT%H:%M:%S")

        # Build params
        params = {
            "RTDTimestampFrom": start_time,
            "size": 50000,  # Max per page
            "sort": "RTDTimestamp",
            "dir": "desc"
        }

        if settlement_point_type:
            params["settlementPointType"] = settlement_point_type

        try:
            # Get first page to extract column names
            page1_params = {**params, "page": 1, "size": 1}
            first_response = await self.client._make_request(ENDPOINT, page1_params)

            # Extract column names from fields metadata
            column_names = None
            if isinstance(first_response, dict) and "fields" in first_response:
                column_names = [field["name"] for field in first_response["fields"]]
                logger.info(f"Extracted {len(column_names)} column names from API schema")

            # Use existing client's pagination
            data = await self.client.get_paginated_data(
                endpoint=ENDPOINT,
                params=params,
                page_size=50000,
                max_pages=None  # Get all available data
            )

            if not data:
                logger.warning("No data returned from API")
                return pd.DataFrame()

            # Convert to DataFrame
            if column_names and isinstance(data[0], (list, tuple)):
                # Data is list of lists - use extracted column names
                df = pd.DataFrame(data, columns=column_names)
                logger.info(f"Created DataFrame with {len(column_names)} columns from schema")
            else:
                # Data is list of dicts - let pandas infer
                df = pd.DataFrame(data)

            # Standardize column names (API may return different case)
            column_mapping = {
                'RTDTimestamp': 'rtd_timestamp',
                'rtdTimestamp': 'rtd_timestamp',
                'intervalEnding': 'interval_ending',
                'IntervalEnding': 'interval_ending',
                'intervalId': 'interval_id',
                'IntervalId': 'interval_id',
                'settlementPoint': 'settlement_point',
                'SettlementPoint': 'settlement_point',
                'settlementPointType': 'settlement_point_type',
                'SettlementPointType': 'settlement_point_type',
                'LMP': 'lmp',
                'lmp': 'lmp',
                'repeatHourFlag': 'repeat_hour_flag',
                'RepeatHourFlag': 'repeat_hour_flag'
            }

            df = df.rename(columns=column_mapping)

            # Ensure we have required columns
            required = ['rtd_timestamp', 'interval_ending', 'settlement_point', 'lmp']
            missing = [col for col in required if col not in df.columns]
            if missing:
                logger.error(f"Missing required columns: {missing}")
                logger.error(f"Available columns: {df.columns.tolist()}")
                return pd.DataFrame()

            # Convert timestamps
            df['rtd_timestamp'] = pd.to_datetime(df['rtd_timestamp'])
            df['interval_ending'] = pd.to_datetime(df['interval_ending'])
            df['fetch_time'] = datetime.now()

            # Keep only relevant columns
            keep_cols = [
                'rtd_timestamp', 'interval_ending', 'interval_id',
                'settlement_point', 'settlement_point_type',
                'lmp', 'repeat_hour_flag', 'fetch_time'
            ]
            df = df[[col for col in keep_cols if col in df.columns]]

            logger.info(f"Fetched {len(df):,} forecast records")
            logger.info(f"RTD runs: {df['rtd_timestamp'].nunique()} unique")
            logger.info(f"Forecast intervals: {df['interval_ending'].nunique()} unique")
            logger.info(f"Settlement points: {df['settlement_point'].nunique()} unique")
            logger.info(f"Time range: {df['rtd_timestamp'].min()} to {df['rtd_timestamp'].max()}")

            return df

        except Exception as e:
            logger.error(f"Error fetching forecasts: {e}", exc_info=True)
            return pd.DataFrame()

    def load_existing_catalog(self) -> pd.DataFrame:
        """Load existing forecast catalog."""
        if not CATALOG_FILE.exists():
            logger.info("No existing catalog found, creating new one")
            return pd.DataFrame()

        try:
            df = pd.read_csv(
                CATALOG_FILE,
                parse_dates=['rtd_timestamp', 'interval_ending', 'fetch_time']
            )
            logger.info(f"Loaded {len(df):,} existing forecast records")
            if len(df) > 0:
                logger.info(f"Existing range: {df['rtd_timestamp'].min()} to {df['rtd_timestamp'].max()}")
            return df
        except Exception as e:
            logger.error(f"Error loading existing catalog: {e}")
            return pd.DataFrame()

    def merge_catalogs(self, existing_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge new forecasts with existing catalog, preserving vintages.

        Deduplicates based on (rtd_timestamp, interval_ending, settlement_point) tuple.
        This ensures we keep each unique forecast vintage.
        """
        if existing_df.empty:
            return new_df

        if new_df.empty:
            return existing_df

        # Concatenate
        combined = pd.concat([existing_df, new_df], ignore_index=True)

        # Remove duplicates - keep the version with latest fetch_time
        # This preserves forecast vintages (unique rtd_timestamp + interval_ending + settlement_point)
        combined = combined.sort_values('fetch_time')
        combined = combined.drop_duplicates(
            subset=['rtd_timestamp', 'interval_ending', 'settlement_point'],
            keep='last'
        )

        # Sort by RTD timestamp (most recent first), then interval
        combined = combined.sort_values(
            ['rtd_timestamp', 'interval_ending', 'settlement_point'],
            ascending=[False, True, True]
        ).reset_index(drop=True)

        return combined

    def save_catalog(self, df: pd.DataFrame):
        """Save forecast catalog to CSV."""
        if df.empty:
            logger.warning("No data to save")
            return

        df.to_csv(CATALOG_FILE, index=False)
        logger.info(f"Saved {len(df):,} forecast records to {CATALOG_FILE}")

    def get_statistics(self, df: pd.DataFrame):
        """Print statistics about forecast catalog."""
        if df.empty:
            logger.info("No data for statistics")
            return

        logger.info("=" * 80)
        logger.info("FORECAST CATALOG STATISTICS")
        logger.info("=" * 80)
        logger.info(f"Total forecast records: {len(df):,}")
        logger.info(f"Unique RTD runs: {df['rtd_timestamp'].nunique():,}")
        logger.info(f"Unique forecast intervals: {df['interval_ending'].nunique():,}")
        logger.info(f"Settlement points: {df['settlement_point'].nunique():,}")
        logger.info(f"RTD timestamp range: {df['rtd_timestamp'].min()} to {df['rtd_timestamp'].max()}")
        logger.info(f"Forecast horizon: {df['interval_ending'].min()} to {df['interval_ending'].max()}")

        # Settlement point type breakdown
        if 'settlement_point_type' in df.columns:
            logger.info("\nSettlement Point Types:")
            type_counts = df['settlement_point_type'].value_counts()
            for sp_type, count in type_counts.items():
                pct = count / len(df) * 100
                logger.info(f"  {sp_type}: {count:,} records ({pct:.1f}%)")

        # Price statistics
        logger.info(f"\nPrice Statistics ($/MWh):")
        logger.info(f"  Mean: ${df['lmp'].mean():.2f}")
        logger.info(f"  Median: ${df['lmp'].median():.2f}")
        logger.info(f"  Min: ${df['lmp'].min():.2f}")
        logger.info(f"  Max: ${df['lmp'].max():.2f}")
        logger.info(f"  Std Dev: ${df['lmp'].std():.2f}")

        # Top settlement points by average price
        logger.info(f"\nTop 10 Highest Average LMP:")
        top_points = df.groupby('settlement_point')['lmp'].agg(['mean', 'count']).sort_values('mean', ascending=False).head(10)
        for point, row in top_points.iterrows():
            logger.info(f"  {point}: ${row['mean']:.2f} avg (n={row['count']:,})")

        # Forecast vintage analysis
        if len(df) > 0:
            # How many forecast runs do we have?
            rtd_runs = df['rtd_timestamp'].nunique()
            avg_forecasts_per_run = len(df) / rtd_runs if rtd_runs > 0 else 0
            logger.info(f"\nForecast Vintage Analysis:")
            logger.info(f"  RTD runs captured: {rtd_runs:,}")
            logger.info(f"  Avg forecasts per run: {avg_forecasts_per_run:.1f}")

            # Show latest RTD run
            latest_rtd = df['rtd_timestamp'].max()
            latest_forecasts = df[df['rtd_timestamp'] == latest_rtd]
            logger.info(f"\nLatest RTD Run ({latest_rtd}):")
            logger.info(f"  Forecasts generated: {len(latest_forecasts)}")
            logger.info(f"  Settlement points: {latest_forecasts['settlement_point'].nunique()}")
            logger.info(f"  Forecast intervals: {latest_forecasts['interval_ending'].min()} to {latest_forecasts['interval_ending'].max()}")

        logger.info("=" * 80)

    async def run_harvest(self, hours_back: int = 48):
        """Harvest maximum available historical forecasts."""
        logger.info("=" * 80)
        logger.info("ERCOT SCED FORECAST HARVEST")
        logger.info("=" * 80)

        # Fetch new data
        new_df = await self.fetch_forecasts(hours_back=hours_back)

        if new_df.empty:
            logger.error("No data fetched")
            return 1

        # Load existing catalog
        existing_df = self.load_existing_catalog()

        # Merge with existing
        logger.info("Merging with existing catalog...")
        merged_df = self.merge_catalogs(existing_df, new_df)

        # Calculate what's new
        new_records = len(merged_df) - len(existing_df)
        logger.info(f"Added {new_records:,} new forecast records")

        # Save
        self.save_catalog(merged_df)

        # Show statistics
        self.get_statistics(merged_df)

        return 0

    async def run_continuous(self, hours_back: int = 3):
        """Continuous mode for cron - fetch recent data only."""
        logger.info("ERCOT SCED Forecast Collector - Continuous Mode")

        # Fetch recent data (smaller window for cron)
        new_df = await self.fetch_forecasts(hours_back=hours_back)

        if new_df.empty:
            logger.warning("No new data fetched")
            return 1

        # Load existing
        existing_df = self.load_existing_catalog()

        # Merge
        merged_df = self.merge_catalogs(existing_df, new_df)

        # Calculate additions
        new_records = len(merged_df) - len(existing_df)

        # Save
        self.save_catalog(merged_df)

        if new_records > 0:
            logger.info(f"✓ Added {new_records:,} new forecast records")
            logger.info(f"  Total catalog size: {len(merged_df):,} records")
        else:
            logger.info("✓ No new forecast records (data up to date)")

        return 0


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ERCOT SCED LMP Forecast Collector with Vintage Preservation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Harvest maximum available history (first run)
  python3 ercot_sced_forecast_collector.py --harvest

  # Harvest with longer lookback
  python3 ercot_sced_forecast_collector.py --harvest --hours-back 72

  # Continuous mode (for cron)
  python3 ercot_sced_forecast_collector.py --continuous

  # Show statistics only
  python3 ercot_sced_forecast_collector.py --stats
        """
    )

    parser.add_argument(
        '--harvest',
        action='store_true',
        help='Harvest maximum available historical forecasts'
    )
    parser.add_argument(
        '--continuous',
        action='store_true',
        help='Continuous mode for cron (fetch recent data)'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show statistics for existing catalog'
    )
    parser.add_argument(
        '--hours-back',
        type=int,
        default=48,
        help='Hours of historical data to fetch (default: 48)'
    )

    args = parser.parse_args()

    # Must specify mode
    if not any([args.harvest, args.continuous, args.stats]):
        parser.error("Must specify --harvest, --continuous, or --stats")

    collector = SCEDForecastCollector()

    if args.stats:
        # Just show statistics
        existing_df = collector.load_existing_catalog()
        collector.get_statistics(existing_df)
        return 0

    if args.harvest:
        return await collector.run_harvest(hours_back=args.hours_back)

    if args.continuous:
        return await collector.run_continuous(hours_back=3)


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

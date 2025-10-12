#!/usr/bin/env python3
"""
Download ERCOT SCED LMP Forecasts (RTD Indicative LMPs)

This script downloads the latest SCED forecast data from ERCOT's public API.
Dataset: NP6-970-CD - RTD Indicative LMPs by Resource Nodes, Load Zones and Hubs

Usage:
    python3 download_ercot_sced_forecasts.py

    # With custom settlement point filter
    python3 download_ercot_sced_forecasts.py --settlement-point HB_NORTH

    # Continuous collection mode (for cron)
    python3 download_ercot_sced_forecasts.py --continuous

Environment Variables Required:
    ERCOT_SUBSCRIPTION_KEY: Your ERCOT API subscription key
    ERCOT_USERNAME: Your ERCOT API username
    ERCOT_PASSWORD: Your ERCOT API password

Get credentials at: https://apiexplorer.ercot.com/
"""

import os
import sys
import json
import base64
import argparse
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Add ercot_webservices to path
sys.path.insert(0, str(Path(__file__).parent / "ercot_webservices"))

from ercot_webservices.ercot_api.client import Client
from ercot_webservices.ercot_api.api.np6_970_cd import get_data_rtd_lmp_node_zone_hub

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
DATA_DIR = Path(__file__).parent / "ercot_battery_storage_data" / "sced_forecasts"
API_BASE_URL = "https://api.ercot.com/api/public-reports"


class SCEDForecastDownloader:
    """Download ERCOT SCED LMP forecast data."""

    def __init__(
        self,
        subscription_key: str,
        username: str,
        password: str,
        output_dir: Path = DATA_DIR
    ):
        """Initialize downloader with credentials."""
        self.subscription_key = subscription_key
        self.username = username
        self.password = password
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create authenticated client
        auth_string = base64.b64encode(f"{username}:{password}".encode()).decode()
        self.client = Client(
            base_url=API_BASE_URL,
            headers={
                "Authorization": f"Basic {auth_string}",
                "Accept": "application/json"
            },
            timeout=30.0
        )

    def fetch_forecasts(
        self,
        settlement_point: str = None,
        settlement_point_type: str = None,
        hours_back: int = 2,
        max_records: int = 10000
    ) -> pd.DataFrame:
        """
        Fetch SCED forecast data from ERCOT API.

        Args:
            settlement_point: Filter by specific settlement point (e.g., "HB_NORTH")
            settlement_point_type: Filter by type ("HU"=Hub, "LZ"=Load Zone, "RN"=Resource Node)
            hours_back: How many hours back to fetch data
            max_records: Maximum records per request (max 10,000)

        Returns:
            DataFrame with forecast data
        """
        logger.info("Fetching SCED forecasts from ERCOT API...")

        # Calculate time range
        now = datetime.now()
        start_time = (now - timedelta(hours=hours_back)).strftime("%Y-%m-%dT%H:%M:%S")

        try:
            with self.client as client:
                response = get_data_rtd_lmp_node_zone_hub.sync_detailed(
                    client=client,
                    ocp_apim_subscription_key=self.subscription_key,
                    rtd_timestamp_from=start_time,
                    settlement_point=settlement_point,
                    settlement_point_type=settlement_point_type,
                    size=max_records,
                    sort="RTDTimestamp",
                    dir_="desc"
                )

                if response.status_code != 200:
                    logger.error(f"API request failed with status {response.status_code}")
                    logger.error(f"Response: {response.content}")
                    return pd.DataFrame()

                if not response.parsed or not hasattr(response.parsed, 'data'):
                    logger.error("No data in API response")
                    return pd.DataFrame()

                # Convert to DataFrame
                records = []
                for item in response.parsed.data:
                    # Convert attrs object to dict
                    record = {
                        'rtd_timestamp': getattr(item, 'rtd_timestamp', None),
                        'interval_ending': getattr(item, 'interval_ending', None),
                        'interval_id': getattr(item, 'interval_id', None),
                        'settlement_point': getattr(item, 'settlement_point', None),
                        'settlement_point_type': getattr(item, 'settlement_point_type', None),
                        'lmp': getattr(item, 'lmp', None),
                        'repeat_hour_flag': getattr(item, 'repeat_hour_flag', None),
                        'fetch_time': datetime.now().isoformat()
                    }
                    records.append(record)

                df = pd.DataFrame(records)

                if len(df) > 0:
                    # Convert timestamps
                    df['rtd_timestamp'] = pd.to_datetime(df['rtd_timestamp'])
                    df['interval_ending'] = pd.to_datetime(df['interval_ending'])

                    logger.info(f"Fetched {len(df)} forecast records")
                    logger.info(f"RTD timestamps: {df['rtd_timestamp'].min()} to {df['rtd_timestamp'].max()}")
                    logger.info(f"Forecast intervals: {df['interval_ending'].min()} to {df['interval_ending'].max()}")
                    logger.info(f"Settlement points: {df['settlement_point'].nunique()}")

                return df

        except Exception as e:
            logger.error(f"Error fetching forecasts: {e}", exc_info=True)
            return pd.DataFrame()

    def save_forecasts(self, df: pd.DataFrame, filename: str = None):
        """
        Save forecast data to CSV.

        Args:
            df: DataFrame with forecast data
            filename: Output filename (defaults to timestamped name)
        """
        if df.empty:
            logger.warning("No data to save")
            return

        if filename is None:
            filename = f"forecasts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        filepath = self.output_dir / filename
        df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(df)} records to {filepath}")

    def get_statistics(self, df: pd.DataFrame):
        """Print statistics about the forecast data."""
        if df.empty:
            logger.info("No data for statistics")
            return

        logger.info("=" * 80)
        logger.info("FORECAST DATA STATISTICS")
        logger.info("=" * 80)
        logger.info(f"Total records: {len(df)}")
        logger.info(f"RTD runs: {df['rtd_timestamp'].nunique()}")
        logger.info(f"Forecast intervals: {df['interval_ending'].nunique()}")
        logger.info(f"Settlement points: {df['settlement_point'].nunique()}")

        # Price statistics
        logger.info(f"\nPrice Statistics ($/MWh):")
        logger.info(f"  Mean: ${df['lmp'].mean():.2f}")
        logger.info(f"  Median: ${df['lmp'].median():.2f}")
        logger.info(f"  Min: ${df['lmp'].min():.2f}")
        logger.info(f"  Max: ${df['lmp'].max():.2f}")
        logger.info(f"  Std Dev: ${df['lmp'].std():.2f}")

        # Top settlement points by average price
        logger.info(f"\nTop 10 Settlement Points by Average LMP:")
        top_points = df.groupby('settlement_point')['lmp'].agg(['mean', 'count']).sort_values('mean', ascending=False).head(10)
        for point, row in top_points.iterrows():
            logger.info(f"  {point}: ${row['mean']:.2f} (n={row['count']})")

        logger.info("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download ERCOT SCED LMP Forecasts (RTD Indicative LMPs)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all forecasts
  python3 download_ercot_sced_forecasts.py

  # Download forecasts for specific hub
  python3 download_ercot_sced_forecasts.py --settlement-point HB_NORTH

  # Download only hub forecasts
  python3 download_ercot_sced_forecasts.py --settlement-point-type HU

  # Download with longer history
  python3 download_ercot_sced_forecasts.py --hours-back 6

  # Continuous mode (for cron jobs)
  python3 download_ercot_sced_forecasts.py --continuous
        """
    )

    parser.add_argument(
        '--settlement-point',
        help='Filter by settlement point (e.g., HB_NORTH, HB_SOUTH)'
    )
    parser.add_argument(
        '--settlement-point-type',
        choices=['HU', 'LZ', 'RN'],
        help='Filter by settlement point type (HU=Hub, LZ=Load Zone, RN=Resource Node)'
    )
    parser.add_argument(
        '--hours-back',
        type=int,
        default=2,
        help='Hours of historical data to fetch (default: 2)'
    )
    parser.add_argument(
        '--max-records',
        type=int,
        default=10000,
        help='Maximum records to fetch (default: 10000, max: 10000)'
    )
    parser.add_argument(
        '--output',
        help='Output filename (default: timestamped)'
    )
    parser.add_argument(
        '--continuous',
        action='store_true',
        help='Continuous mode: save to timestamped files (for cron)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress statistics output'
    )

    args = parser.parse_args()

    # Get credentials from environment
    subscription_key = os.getenv('ERCOT_SUBSCRIPTION_KEY')
    username = os.getenv('ERCOT_USERNAME')
    password = os.getenv('ERCOT_PASSWORD')

    if not all([subscription_key, username, password]):
        logger.error("Missing ERCOT API credentials!")
        logger.error("Please set environment variables:")
        logger.error("  ERCOT_SUBSCRIPTION_KEY")
        logger.error("  ERCOT_USERNAME")
        logger.error("  ERCOT_PASSWORD")
        logger.error("\nGet credentials at: https://apiexplorer.ercot.com/")
        sys.exit(1)

    # Create downloader
    downloader = SCEDForecastDownloader(
        subscription_key=subscription_key,
        username=username,
        password=password
    )

    # Fetch data
    df = downloader.fetch_forecasts(
        settlement_point=args.settlement_point,
        settlement_point_type=args.settlement_point_type,
        hours_back=args.hours_back,
        max_records=args.max_records
    )

    if df.empty:
        logger.error("No data fetched")
        sys.exit(1)

    # Save data
    if args.continuous:
        # Continuous mode: always use timestamped filename
        downloader.save_forecasts(df)
    else:
        downloader.save_forecasts(df, filename=args.output)

    # Show statistics
    if not args.quiet:
        downloader.get_statistics(df)

    logger.info("Done!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Daily weather forecast updater.
Downloads the latest forecasts from Open-Meteo and HRRR.

This script is designed to be run daily via cron to keep forecast data up-to-date.
It only downloads new forecasts since the last update (incremental).

Usage:
  python update_weather_forecasts.py
  python update_weather_forecasts.py --source openmeteo  # Only Open-Meteo
  python update_weather_forecasts.py --source hrrr       # Only HRRR
"""

import os
import sys
import pandas as pd
import requests
from datetime import datetime, timedelta
from pathlib import Path
import time
from typing import Optional, List
import json
import argparse
from dotenv import load_dotenv

# Open-Meteo Forecast API endpoint (for current forecasts)
OPENMETEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Forecast variables for Open-Meteo
OPENMETEO_VARIABLES = [
    'temperature_2m',
    'windspeed_10m',
    'windspeed_100m',
    'shortwave_radiation',
    'precipitation',
]


def get_latest_openmeteo_forecast(
    lat: float,
    lon: float,
    forecast_days: int = 7
) -> Optional[pd.DataFrame]:
    """
    Download current weather forecast from Open-Meteo.

    Args:
        lat: Latitude
        lon: Longitude
        forecast_days: Number of days ahead to forecast

    Returns:
        DataFrame with forecast data or None
    """
    params = {
        'latitude': lat,
        'longitude': lon,
        'hourly': ','.join(OPENMETEO_VARIABLES),
        'timezone': 'America/Chicago',
        'forecast_days': forecast_days
    }

    try:
        response = requests.get(OPENMETEO_FORECAST_URL, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()

        if 'hourly' not in data:
            return None

        # Convert to DataFrame
        hourly_data = data['hourly']
        df = pd.DataFrame(hourly_data)

        # Convert time to datetime
        df['time'] = pd.to_datetime(df['time'])

        return df

    except Exception as e:
        print(f"    ERROR: {e}")
        return None


def update_openmeteo_forecasts(weather_dir: Path, locations_df: pd.DataFrame):
    """Update Open-Meteo forecasts for all locations."""
    print("=" * 80)
    print("UPDATING OPEN-METEO FORECASTS")
    print("=" * 80)
    print(f"Locations: {len(locations_df)}")
    print(f"Forecast horizon: 7 days")
    print()

    forecast_dir = weather_dir / 'openmeteo_current_forecasts'
    forecast_dir.mkdir(exist_ok=True, parents=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    all_forecasts = []

    for idx, row in locations_df.iterrows():
        location_name = row['name']
        lat = row['lat']
        lon = row['lon']

        print(f"  [{idx+1}/{len(locations_df)}] {location_name}... ", end='', flush=True)

        df = get_latest_openmeteo_forecast(lat=lat, lon=lon, forecast_days=7)

        if df is not None and len(df) > 0:
            df.insert(0, 'location_name', location_name)
            df.insert(1, 'latitude', lat)
            df.insert(2, 'longitude', lon)
            df['forecast_timestamp'] = datetime.now()

            all_forecasts.append(df)
            print(f"✓ ({len(df)} hours)")
        else:
            print("✗")

        # Rate limiting
        time.sleep(1)

    if all_forecasts:
        # Combine all forecasts
        combined = pd.concat(all_forecasts, ignore_index=True)

        # Save with timestamp
        csv_file = forecast_dir / f'forecast_{timestamp}.csv'
        combined.to_csv(csv_file, index=False)

        parquet_file = forecast_dir / f'forecast_{timestamp}.parquet'
        combined.to_parquet(parquet_file, compression='snappy')

        # Also save as "latest"
        latest_csv = forecast_dir / 'latest_forecast.csv'
        combined.to_csv(latest_csv, index=False)

        latest_parquet = forecast_dir / 'latest_forecast.parquet'
        combined.to_parquet(latest_parquet, compression='snappy')

        print(f"\n✓ Saved {len(combined)} hourly forecast records")
        print(f"  CSV: {csv_file}")
        print(f"  Parquet: {parquet_file}")
        print(f"  Latest: {latest_parquet}")

        # Clean up old forecasts (keep last 7 days)
        cutoff_date = datetime.now() - timedelta(days=7)
        cutoff_str = cutoff_date.strftime('%Y%m%d')

        for old_file in forecast_dir.glob('forecast_*.csv'):
            # Extract date from filename
            parts = old_file.stem.split('_')
            if len(parts) >= 2 and parts[1] < cutoff_str:
                old_file.unlink()
                # Also remove corresponding parquet
                old_parquet = old_file.with_suffix('.parquet')
                if old_parquet.exists():
                    old_parquet.unlink()

        return True
    else:
        print("\n✗ No forecasts downloaded")
        return False


def update_hrrr_forecasts(weather_dir: Path, locations_df: pd.DataFrame):
    """Update HRRR forecasts (placeholder - requires Herbie)."""
    print("=" * 80)
    print("UPDATING HRRR FORECASTS")
    print("=" * 80)
    print("Note: HRRR download requires Herbie library")
    print("Skipping for now - implement when HRRR downloader is working")
    print()
    return False


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description='Update weather forecasts')
    parser.add_argument('--source', type=str, choices=['openmeteo', 'hrrr', 'both'],
                       default='both', help='Which forecast source to update')
    args = parser.parse_args()

    print("=" * 80)
    print("WEATHER FORECAST UPDATER")
    print("=" * 80)
    print(f"Update time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Source: {args.source}")
    print()

    # Load environment
    load_dotenv()
    weather_dir = Path(os.getenv('WEATHER_DATA_DIR', '/pool/ssd8tb/data/weather_data'))

    # Load locations
    locations_file = weather_dir / 'weather_locations.csv'
    if not locations_file.exists():
        print(f"ERROR: Locations file not found: {locations_file}")
        sys.exit(1)

    locations_df = pd.read_csv(locations_file)
    print(f"Loaded {len(locations_df)} locations")
    print()

    # Update forecasts
    success = False

    if args.source in ['openmeteo', 'both']:
        success = update_openmeteo_forecasts(weather_dir, locations_df) or success

    if args.source in ['hrrr', 'both']:
        success = update_hrrr_forecasts(weather_dir, locations_df) or success

    # Log update
    log_dir = weather_dir / 'logs'
    log_dir.mkdir(exist_ok=True)

    log_file = log_dir / 'forecast_updates.log'
    with open(log_file, 'a') as f:
        f.write(f"{datetime.now().isoformat()} - {args.source} - {'SUCCESS' if success else 'FAILED'}\n")

    print("\n" + "=" * 80)
    if success:
        print("UPDATE COMPLETE")
    else:
        print("UPDATE FAILED")
    print("=" * 80)

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

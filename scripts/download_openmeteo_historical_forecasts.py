#!/usr/bin/env python3
"""
Download historical weather FORECASTS from Open-Meteo Historical Forecast API.

This API provides actual archived weather model forecasts from 2022 onwards.
This is what was PREDICTED at the time, not what actually happened.

Open-Meteo Historical Forecast API:
- Available from 2022 onwards (some models back to 2016-2018)
- High-resolution models including HRRR
- Hourly and daily data
- FREE for non-commercial use

Rate Limits (Free Tier):
- 600 calls/minute
- 10,000 calls/day
- 300,000 calls/month

Documentation: https://open-meteo.com/en/docs/historical-forecast-api
"""

import os
import pandas as pd
import requests
from datetime import datetime, timedelta
from pathlib import Path
import time
from typing import Optional, List
import json
import argparse
from dotenv import load_dotenv

# Open-Meteo Historical Forecast API endpoint
HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"

# Variables to download (hourly)
# Note: Using exact parameter names from Historical Forecast API
HOURLY_VARIABLES = [
    'temperature_2m',
    'relative_humidity_2m',
    'dewpoint_2m',
    'precipitation',
    'surface_pressure',
    'cloud_cover',
    'wind_speed_10m',
    'wind_speed_80m',  # closest to 100m
    'wind_speed_120m',
    'wind_direction_10m',
    'wind_direction_80m',
    'wind_direction_120m',
]

# Daily aggregated variables
DAILY_VARIABLES = [
    'temperature_2m_max',
    'temperature_2m_min',
    'precipitation_sum',
    'wind_speed_10m_max',
    'shortwave_radiation_sum',
]


def download_historical_forecast(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    hourly_vars: List[str],
    daily_vars: List[str] = None
) -> Optional[pd.DataFrame]:
    """
    Download historical forecast data from Open-Meteo Historical Forecast API.

    Args:
        lat: Latitude
        lon: Longitude
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        hourly_vars: List of hourly variables
        daily_vars: List of daily variables (optional)

    Returns:
        DataFrame with forecast data or None
    """
    params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': start_date,
        'end_date': end_date,
        'hourly': ','.join(hourly_vars),
        'timezone': 'America/Chicago'
    }

    if daily_vars:
        params['daily'] = ','.join(daily_vars)

    try:
        response = requests.get(HISTORICAL_FORECAST_URL, params=params, timeout=120)
        response.raise_for_status()

        data = response.json()

        # Check for error in response
        if 'error' in data and data['error']:
            print(f"    API ERROR: {data.get('reason', 'Unknown error')}")
            return None

        if 'hourly' not in data:
            print(f"    WARNING: No hourly data in response")
            return None

        # Convert hourly data to DataFrame
        hourly_data = data['hourly']
        df = pd.DataFrame(hourly_data)

        # Convert time to datetime
        df['time'] = pd.to_datetime(df['time'])

        # Add daily data if available
        if 'daily' in data and daily_vars:
            daily_data = data['daily']
            df_daily = pd.DataFrame(daily_data)
            df_daily['time'] = pd.to_datetime(df_daily['time'])
            # Merge daily data onto hourly (broadcast daily values to all hours of that day)
            df['date'] = df['time'].dt.date
            df_daily['date'] = df_daily['time'].dt.date
            df = df.merge(df_daily.drop(columns=['time']), on='date', how='left', suffixes=('', '_daily'))
            df = df.drop(columns=['date'])

        return df

    except requests.exceptions.RequestException as e:
        print(f"    ERROR: API request failed: {e}")
        return None
    except Exception as e:
        print(f"    ERROR: {e}")
        return None


def download_all_locations(
    locations_df: pd.DataFrame,
    start_date: datetime,
    end_date: datetime,
    output_dir: Path,
    chunk_months: int = 3
):
    """
    Download historical forecast data for all locations.

    Args:
        locations_df: DataFrame with location information
        start_date: Start date
        end_date: End date
        output_dir: Output directory
        chunk_months: Download in chunks of N months (to avoid timeouts)
    """
    print("=" * 80)
    print("OPEN-METEO HISTORICAL FORECAST DOWNLOAD")
    print("=" * 80)
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    print(f"Locations: {len(locations_df)}")
    print(f"Chunk size: {chunk_months} months")
    print()
    print("Rate limiting: 0.15 seconds between calls (400 calls/min, safe margin)")
    print()

    # Create output directories
    csv_dir = output_dir / 'csv_files'
    csv_dir.mkdir(exist_ok=True, parents=True)

    parquet_dir = output_dir / 'parquet_files'
    parquet_dir.mkdir(exist_ok=True, parents=True)

    successful = 0
    failed = 0
    total_records = 0

    # Process each location
    for idx, row in locations_df.iterrows():
        location_name = row['name']
        lat = row['lat']
        lon = row['lon']

        print(f"\n[{idx+1}/{len(locations_df)}] {location_name}")
        print(f"  Coordinates: ({lat:.4f}, {lon:.4f})")

        safe_name = location_name.replace(' ', '_').replace('/', '_')
        csv_file = csv_dir / f"{safe_name}_historical_forecast.csv"
        parquet_file = parquet_dir / f"{safe_name}_historical_forecast.parquet"

        # Check if already downloaded
        if csv_file.exists():
            print(f"  ✓ Already downloaded (skipping)")
            successful += 1
            continue

        # Download in chunks to avoid timeouts
        all_chunks = []
        current_date = start_date
        chunk_num = 0

        while current_date < end_date:
            chunk_end = min(
                current_date + timedelta(days=chunk_months * 30),
                end_date
            )
            chunk_num += 1

            start_str = current_date.strftime('%Y-%m-%d')
            end_str = chunk_end.strftime('%Y-%m-%d')

            print(f"  Chunk {chunk_num}: {start_str} to {end_str}... ", end='', flush=True)

            df = download_historical_forecast(
                lat=lat,
                lon=lon,
                start_date=start_str,
                end_date=end_str,
                hourly_vars=HOURLY_VARIABLES,
                daily_vars=DAILY_VARIABLES
            )

            if df is not None and len(df) > 0:
                all_chunks.append(df)
                print(f"✓ ({len(df)} records)")
            else:
                print("✗")

            # Rate limiting: 0.15 seconds = 400 calls/min (safe margin under 600 limit)
            time.sleep(0.15)

            current_date = chunk_end

        # Combine all chunks for this location
        if all_chunks:
            df_combined = pd.concat(all_chunks, ignore_index=True)

            # Add metadata
            df_combined.insert(0, 'location_name', location_name)
            df_combined.insert(1, 'latitude', lat)
            df_combined.insert(2, 'longitude', lon)

            # Save CSV
            df_combined.to_csv(csv_file, index=False)

            # Save Parquet
            df_combined.to_parquet(parquet_file, compression='snappy')

            print(f"  ✓ Total: {len(df_combined):,} hourly records")
            successful += 1
            total_records += len(df_combined)
        else:
            print(f"  ✗ Failed to download any data")
            failed += 1

    # Create combined datasets
    print("\n" + "=" * 80)
    print("CREATING COMBINED DATASETS")
    print("=" * 80)

    csv_files = list(csv_dir.glob('*_historical_forecast.csv'))

    if csv_files:
        print(f"\nCombining {len(csv_files)} location files...")

        all_data = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file, parse_dates=['time'])
            all_data.append(df)

        combined = pd.concat(all_data, ignore_index=True)
        combined = combined.sort_values(['location_name', 'time'])

        # Save combined CSV
        combined_csv = output_dir / 'all_historical_forecasts.csv'
        combined.to_csv(combined_csv, index=False)
        print(f"  ✓ CSV: {combined_csv} ({len(combined):,} records)")

        # Save combined Parquet
        combined_parquet = output_dir / 'all_historical_forecasts.parquet'
        combined.to_parquet(combined_parquet, compression='snappy')
        print(f"  ✓ Parquet: {combined_parquet}")

        # Create summary
        summary = {
            'source': 'Open-Meteo Historical Forecast API',
            'url': 'https://historical-forecast-api.open-meteo.com',
            'description': 'Historical weather forecasts (archived model predictions)',
            'date_range': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'total_locations': successful,
            'failed_downloads': failed,
            'total_records': len(combined),
            'hourly_variables': HOURLY_VARIABLES,
            'daily_variables': DAILY_VARIABLES,
            'download_date': datetime.now().isoformat(),
            'use_cases': [
                'Forecast accuracy analysis',
                'Forecast vs actual comparison',
                'Weather prediction error correlation with energy prices',
                'Trading strategy backtesting with realistic forecasts',
                'Understanding market behavior based on forecasts'
            ]
        }

        summary_file = output_dir / 'historical_forecast_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\n  ✓ Summary: {summary_file}")

    # Final summary
    print("\n" + "=" * 80)
    print("DOWNLOAD COMPLETE")
    print("=" * 80)
    print(f"\nSuccessful: {successful}/{len(locations_df)}")
    print(f"Failed: {failed}/{len(locations_df)}")
    print(f"Total records: {total_records:,}")
    print(f"\nData saved to: {output_dir}")

    return successful > 0


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description='Download historical forecast data from Open-Meteo'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        default='2022-01-01',
        help='Start date (YYYY-MM-DD), default: 2022-01-01'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help='End date (YYYY-MM-DD), default: yesterday'
    )
    parser.add_argument(
        '--chunk-months',
        type=int,
        default=3,
        help='Download in chunks of N months (default: 3)'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test mode: download only 1 month of data for 1 location'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("OPEN-METEO HISTORICAL FORECAST DOWNLOADER")
    print("=" * 80)
    print("Using FREE non-commercial API tier")
    print()

    # Load environment
    load_dotenv()
    weather_dir = Path(os.getenv('WEATHER_DATA_DIR', '/pool/ssd8tb/data/weather_data'))
    weather_dir.mkdir(exist_ok=True, parents=True)

    forecast_dir = weather_dir / 'openmeteo_historical_forecasts'
    forecast_dir.mkdir(exist_ok=True, parents=True)

    # Load locations
    locations_file = weather_dir / 'weather_locations.csv'
    if not locations_file.exists():
        print(f"ERROR: Locations file not found: {locations_file}")
        return

    locations_df = pd.read_csv(locations_file)

    # Date range
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')

    if args.end_date:
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    else:
        # Default to yesterday (today's data might not be complete)
        end_date = datetime.now() - timedelta(days=1)

    # Test mode
    if args.test:
        print("TEST MODE: Downloading 1 month for Houston only")
        locations_df = locations_df[locations_df['name'] == 'CITY_Houston']
        end_date = start_date + timedelta(days=30)

    # Download
    success = download_all_locations(
        locations_df=locations_df,
        start_date=start_date,
        end_date=end_date,
        output_dir=forecast_dir,
        chunk_months=args.chunk_months
    )

    print("\n" + "=" * 80)
    print("WHAT YOU CAN DO WITH THIS DATA")
    print("=" * 80)
    print("✓ Compare forecasted vs actual weather")
    print("✓ Identify forecast errors and patterns")
    print("✓ Correlate forecast accuracy with energy price movements")
    print("✓ Analyze wind/solar generation forecast errors")
    print("✓ Backtest trading strategies with realistic forecast data")
    print("✓ Study how weather prediction errors affected market behavior")

    return 0 if success else 1


if __name__ == '__main__':
    exit(main())

#!/usr/bin/env python3
"""
Download historical weather FORECASTS from Open-Meteo.

This downloads what was PREDICTED at the time, not what actually happened.
Useful for analyzing forecast errors and their correlation with energy prices.

Open-Meteo Forecast Archive:
- Available from June 2022 to present
- Multiple forecast models (GFS, ECMWF, etc.)
- Forecast horizons: 0-16 days ahead
- Free, no API key required

Documentation: https://open-meteo.com/en/docs/ensemble-api
"""

import os
import pandas as pd
import requests
from datetime import datetime, timedelta
from pathlib import Path
import time
from typing import Optional, List
import json
from dotenv import load_dotenv

# Open-Meteo Ensemble/Forecast API endpoint
OPENMETEO_FORECAST_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"

# Variables to forecast
FORECAST_VARIABLES = [
    'temperature_2m_mean',
    'temperature_2m_max',
    'temperature_2m_min',
    'windspeed_10m_mean',
    'windspeed_100m_mean',
    'shortwave_radiation_sum',
    'precipitation_sum',
]


def download_forecast_archive(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    past_days: int = 7  # How many days ahead the forecast was made
) -> Optional[pd.DataFrame]:
    """
    Download historical weather forecasts from Open-Meteo.

    Args:
        lat: Latitude
        lon: Longitude
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        past_days: Days in the past to get forecasts for (1-16)

    Returns:
        DataFrame with forecast data or None
    """
    params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': start_date,
        'end_date': end_date,
        'daily': ','.join(FORECAST_VARIABLES),
        'timezone': 'America/Chicago',
        'past_days': past_days  # Get forecasts made N days ago
    }

    try:
        response = requests.get(OPENMETEO_FORECAST_URL, params=params, timeout=120)
        response.raise_for_status()

        data = response.json()

        if 'daily' not in data:
            print(f"    WARNING: No daily data in response")
            return None

        # Convert to DataFrame
        daily_data = data['daily']
        df = pd.DataFrame(daily_data)

        # Convert time to datetime
        df['time'] = pd.to_datetime(df['time'])
        df = df.set_index('time')

        return df

    except requests.exceptions.RequestException as e:
        print(f"    ERROR downloading: {e}")
        return None
    except Exception as e:
        print(f"    ERROR processing: {e}")
        return None


def main():
    """Main execution function."""
    # Setup directories
    load_dotenv()
    weather_dir = Path(os.getenv('WEATHER_DATA_DIR', '/pool/ssd8tb/data/weather_data'))
    weather_dir.mkdir(exist_ok=True, parents=True)

    forecast_dir = weather_dir / 'openmeteo_forecasts'
    forecast_dir.mkdir(exist_ok=True, parents=True)

    forecast_csv_dir = forecast_dir / 'csv_files'
    forecast_csv_dir.mkdir(exist_ok=True)

    forecast_parquet_dir = forecast_dir / 'parquet_files'
    forecast_parquet_dir.mkdir(exist_ok=True)

    # Load locations
    locations_file = weather_dir / 'weather_locations.csv'
    if not locations_file.exists():
        print(f"ERROR: Locations file not found: {locations_file}")
        print("Using ERCOT locations from weather_locations.csv")
        return

    locations_df = pd.read_csv(locations_file)

    # Date range: Open-Meteo forecast archive starts June 2022
    # We'll get forecasts from June 2022 to now
    start_date = datetime(2022, 6, 1)
    end_date = datetime.now()

    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    print("=" * 80)
    print("OPEN-METEO HISTORICAL WEATHER FORECASTS")
    print("=" * 80)
    print("Downloading what was PREDICTED, not what actually happened")
    print()
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    print(f"Forecast horizon: 7-day ahead forecasts")
    print(f"Purpose: Analyze forecast errors vs energy price movements")
    print()
    print("Note: This is complementary to actual weather data")
    print("      Use to study how forecast accuracy affects trading")
    print()

    # Forecast horizons to download
    # We'll get multiple horizons to analyze forecast accuracy over time
    forecast_horizons = {
        '1day': 1,    # Next-day forecast
        '3day': 3,    # 3-day ahead
        '7day': 7,    # Week ahead
    }

    successful = 0
    failed = 0

    print("=" * 80)
    print("DOWNLOADING FORECAST DATA")
    print("=" * 80)

    for idx, row in locations_df.iterrows():
        location_name = row['name']
        lat = row['lat']
        lon = row['lon']

        print(f"\n[{idx+1}/{len(locations_df)}] {location_name}")
        print(f"  Coordinates: ({lat:.4f}, {lon:.4f})")

        safe_name = location_name.replace(' ', '_').replace('/', '_')

        location_success = False

        # Download each forecast horizon
        for horizon_name, horizon_days in forecast_horizons.items():
            csv_file = forecast_csv_dir / f"{safe_name}_forecast_{horizon_name}.csv"

            # Check if already downloaded
            if csv_file.exists():
                print(f"  ✓ {horizon_name}: Already downloaded (skipping)")
                location_success = True
                continue

            print(f"  Downloading {horizon_name} forecast...")

            df = download_forecast_archive(
                lat=lat,
                lon=lon,
                start_date=start_str,
                end_date=end_str,
                past_days=horizon_days
            )

            if df is not None and len(df) > 0:
                # Add metadata
                df.insert(0, 'location_name', location_name)
                df.insert(1, 'latitude', lat)
                df.insert(2, 'longitude', lon)
                df.insert(3, 'forecast_horizon_days', horizon_days)

                # Save to CSV
                df.to_csv(csv_file)

                print(f"  ✓ {horizon_name}: {len(df)} forecast days")
                location_success = True

            else:
                print(f"  ✗ {horizon_name}: Failed")

            # Rate limiting
            time.sleep(15)  # 4 requests per minute

        if location_success:
            successful += 1
        else:
            failed += 1

    # Convert to Parquet
    print("\n" + "=" * 80)
    print("CONVERTING TO PARQUET FORMAT")
    print("=" * 80)

    csv_files = list(forecast_csv_dir.glob('*.csv'))
    print(f"\nConverting {len(csv_files)} CSV files to Parquet...")

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
            parquet_file = forecast_parquet_dir / csv_file.with_suffix('.parquet').name
            df.to_parquet(parquet_file, compression='snappy', index=True)
            print(f"  ✓ {parquet_file.name}")
        except Exception as e:
            print(f"  ERROR converting {csv_file.name}: {e}")

    # Create combined datasets
    if csv_files:
        print("\n" + "=" * 80)
        print("CREATING COMBINED DATASETS")
        print("=" * 80)

        for horizon_name in forecast_horizons.keys():
            print(f"\nCombining {horizon_name} forecasts...")

            horizon_files = list(forecast_csv_dir.glob(f'*_forecast_{horizon_name}.csv'))

            if not horizon_files:
                print(f"  No files for {horizon_name}")
                continue

            all_data = []
            for csv_file in horizon_files:
                df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
                all_data.append(df)

            combined = pd.concat(all_data, ignore_index=False)
            combined = combined.sort_index()

            # Save
            combined_csv = forecast_dir / f'all_forecasts_{horizon_name}.csv'
            combined.to_csv(combined_csv)
            print(f"  ✓ CSV: {combined_csv} ({len(combined):,} records)")

            combined_parquet = forecast_dir / f'all_forecasts_{horizon_name}.parquet'
            combined.to_parquet(combined_parquet, compression='snappy')
            print(f"  ✓ Parquet: {combined_parquet}")

        # Create summary
        summary = {
            'source': 'Open-Meteo Ensemble Forecast Archive',
            'url': 'https://open-meteo.com/en/docs/ensemble-api',
            'description': 'Historical weather forecasts (what was predicted)',
            'date_range': f"{start_date.date()} to {end_date.date()}",
            'forecast_horizons': list(forecast_horizons.keys()),
            'total_locations': successful,
            'failed_downloads': failed,
            'variables': FORECAST_VARIABLES,
            'use_cases': [
                'Forecast vs actual error analysis',
                'Correlation of forecast errors with price spikes',
                'Wind/solar production forecast accuracy',
                'Load forecast validation',
                'Trading strategy backtesting with realistic forecasts'
            ]
        }

        summary_file = forecast_dir / 'forecast_archive_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\n  ✓ Summary: {summary_file}")

    # Final summary
    print("\n" + "=" * 80)
    print("DOWNLOAD COMPLETE")
    print("=" * 80)
    print(f"\nSuccessful: {successful}/{len(locations_df)}")
    print(f"Failed: {failed}/{len(locations_df)}")
    print(f"\nData saved to: {forecast_dir}")

    print("\n" + "=" * 80)
    print("WHAT YOU CAN DO WITH FORECAST DATA")
    print("=" * 80)
    print("✓ Compare forecast vs actual weather")
    print("✓ Identify when forecast errors were large")
    print("✓ Correlate forecast errors with ERCOT price spikes")
    print("✓ Analyze wind/solar forecast accuracy")
    print("✓ Study how forecast improvements affected trading")
    print("✓ Backtest trading strategies with realistic forecasts")

    print("\n" + "=" * 80)
    print("ANALYSIS EXAMPLE")
    print("=" * 80)
    print("""
    # Load forecast and actual
    forecast = pd.read_parquet('weather_data/openmeteo_forecasts/all_forecasts_1day.parquet')
    actual = pd.read_parquet('weather_data/all_weather_data.parquet')

    # Calculate forecast error
    houston_forecast = forecast[forecast['location_name'] == 'CITY_Houston']['temperature_2m_mean']
    houston_actual = actual[actual['location_name'] == 'CITY_Houston']['T2M']

    error = houston_forecast - houston_actual
    large_errors = error[abs(error) > 5]  # >5°C error

    # Correlate with ERCOT prices on those days
    # Did forecast errors predict price spikes?
    """)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Download historical weather data from Open-Meteo API.

Open-Meteo provides free access to high-resolution weather reanalysis data:
- 11km resolution (vs NASA POWER's 50km)
- Hourly data (can aggregate to daily)
- Wind speed at 10m, 100m, 120m heights
- Solar radiation
- No API key required
- No rate limits for reasonable use

Documentation: https://open-meteo.com/en/docs/historical-weather-api
"""

import pandas as pd
import requests
from datetime import datetime, timedelta
from pathlib import Path
import time
from typing import Optional, Dict, List
import json
import numpy as np

# Open-Meteo API endpoint
OPENMETEO_BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Hourly weather variables to download
# See: https://open-meteo.com/en/docs/historical-weather-api
HOURLY_VARIABLES = [
    'temperature_2m',              # Air temperature at 2 meters (°C)
    'relativehumidity_2m',         # Relative humidity at 2 meters (%)
    'dewpoint_2m',                 # Dew point temperature (°C)
    'precipitation',               # Total precipitation (mm)
    'surface_pressure',            # Surface pressure (hPa)
    'cloudcover',                  # Total cloud cover (%)
    'windspeed_10m',              # Wind speed at 10 meters (km/h)
    'windspeed_100m',             # Wind speed at 100 meters (km/h) - CRITICAL for turbines
    'winddirection_10m',          # Wind direction at 10 meters (°)
    'winddirection_100m',         # Wind direction at 100 meters (°)
    'shortwave_radiation',        # Shortwave solar radiation (W/m²) - CRITICAL for solar
    'direct_radiation',           # Direct solar radiation (W/m²)
    'diffuse_radiation',          # Diffuse solar radiation (W/m²)
]

# Daily aggregations we want
DAILY_AGGREGATIONS = {
    'temperature_2m': ['mean', 'min', 'max'],
    'relativehumidity_2m': ['mean'],
    'dewpoint_2m': ['mean'],
    'precipitation': ['sum'],
    'surface_pressure': ['mean'],
    'cloudcover': ['mean'],
    'windspeed_10m': ['mean', 'max'],
    'windspeed_100m': ['mean', 'max'],
    'winddirection_10m': ['mean'],
    'winddirection_100m': ['mean'],
    'shortwave_radiation': ['sum', 'mean', 'max'],
    'direct_radiation': ['sum', 'mean'],
    'diffuse_radiation': ['sum', 'mean'],
}


def download_openmeteo_data(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    hourly_vars: List[str]
) -> Optional[pd.DataFrame]:
    """
    Download hourly weather data from Open-Meteo.

    Args:
        lat: Latitude
        lon: Longitude
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        hourly_vars: List of hourly variables to download

    Returns:
        DataFrame with hourly weather data or None
    """
    params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': start_date,
        'end_date': end_date,
        'hourly': ','.join(hourly_vars),
        'timezone': 'America/Chicago'  # Texas timezone
    }

    try:
        response = requests.get(OPENMETEO_BASE_URL, params=params, timeout=120)
        response.raise_for_status()

        data = response.json()

        if 'hourly' not in data:
            print(f"    WARNING: No hourly data in response")
            return None

        # Convert to DataFrame
        hourly_data = data['hourly']
        df = pd.DataFrame(hourly_data)

        # Convert time to datetime
        df['time'] = pd.to_datetime(df['time'])
        df = df.set_index('time')

        return df

    except requests.exceptions.RequestException as e:
        print(f"    ERROR downloading data: {e}")
        return None
    except Exception as e:
        print(f"    ERROR processing data: {e}")
        return None


def aggregate_to_daily(df_hourly: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate hourly data to daily using appropriate methods.

    Args:
        df_hourly: DataFrame with hourly data

    Returns:
        DataFrame with daily aggregated data
    """
    daily_data = {}

    for col in df_hourly.columns:
        if col in DAILY_AGGREGATIONS:
            agg_methods = DAILY_AGGREGATIONS[col]
            for method in agg_methods:
                if method == 'sum':
                    daily_data[f"{col}_sum"] = df_hourly[col].resample('D').sum()
                elif method == 'mean':
                    daily_data[f"{col}_mean"] = df_hourly[col].resample('D').mean()
                elif method == 'min':
                    daily_data[f"{col}_min"] = df_hourly[col].resample('D').min()
                elif method == 'max':
                    daily_data[f"{col}_max"] = df_hourly[col].resample('D').max()

    df_daily = pd.DataFrame(daily_data)
    return df_daily


def main():
    """Main execution function."""
    # Setup directories
    project_root = Path(__file__).parent
    weather_dir = project_root / 'weather_data'

    openmeteo_dir = weather_dir / 'openmeteo'
    openmeteo_dir.mkdir(exist_ok=True)

    # Create subdirectories for hourly and daily data
    hourly_csv_dir = openmeteo_dir / 'hourly_csv'
    hourly_csv_dir.mkdir(exist_ok=True)

    hourly_parquet_dir = openmeteo_dir / 'hourly_parquet'
    hourly_parquet_dir.mkdir(exist_ok=True)

    daily_csv_dir = openmeteo_dir / 'daily_csv'
    daily_csv_dir.mkdir(exist_ok=True)

    daily_parquet_dir = openmeteo_dir / 'daily_parquet'
    daily_parquet_dir.mkdir(exist_ok=True)

    # Load locations
    locations_file = weather_dir / 'weather_locations.csv'
    locations_df = pd.read_csv(locations_file)

    # Date range: last 5 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)

    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    print("=" * 80)
    print("OPEN-METEO HISTORICAL WEATHER DATA DOWNLOAD")
    print("=" * 80)
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    print(f"Resolution: 11km (vs NASA POWER's 50km)")
    print(f"Frequency: Hourly (will also aggregate to daily)")
    print(f"Variables: {len(HOURLY_VARIABLES)}")
    print(f"  - Wind at 100m height (modern turbine hub height)")
    print(f"  - Solar radiation (direct + diffuse)")
    print(f"  - Temperature, humidity, precipitation, pressure")
    print()

    # Download data
    print("=" * 80)
    print("DOWNLOADING HOURLY WEATHER DATA")
    print("=" * 80)

    successful = 0
    failed = 0
    total_hourly_records = 0
    total_daily_records = 0

    for idx, row in locations_df.iterrows():
        location_name = row['name']
        lat = row['lat']
        lon = row['lon']

        print(f"\n[{idx+1}/{len(locations_df)}] {location_name}")
        print(f"  Coordinates: ({lat:.4f}, {lon:.4f})")

        # Check if already downloaded (for resume capability)
        safe_name = location_name.replace(' ', '_').replace('/', '_')
        hourly_csv = hourly_csv_dir / f"{safe_name}_hourly.csv"
        daily_csv = daily_csv_dir / f"{safe_name}_daily.csv"

        if hourly_csv.exists() and daily_csv.exists():
            print(f"  ✓ Already downloaded (skipping)")
            successful += 1
            continue

        # Download hourly data
        df_hourly = download_openmeteo_data(
            lat=lat,
            lon=lon,
            start_date=start_str,
            end_date=end_str,
            hourly_vars=HOURLY_VARIABLES
        )

        if df_hourly is not None and len(df_hourly) > 0:
            # Add metadata
            df_hourly.insert(0, 'location_name', location_name)
            df_hourly.insert(1, 'latitude', lat)
            df_hourly.insert(2, 'longitude', lon)

            # Save hourly data
            df_hourly.to_csv(hourly_csv)

            hourly_records = len(df_hourly)
            total_hourly_records += hourly_records

            print(f"  ✓ Hourly: {hourly_records:,} records ({hourly_records/24:.0f} days)")

            # Aggregate to daily
            df_daily = aggregate_to_daily(df_hourly.drop(columns=['location_name', 'latitude', 'longitude']))

            # Add metadata to daily
            df_daily.insert(0, 'location_name', location_name)
            df_daily.insert(1, 'latitude', lat)
            df_daily.insert(2, 'longitude', lon)

            # Save daily data (using same safe_name from above)
            df_daily.to_csv(daily_csv)

            daily_records = len(df_daily)
            total_daily_records += daily_records

            print(f"  ✓ Daily:  {daily_records:,} records")

            successful += 1

            # Longer delay to respect rate limits (5 requests per minute max)
            time.sleep(15)

        else:
            print(f"  ✗ Failed to download data")
            failed += 1

    print("\n" + "=" * 80)
    print("CONVERTING TO PARQUET FORMAT")
    print("=" * 80)

    # Convert hourly CSV to Parquet
    hourly_csv_files = list(hourly_csv_dir.glob('*.csv'))
    print(f"\nConverting {len(hourly_csv_files)} hourly CSV files to Parquet...")

    for csv_file in hourly_csv_files:
        try:
            df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
            parquet_file = hourly_parquet_dir / csv_file.with_suffix('.parquet').name
            df.to_parquet(parquet_file, compression='snappy', index=True)
            print(f"  ✓ {parquet_file.name}")
        except Exception as e:
            print(f"  ERROR converting {csv_file.name}: {e}")

    # Convert daily CSV to Parquet
    daily_csv_files = list(daily_csv_dir.glob('*.csv'))
    print(f"\nConverting {len(daily_csv_files)} daily CSV files to Parquet...")

    for csv_file in daily_csv_files:
        try:
            df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
            parquet_file = daily_parquet_dir / csv_file.with_suffix('.parquet').name
            df.to_parquet(parquet_file, compression='snappy', index=True)
            print(f"  ✓ {parquet_file.name}")
        except Exception as e:
            print(f"  ERROR converting {csv_file.name}: {e}")

    # Create combined datasets
    if hourly_csv_files:
        print("\n" + "=" * 80)
        print("CREATING COMBINED DATASETS")
        print("=" * 80)

        # Combined hourly
        print("\nCreating combined hourly dataset...")
        all_hourly = []
        for csv_file in hourly_csv_files:
            df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
            all_hourly.append(df)

        combined_hourly = pd.concat(all_hourly, ignore_index=False)
        combined_hourly = combined_hourly.sort_index()

        combined_hourly_csv = openmeteo_dir / 'all_openmeteo_hourly.csv'
        combined_hourly.to_csv(combined_hourly_csv)
        print(f"  ✓ CSV: {combined_hourly_csv} ({len(combined_hourly):,} records)")

        combined_hourly_parquet = openmeteo_dir / 'all_openmeteo_hourly.parquet'
        combined_hourly.to_parquet(combined_hourly_parquet, compression='snappy')
        print(f"  ✓ Parquet: {combined_hourly_parquet}")

        # Combined daily
        print("\nCreating combined daily dataset...")
        all_daily = []
        for csv_file in daily_csv_files:
            df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
            all_daily.append(df)

        combined_daily = pd.concat(all_daily, ignore_index=False)
        combined_daily = combined_daily.sort_index()

        combined_daily_csv = openmeteo_dir / 'all_openmeteo_daily.csv'
        combined_daily.to_csv(combined_daily_csv)
        print(f"  ✓ CSV: {combined_daily_csv} ({len(combined_daily):,} records)")

        combined_daily_parquet = openmeteo_dir / 'all_openmeteo_daily.parquet'
        combined_daily.to_parquet(combined_daily_parquet, compression='snappy')
        print(f"  ✓ Parquet: {combined_daily_parquet}")

        # Create summary
        summary = {
            'source': 'Open-Meteo Historical Weather API',
            'url': 'https://open-meteo.com',
            'total_locations': successful,
            'failed_downloads': failed,
            'date_range': f"{start_date.date()} to {end_date.date()}",
            'resolution': '11 km',
            'timezone': 'America/Chicago',
            'hourly_records': len(combined_hourly),
            'daily_records': len(combined_daily),
            'hourly_variables': HOURLY_VARIABLES,
            'daily_aggregations': {k: v for k, v in DAILY_AGGREGATIONS.items()},
            'key_features': [
                'Wind speed at 100m (turbine height)',
                'Solar radiation (direct + diffuse + shortwave)',
                'Higher resolution than NASA POWER (11km vs 50km)',
                'Hourly temporal resolution',
                'Free, no API key required'
            ]
        }

        summary_file = openmeteo_dir / 'download_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\n  ✓ Summary: {summary_file}")

    # Final summary
    print("\n" + "=" * 80)
    print("DOWNLOAD COMPLETE")
    print("=" * 80)
    print(f"\nSuccessful: {successful}/{len(locations_df)}")
    print(f"Failed: {failed}/{len(locations_df)}")
    print(f"\nTotal hourly records: {total_hourly_records:,}")
    print(f"Total daily records: {total_daily_records:,}")
    print(f"\nData saved to:")
    print(f"  Hourly CSV: {hourly_csv_dir}")
    print(f"  Hourly Parquet: {hourly_parquet_dir}")
    print(f"  Daily CSV: {daily_csv_dir}")
    print(f"  Daily Parquet: {daily_parquet_dir}")
    print(f"  Combined: {openmeteo_dir}")

    print("\n" + "=" * 80)
    print("KEY ADVANTAGES OF OPEN-METEO DATA")
    print("=" * 80)
    print("✓ Wind at 100m height (modern turbine hub height)")
    print("✓ Higher resolution (11km vs NASA's 50km)")
    print("✓ Hourly data (24x more granular)")
    print("✓ Direct + diffuse solar radiation")
    print("✓ Completely free, no API key needed")
    print("✓ Perfect for energy market correlation")


if __name__ == '__main__':
    main()

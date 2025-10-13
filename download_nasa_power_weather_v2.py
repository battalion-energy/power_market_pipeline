#!/usr/bin/env python3
"""
Download NASA POWER satellite weather data for all ISOs.

Features:
- Multi-ISO support
- Incremental downloads (only new data since last update)
- Resume capability
- ISO filtering
- Designed for cron jobs

Usage:
  # Initial full download
  python download_nasa_power_weather_v2.py

  # Incremental update (only new data)
  python download_nasa_power_weather_v2.py --incremental

  # Download for specific ISO
  python download_nasa_power_weather_v2.py --iso ERCOT

  # Incremental update for specific ISO
  python download_nasa_power_weather_v2.py --incremental --iso CAISO
"""

import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import time
import json
import argparse
from typing import Optional
from dotenv import load_dotenv

# NASA POWER API configuration
BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

# Weather parameters to download
PARAMETERS = [
    'T2M',              # Temperature at 2 Meters (°C)
    'T2M_MAX',          # Temperature at 2 Meters Maximum (°C)
    'T2M_MIN',          # Temperature at 2 Meters Minimum (°C)
    'WS10M',            # Wind Speed at 10 Meters (m/s)
    'WS50M',            # Wind Speed at 50 Meters (m/s)
    'WD10M',            # Wind Direction at 10 Meters (Degrees)
    'WD50M',            # Wind Direction at 50 Meters (Degrees)
    'ALLSKY_SFC_SW_DWN',  # All Sky Surface Shortwave Downward Irradiance (kW-hr/m^2/day)
    'CLRSKY_SFC_SW_DWN',  # Clear Sky Surface Shortwave Downward Irradiance (kW-hr/m^2/day)
    'PRECTOTCORR',      # Precipitation Corrected (mm/day)
    'RH2M',             # Relative Humidity at 2 Meters (%)
    'PS',               # Surface Pressure (kPa)
]


def get_last_date_in_file(file_path: Path) -> Optional[datetime]:
    """Get the last date in an existing CSV file."""
    if not file_path.exists():
        return None

    try:
        df = pd.read_csv(file_path, parse_dates=['date'])
        if len(df) == 0:
            return None
        return df['date'].max()
    except Exception as e:
        print(f"    Warning: Could not read {file_path}: {e}")
        return None


def download_location_weather(
    location_name: str,
    lat: float,
    lon: float,
    start_date: datetime,
    end_date: datetime,
    iso: str
) -> Optional[pd.DataFrame]:
    """
    Download weather data for a single location from NASA POWER API.

    Args:
        location_name: Name identifier for the location
        lat: Latitude
        lon: Longitude
        start_date: Start date for data
        end_date: End date for data
        iso: ISO/region name

    Returns:
        DataFrame with weather data or None if error
    """
    # Build API URL
    params = {
        'parameters': ','.join(PARAMETERS),
        'community': 'RE',  # Renewable Energy community
        'longitude': lon,
        'latitude': lat,
        'start': start_date.strftime('%Y%m%d'),
        'end': end_date.strftime('%Y%m%d'),
        'format': 'JSON'
    }

    try:
        response = requests.get(BASE_URL, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()

        # Extract parameters
        if 'properties' not in data or 'parameter' not in data['properties']:
            print(f"    ERROR: Unexpected API response format")
            return None

        params_data = data['properties']['parameter']

        # Convert to DataFrame
        df_data = {}
        for param, values in params_data.items():
            df_data[param] = values

        df = pd.DataFrame(df_data)

        # Reset index to get dates as column
        df.index.name = 'date'
        df.reset_index(inplace=True)

        # Convert date strings to datetime
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')

        # Add metadata columns
        df['location_name'] = location_name
        df['iso'] = iso
        df['latitude'] = lat
        df['longitude'] = lon

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
    incremental: bool = False,
    iso_filter: Optional[str] = None,
    combine_only: bool = False
):
    """
    Download weather data for all locations.

    Args:
        locations_df: DataFrame with location information
        start_date: Start date for data
        end_date: End date for data
        output_dir: Output directory
        incremental: If True, only download data since last update
        iso_filter: If provided, only download for this ISO
    """
    # Filter by ISO if specified
    if iso_filter:
        locations_df = locations_df[locations_df['iso'] == iso_filter]
        print(f"Filtered to {len(locations_df)} locations in {iso_filter}")

    # Create output directories
    csv_dir = output_dir / 'csv_files'
    csv_dir.mkdir(exist_ok=True, parents=True)

    parquet_dir = output_dir / 'parquet_files'
    parquet_dir.mkdir(exist_ok=True, parents=True)

    print(f"\nDownloading data for {len(locations_df)} locations")
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    print(f"Output directory: {output_dir}")
    print()

    successful = 0
    failed = 0
    skipped = 0

    if not combine_only:
        for idx, row in locations_df.iterrows():
            location_name = row['name']
            lat = row['lat']
            lon = row['lon']
            iso = row['iso']

            # Create safe filename
            safe_name = location_name.replace(' ', '_').replace('/', '_')
            csv_file = csv_dir / f"{safe_name}.csv"

            # Check for incremental update
            download_start = start_date
            if incremental and csv_file.exists():
                last_date = get_last_date_in_file(csv_file)
                if last_date:
                    # Start from day after last date
                    download_start = last_date + timedelta(days=1)

                    if download_start > end_date:
                        print(f"  {location_name} ({iso}): Already up to date")
                        skipped += 1
                        continue

                    print(f"  {location_name} ({iso}): Updating from {download_start.date()}... ", end='', flush=True)
                else:
                    print(f"  {location_name} ({iso}): Downloading... ", end='', flush=True)
            else:
                print(f"  {location_name} ({iso}): Downloading... ", end='', flush=True)

            # Download data
            df = download_location_weather(
                location_name=location_name,
                lat=lat,
                lon=lon,
                start_date=download_start,
                end_date=end_date,
                iso=iso
            )

            if df is not None:
                # If incremental, append to existing data
                if incremental and csv_file.exists() and download_start > start_date:
                    try:
                        existing_df = pd.read_csv(csv_file, parse_dates=['date'])
                        df = pd.concat([existing_df, df], ignore_index=True)
                        df = df.drop_duplicates(subset=['date'], keep='last')
                        df = df.sort_values('date')
                    except Exception as e:
                        print(f"\n    Warning: Could not append to existing file: {e}")

                # Save CSV
                df.to_csv(csv_file, index=False)

                # Save Parquet
                parquet_file = parquet_dir / f"{safe_name}.parquet"
                df.to_parquet(parquet_file, compression='snappy')

                print(f"✓ ({len(df)} records)")
                successful += 1
            else:
                print("✗")
                failed += 1

            # Rate limiting
            time.sleep(0.5)
    else:
        print("Skipping downloads (--combine-only mode)")
        print()

    # Create combined files
    print("\n" + "="*80)
    print("Creating combined dataset...")

    all_data = []
    for csv_file in csv_dir.glob('*.csv'):
        df = pd.read_csv(csv_file, parse_dates=['date'])
        all_data.append(df)

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.sort_values(['location_name', 'date'])

        # Save combined CSV
        combined_csv = output_dir / 'all_weather_data.csv'
        combined_df.to_csv(combined_csv, index=False)
        print(f"  CSV: {combined_csv}")

        # Save combined Parquet
        combined_parquet = output_dir / 'all_weather_data.parquet'
        combined_df.to_parquet(combined_parquet, compression='snappy')
        print(f"  Parquet: {combined_parquet}")

        # Create per-ISO Parquet files
        iso_parquet_dir = output_dir / 'parquet_by_iso'
        iso_parquet_dir.mkdir(exist_ok=True)

        print("\nCreating per-ISO Parquet files...")
        # Filter out NaN values before sorting
        iso_list = combined_df['iso'].dropna().unique()
        for iso in sorted(iso_list):
            iso_df = combined_df[combined_df['iso'] == iso]
            iso_parquet = iso_parquet_dir / f'{iso}_weather_data.parquet'
            iso_df.to_parquet(iso_parquet, compression='snappy')
            n_locs = iso_df['location_name'].nunique()
            print(f"  {iso}: {len(iso_df):,} records ({n_locs} locations) -> {iso_parquet.name}")

        # Summary
        print("\n" + "="*80)
        print("DOWNLOAD SUMMARY")
        print("="*80)
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Skipped (up to date): {skipped}")
        print(f"Total records: {len(combined_df):,}")
        print(f"Date range: {combined_df['date'].min().date()} to {combined_df['date'].max().date()}")

        # ISO breakdown
        print("\nBy ISO:")
        for iso in sorted(iso_list):
            iso_df = combined_df[combined_df['iso'] == iso]
            n_locs = iso_df['location_name'].nunique()
            n_records = len(iso_df)
            print(f"  {iso}: {n_locs} locations, {n_records:,} records")

        # Save metadata
        metadata = {
            'last_update': datetime.now().isoformat(),
            'date_range': {
                'start': combined_df['date'].min().isoformat(),
                'end': combined_df['date'].max().isoformat()
            },
            'total_locations': combined_df['location_name'].nunique(),
            'total_records': len(combined_df),
            'isos': sorted(iso_list),
            'parameters': PARAMETERS,
            'source': 'NASA POWER API',
            'download_stats': {
                'successful': successful,
                'failed': failed,
                'skipped': skipped
            }
        }

        metadata_file = output_dir / 'download_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        print(f"\nMetadata saved to: {metadata_file}")


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description='Download NASA POWER weather data for all ISOs')
    parser.add_argument('--incremental', action='store_true',
                       help='Only download new data since last update')
    parser.add_argument('--iso', type=str,
                       help='Download data for specific ISO only (e.g., ERCOT, CAISO)')
    parser.add_argument('--start-date', type=str,
                       help='Start date (YYYY-MM-DD), default: 2019-01-01')
    parser.add_argument('--end-date', type=str,
                       help='End date (YYYY-MM-DD), default: today')
    parser.add_argument('--combine-only', action='store_true',
                       help='Only create combined/per-ISO files from existing CSVs (skip downloads)')

    args = parser.parse_args()

    print("="*80)
    print("NASA POWER WEATHER DATA DOWNLOADER (Multi-ISO)")
    print("="*80)
    print()

    # Setup directories from environment
    load_dotenv()
    weather_dir = Path(os.getenv('WEATHER_DATA_DIR', '/pool/ssd8tb/data/weather_data'))
    weather_dir.mkdir(exist_ok=True, parents=True)

    # Load locations
    locations_file = weather_dir / 'weather_locations_all_isos.csv'
    if not locations_file.exists():
        print(f"ERROR: Locations file not found: {locations_file}")
        print("Run create_multi_iso_weather_locations.py first")
        return

    locations_df = pd.read_csv(locations_file)
    print(f"Loaded {len(locations_df)} locations from {locations_file}")

    # Date range
    if args.start_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    else:
        start_date = datetime(2019, 1, 1)

    if args.end_date:
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    else:
        end_date = datetime.now()

    # Download mode
    if args.combine_only:
        print("Mode: COMBINE ONLY (skip downloads, create combined files)")
    elif args.incremental:
        print("Mode: INCREMENTAL (only new data since last update)")
    else:
        print("Mode: FULL (complete date range)")

    if args.iso:
        print(f"ISO Filter: {args.iso}")

    # Download data
    download_all_locations(
        locations_df=locations_df,
        start_date=start_date,
        end_date=end_date,
        output_dir=weather_dir,
        incremental=args.incremental,
        iso_filter=args.iso,
        combine_only=args.combine_only
    )

    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()

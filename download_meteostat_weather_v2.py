#!/usr/bin/env python3
"""
Download Meteostat ground station weather data for all ISOs.

Features:
- Multi-ISO support
- Incremental downloads (only new data since last update)
- Resume capability
- ISO filtering
- Designed for cron jobs

Usage:
  # Initial full download
  python download_meteostat_weather_v2.py

  # Incremental update (only new data)
  python download_meteostat_weather_v2.py --incremental

  # Download for specific ISO
  python download_meteostat_weather_v2.py --iso ERCOT

  # Incremental update for specific ISO
  python download_meteostat_weather_v2.py --incremental --iso CAISO
"""

import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import time
import json
import argparse
from typing import Optional
from meteostat import Point, Daily, Stations

def get_last_date_in_file(file_path: Path) -> Optional[datetime]:
    """Get the last date in an existing CSV file."""
    if not file_path.exists():
        return None

    try:
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        if len(df) == 0:
            return None
        return df.index.max()
    except Exception as e:
        print(f"    Warning: Could not read {file_path}: {e}")
        return None


def find_nearest_station(lat: float, lon: float) -> Optional[dict]:
    """
    Find the nearest Meteostat weather station for a location.

    Args:
        lat: Latitude
        lon: Longitude

    Returns:
        Dictionary with station info or None
    """
    try:
        # Find nearest station
        stations = Stations()
        stations = stations.nearby(lat, lon)
        station = stations.fetch(1)

        if station.empty:
            return None

        return {
            'station_id': station.index[0],
            'station_name': station.iloc[0]['name'],
            'distance_km': station.iloc[0]['distance'] / 1000,  # Convert m to km
            'latitude': station.iloc[0]['latitude'],
            'longitude': station.iloc[0]['longitude']
        }
    except Exception as e:
        print(f"      Error finding station: {e}")
        return None


def download_station_data(
    location_name: str,
    lat: float,
    lon: float,
    start_date: datetime,
    end_date: datetime,
    iso: str
) -> Optional[pd.DataFrame]:
    """
    Download weather data from nearest station to a location.

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
    # Find nearest station
    station_info = find_nearest_station(lat, lon)

    if not station_info:
        return None

    station_id = station_info['station_id']
    distance_km = station_info['distance_km']

    try:
        # Get data
        point = Point(lat, lon)
        data = Daily(point, start_date, end_date)
        df = data.fetch()

        if df.empty:
            return None

        # Add metadata columns
        df['location_name'] = location_name
        df['iso'] = iso
        df['latitude'] = lat
        df['longitude'] = lon
        df['station_id'] = station_id
        df['station_name'] = station_info['station_name']
        df['distance_km'] = distance_km

        return df

    except Exception as e:
        print(f"      Error downloading data: {e}")
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
    meteostat_dir = output_dir / 'meteostat_stations'
    meteostat_dir.mkdir(exist_ok=True, parents=True)

    csv_dir = meteostat_dir / 'csv_files'
    csv_dir.mkdir(exist_ok=True)

    parquet_dir = meteostat_dir / 'parquet_files'
    parquet_dir.mkdir(exist_ok=True)

    print(f"\nDownloading data for {len(locations_df)} locations")
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    print(f"Output directory: {meteostat_dir}")
    print()

    successful = 0
    failed = 0
    skipped = 0
    station_mapping = {}

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
            df = download_station_data(
                location_name=location_name,
                lat=lat,
                lon=lon,
                start_date=download_start,
                end_date=end_date,
                iso=iso
            )

            if df is not None:
                # Store station mapping
                station_mapping[location_name] = {
                    'station_id': df['station_id'].iloc[0],
                    'station_name': df['station_name'].iloc[0],
                    'distance_km': round(df['distance_km'].iloc[0], 2),
                    'iso': iso
                }

                # If incremental, append to existing data
                if incremental and csv_file.exists() and download_start > start_date:
                    try:
                        existing_df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
                        df = pd.concat([existing_df, df])
                        df = df[~df.index.duplicated(keep='last')]
                        df = df.sort_index()
                    except Exception as e:
                        print(f"\n    Warning: Could not append to existing file: {e}")

                # Save CSV
                df.to_csv(csv_file)

                # Save Parquet
                parquet_file = parquet_dir / f"{safe_name}.parquet"
                df.to_parquet(parquet_file, compression='snappy')

                print(f"✓ ({len(df)} records, {df['distance_km'].iloc[0]:.1f} km)")
                successful += 1
            else:
                print("✗ (no station found)")
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
        df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
        all_data.append(df)

    if all_data:
        combined_df = pd.concat(all_data)
        combined_df = combined_df.sort_index()

        # Convert station_id and station_name to string (mixed types cause Parquet errors)
        if 'station_id' in combined_df.columns:
            combined_df['station_id'] = combined_df['station_id'].astype(str)
        if 'station_name' in combined_df.columns:
            combined_df['station_name'] = combined_df['station_name'].astype(str)

        # Save combined CSV
        combined_csv = meteostat_dir / 'all_meteostat_station_data.csv'
        combined_df.to_csv(combined_csv)
        print(f"  CSV: {combined_csv}")

        # Save combined Parquet
        combined_parquet = meteostat_dir / 'all_meteostat_station_data.parquet'
        combined_df.to_parquet(combined_parquet, compression='snappy')
        print(f"  Parquet: {combined_parquet}")

        # Create per-ISO Parquet files
        iso_parquet_dir = meteostat_dir / 'parquet_by_iso'
        iso_parquet_dir.mkdir(exist_ok=True)

        print("\nCreating per-ISO Parquet files...")
        # Filter out NaN values before sorting
        iso_list = combined_df['iso'].dropna().unique()
        for iso in sorted(iso_list):
            iso_df = combined_df[combined_df['iso'] == iso]
            iso_parquet = iso_parquet_dir / f'{iso}_meteostat_data.parquet'
            iso_df.to_parquet(iso_parquet, compression='snappy')
            n_locs = iso_df['location_name'].nunique()
            print(f"  {iso}: {len(iso_df):,} records ({n_locs} locations) -> {iso_parquet.name}")

        # Save station mapping
        mapping_file = meteostat_dir / 'station_mapping.json'
        with open(mapping_file, 'w') as f:
            json.dump(station_mapping, f, indent=2)
        print(f"  Station mapping: {mapping_file}")

        # Summary
        print("\n" + "="*80)
        print("DOWNLOAD SUMMARY")
        print("="*80)
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Skipped (up to date): {skipped}")
        print(f"Total records: {len(combined_df):,}")
        print(f"Date range: {combined_df.index.min().date()} to {combined_df.index.max().date()}")

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
                'start': combined_df.index.min().isoformat(),
                'end': combined_df.index.max().isoformat()
            },
            'total_locations': combined_df['location_name'].nunique(),
            'total_records': len(combined_df),
            'isos': sorted(iso_list),
            'source': 'Meteostat',
            'download_stats': {
                'successful': successful,
                'failed': failed,
                'skipped': skipped
            }
        }

        metadata_file = meteostat_dir / 'download_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        print(f"\nMetadata saved to: {metadata_file}")


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description='Download Meteostat weather data for all ISOs')
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
    print("METEOSTAT WEATHER DATA DOWNLOADER (Multi-ISO)")
    print("="*80)
    print()

    # Setup directories
    weather_dir = Path('/pool/ssd8tb/data/weather_data')
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

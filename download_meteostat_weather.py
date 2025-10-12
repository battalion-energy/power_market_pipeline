#!/usr/bin/env python3
"""
Download weather station data using Meteostat library.

Meteostat provides free access to historical weather data from ground-based stations
worldwide. No API key required!

Documentation: https://dev.meteostat.net/python/
"""

import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import time
from typing import Optional
import json

try:
    from meteostat import Point, Daily, Stations
except ImportError:
    print("ERROR: meteostat library not installed")
    print("Install with: uv pip install meteostat")
    exit(1)


def find_nearest_station(lat: float, lon: float, radius_km: int = 50) -> Optional[dict]:
    """
    Find the nearest weather station to a location.

    Args:
        lat: Latitude
        lon: Longitude
        radius_km: Search radius in kilometers

    Returns:
        Station info dict or None
    """
    try:
        # Create Point
        location = Point(lat, lon)

        # Find nearby stations
        stations = Stations()
        stations = stations.nearby(lat, lon)
        stations = stations.fetch(limit=10)

        if stations.empty:
            return None

        # Get the first (closest) station
        station = stations.iloc[0]

        return {
            'id': station.name,
            'name': station.name,
            'latitude': station['latitude'],
            'longitude': station['longitude'],
            'elevation': station['elevation'],
            'distance_km': station.get('distance', 0) / 1000  # Convert m to km
        }

    except Exception as e:
        print(f"    ERROR finding station: {e}")
        return None


def download_weather_data(
    lat: float,
    lon: float,
    start_date: datetime,
    end_date: datetime
) -> Optional[pd.DataFrame]:
    """
    Download daily weather data for a location.

    Args:
        lat: Latitude
        lon: Longitude
        start_date: Start date
        end_date: End date

    Returns:
        DataFrame with weather data or None
    """
    try:
        # Create Point
        location = Point(lat, lon)

        # Get daily data
        data = Daily(location, start_date, end_date)
        df = data.fetch()

        if df.empty:
            return None

        return df

    except Exception as e:
        print(f"    ERROR downloading data: {e}")
        return None


def main():
    """Main execution function."""
    # Setup directories
    project_root = Path(__file__).parent
    # Use SSD storage for weather data
    weather_dir = Path('/pool/ssd8tb/data/weather_data')
    weather_dir.mkdir(exist_ok=True, parents=True)

    meteostat_dir = weather_dir / 'meteostat_stations'
    meteostat_dir.mkdir(exist_ok=True, parents=True)

    meteostat_csv_dir = meteostat_dir / 'csv_files'
    meteostat_csv_dir.mkdir(exist_ok=True)

    meteostat_parquet_dir = meteostat_dir / 'parquet_files'
    meteostat_parquet_dir.mkdir(exist_ok=True)

    # Load locations
    locations_file = weather_dir / 'weather_locations.csv'
    locations_df = pd.read_csv(locations_file)

    # Date range: from 2019-01-01 to present
    end_date = datetime.now()
    start_date = datetime(2019, 1, 1)

    print(f"Downloading Meteostat weather station data from {start_date.date()} to {end_date.date()}")
    print(f"Source: Ground-based weather stations (NOAA, DWD, and others)\n")

    # Find stations for all locations
    print("=" * 80)
    print("FINDING NEAREST WEATHER STATIONS")
    print("=" * 80)

    location_stations = []

    for idx, row in locations_df.iterrows():
        location_name = row['name']
        lat = row['lat']
        lon = row['lon']

        print(f"\n[{idx+1}/{len(locations_df)}] {location_name}")
        print(f"  Coordinates: ({lat:.4f}, {lon:.4f})")

        # Find nearest station
        station = find_nearest_station(lat, lon, radius_km=100)

        if station:
            print(f"  Found: {station['name']} - {station['distance_km']:.1f} km away")
            location_stations.append({
                'location_name': location_name,
                'lat': lat,
                'lon': lon,
                'station': station
            })
        else:
            print(f"  WARNING: No station found within 100km")

        time.sleep(0.1)  # Small delay

    # Save station mapping
    station_mapping_file = meteostat_dir / 'station_mapping.json'
    with open(station_mapping_file, 'w') as f:
        json.dump(location_stations, f, indent=2, default=str)
    print(f"\nSaved station mapping to {station_mapping_file}")

    # Download data
    print("\n" + "=" * 80)
    print("DOWNLOADING WEATHER STATION DATA")
    print("=" * 80)

    successful = 0
    failed = 0

    for i, loc_data in enumerate(location_stations, 1):
        location_name = loc_data['location_name']
        lat = loc_data['lat']
        lon = loc_data['lon']
        station = loc_data.get('station')

        print(f"\n[{i}/{len(location_stations)}] {location_name}")

        if station:
            station_name = station['name']
            dist = station.get('distance_km', 0)
            print(f"  Station: {station_name} ({dist:.1f} km)")

        df = download_weather_data(lat, lon, start_date, end_date)

        if df is not None and len(df) > 0:
            # Add metadata columns
            df.insert(0, 'location_name', location_name)
            df.insert(1, 'latitude', lat)
            df.insert(2, 'longitude', lon)

            if station:
                df.insert(3, 'station_id', str(station['id']))  # Convert to string for consistency
                df.insert(4, 'station_name', str(station['name']))
                df.insert(5, 'distance_km', station['distance_km'])

            # Save to CSV
            safe_name = location_name.replace(' ', '_').replace('/', '_')
            csv_file = meteostat_csv_dir / f"{safe_name}.csv"
            df.to_csv(csv_file)

            print(f"  ✓ Downloaded {len(df)} days of data")
            print(f"    Columns: {', '.join(df.columns[6:])}")  # Show weather columns
            successful += 1

        else:
            print(f"  ✗ No data available")
            failed += 1

        time.sleep(0.1)  # Small delay

    print("\n" + "=" * 80)
    print("CONVERTING TO PARQUET FORMAT")
    print("=" * 80)

    # Convert all CSV files to Parquet
    csv_files = list(meteostat_csv_dir.glob('*.csv'))
    print(f"\nConverting {len(csv_files)} CSV files to Parquet...")

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
            parquet_file = meteostat_parquet_dir / csv_file.with_suffix('.parquet').name
            df.to_parquet(parquet_file, compression='snappy', index=True)
            print(f"  ✓ {parquet_file.name}")
        except Exception as e:
            print(f"  ERROR converting {csv_file.name}: {e}")

    # Create combined dataset
    if csv_files:
        print("\n" + "=" * 80)
        print("CREATING COMBINED DATASET")
        print("=" * 80)

        all_data = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
            all_data.append(df)

        combined_df = pd.concat(all_data, ignore_index=False)
        combined_df = combined_df.sort_index()

        # Save combined CSV
        combined_csv = meteostat_dir / 'all_meteostat_station_data.csv'
        combined_df.to_csv(combined_csv)
        print(f"Saved combined CSV: {combined_csv} ({len(combined_df):,} records)")

        # Save combined Parquet
        combined_parquet = meteostat_dir / 'all_meteostat_station_data.parquet'
        combined_df.to_parquet(combined_parquet, compression='snappy')
        print(f"Saved combined Parquet: {combined_parquet}")

        # Create summary
        summary = {
            'total_locations': len(location_stations),
            'successful_downloads': successful,
            'failed_downloads': failed,
            'date_range': f"{start_date.date()} to {end_date.date()}",
            'total_records': len(combined_df),
            'data_columns': list(combined_df.columns[6:])  # Skip metadata columns
        }

        summary_file = meteostat_dir / 'download_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Saved summary: {summary_file}")

    # Summary
    print("\n" + "=" * 80)
    print("DOWNLOAD COMPLETE")
    print("=" * 80)
    print(f"\nSuccessful: {successful}/{len(location_stations)}")
    print(f"Failed: {failed}/{len(location_stations)}")
    print(f"\nData saved to:")
    print(f"  CSV files: {meteostat_csv_dir}")
    print(f"  Parquet files: {meteostat_parquet_dir}")
    print(f"  Station mapping: {station_mapping_file}")

    print("\n" + "=" * 80)
    print("WEATHER PARAMETERS INCLUDED")
    print("=" * 80)
    print("  tavg  - Average temperature (°C)")
    print("  tmin  - Minimum temperature (°C)")
    print("  tmax  - Maximum temperature (°C)")
    print("  prcp  - Precipitation (mm)")
    print("  snow  - Snow depth (mm)")
    print("  wdir  - Wind direction (degrees)")
    print("  wspd  - Wind speed (km/h)")
    print("  wpgt  - Peak wind gust (km/h)")
    print("  pres  - Sea-level air pressure (hPa)")
    print("  tsun  - Sunshine duration (minutes)")


if __name__ == '__main__':
    main()

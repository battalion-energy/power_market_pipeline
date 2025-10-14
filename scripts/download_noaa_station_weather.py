#!/usr/bin/env python3
"""
Download NOAA weather station data (ground-based observations) for Texas locations.

Uses NOAA GHCN-Daily (Global Historical Climatology Network - Daily) data.
This is actual weather station data, not satellite-derived like NASA POWER.

NOAA NCEI API Documentation: https://www.ncdc.noaa.gov/cdo-web/webservices/v2
"""

import pandas as pd
import requests
from datetime import datetime, timedelta
from pathlib import Path
import time
from typing import Dict, List, Tuple, Optional
import json
import math

# NOAA API Configuration
# Free API token - get yours at: https://www.ncdc.noaa.gov/cdo-web/token
NOAA_API_TOKEN = "your_token_here"  # You'll need to register for a free token
NOAA_BASE_URL = "https://www.ncdc.noaa.gov/cdo-web/api/v2"

# GHCN-Daily dataset ID
DATASET_ID = "GHCN"

# Data types to request (GHCN-Daily codes)
# See: https://www.ncei.noaa.gov/pub/data/ghcn/daily/readme.txt
DATA_TYPES = [
    'TMAX',  # Maximum temperature (tenths of degrees C)
    'TMIN',  # Minimum temperature (tenths of degrees C)
    'TAVG',  # Average temperature (tenths of degrees C)
    'PRCP',  # Precipitation (tenths of mm)
    'AWND',  # Average wind speed (tenths of m/s)
    'WSF2',  # Fastest 2-minute wind speed (tenths of m/s)
    'WSF5',  # Fastest 5-second wind speed (tenths of m/s)
    'WDF2',  # Direction of fastest 2-minute wind (degrees)
    'WDF5',  # Direction of fastest 5-second wind (degrees)
]


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points in kilometers using Haversine formula."""
    R = 6371  # Earth's radius in kilometers

    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))

    return R * c


def get_noaa_headers() -> Dict[str, str]:
    """Get headers for NOAA API requests."""
    return {
        'token': NOAA_API_TOKEN
    }


def find_nearby_stations(
    lat: float,
    lon: float,
    radius_km: float = 100,
    dataset: str = DATASET_ID
) -> List[Dict]:
    """
    Find weather stations near a location.

    Args:
        lat: Latitude
        lon: Longitude
        radius_km: Search radius in kilometers (max 100)
        dataset: NOAA dataset ID

    Returns:
        List of station dictionaries
    """
    # NOAA API uses extent parameter for geographic search
    # extent: west,south,east,north
    # Approximate 1 degree = 111 km at equator
    degree_buffer = radius_km / 111.0

    extent = f"{lon - degree_buffer},{lat - degree_buffer},{lon + degree_buffer},{lat + degree_buffer}"

    params = {
        'datasetid': dataset,
        'extent': extent,
        'limit': 100,  # Get up to 100 stations
        'sortfield': 'name',
        'sortorder': 'asc'
    }

    try:
        response = requests.get(
            f"{NOAA_BASE_URL}/stations",
            headers=get_noaa_headers(),
            params=params,
            timeout=30
        )
        response.raise_for_status()

        data = response.json()
        stations = data.get('results', [])

        # Calculate actual distances and add to station data
        for station in stations:
            if 'latitude' in station and 'longitude' in station:
                dist = haversine_distance(lat, lon, station['latitude'], station['longitude'])
                station['distance_km'] = dist

        # Sort by distance
        stations.sort(key=lambda x: x.get('distance_km', float('inf')))

        return stations

    except Exception as e:
        print(f"    ERROR finding stations: {e}")
        return []


def download_station_data(
    station_id: str,
    start_date: str,
    end_date: str,
    datatypes: List[str]
) -> Optional[pd.DataFrame]:
    """
    Download weather data from a specific station.

    Args:
        station_id: NOAA station ID
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        datatypes: List of data type codes

    Returns:
        DataFrame with weather data or None
    """
    params = {
        'datasetid': DATASET_ID,
        'stationid': station_id,
        'startdate': start_date,
        'enddate': end_date,
        'datatypeid': ','.join(datatypes),
        'limit': 1000,  # Max records per request
        'units': 'metric'
    }

    all_data = []

    try:
        # NOAA API limits to 1000 records per request
        # May need pagination for long time periods
        offset = 1

        while True:
            params['offset'] = offset

            response = requests.get(
                f"{NOAA_BASE_URL}/data",
                headers=get_noaa_headers(),
                params=params,
                timeout=60
            )
            response.raise_for_status()

            data = response.json()
            results = data.get('results', [])

            if not results:
                break

            all_data.extend(results)

            # Check if there are more results
            metadata = data.get('metadata', {})
            result_set = metadata.get('resultset', {})
            count = result_set.get('count', 0)
            limit = result_set.get('limit', 1000)

            if offset + limit > count:
                break

            offset += limit
            time.sleep(0.2)  # Rate limiting

        if not all_data:
            return None

        # Convert to DataFrame
        df = pd.DataFrame(all_data)

        # Pivot to wide format (one row per date, columns for each datatype)
        df['date'] = pd.to_datetime(df['date'])
        df_pivot = df.pivot_table(
            index='date',
            columns='datatype',
            values='value',
            aggfunc='first'
        )

        return df_pivot

    except Exception as e:
        print(f"    ERROR downloading data: {e}")
        return None


def main():
    """Main execution function."""
    # Check for API token
    if NOAA_API_TOKEN == "your_token_here":
        print("=" * 80)
        print("ERROR: NOAA API TOKEN REQUIRED")
        print("=" * 80)
        print("\nYou need a free NOAA API token to download weather station data.")
        print("\nSteps to get your token:")
        print("1. Visit: https://www.ncdc.noaa.gov/cdo-web/token")
        print("2. Enter your email address")
        print("3. Check your email for the token")
        print("4. Edit this script and replace 'your_token_here' with your token")
        print("\nThe token is FREE and arrives via email within minutes.")
        print("=" * 80)
        return

    # Setup directories
    project_root = Path(__file__).parent
    weather_dir = project_root / 'weather_data'

    noaa_dir = weather_dir / 'noaa_stations'
    noaa_dir.mkdir(exist_ok=True)

    noaa_csv_dir = noaa_dir / 'csv_files'
    noaa_csv_dir.mkdir(exist_ok=True)

    noaa_parquet_dir = noaa_dir / 'parquet_files'
    noaa_parquet_dir.mkdir(exist_ok=True)

    # Load locations
    locations_file = weather_dir / 'weather_locations.csv'
    locations_df = pd.read_csv(locations_file)

    # Date range: last 5 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)

    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    print(f"Downloading NOAA weather station data from {start_date.date()} to {end_date.date()}")
    print(f"Data types: {', '.join(DATA_TYPES)}\n")

    # Find stations for all locations first
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

        # Find nearby stations
        stations = find_nearby_stations(lat, lon, radius_km=50)

        if stations:
            # Take the 3 closest stations
            closest = stations[:3]
            print(f"  Found {len(stations)} stations within 50km")
            for i, station in enumerate(closest, 1):
                dist = station.get('distance_km', 0)
                station_name = station.get('name', 'Unknown')
                station_id = station.get('id', '')
                print(f"    {i}. {station_name} ({station_id}) - {dist:.1f} km")

            location_stations.append({
                'location_name': location_name,
                'lat': lat,
                'lon': lon,
                'stations': closest
            })
        else:
            print(f"  WARNING: No stations found within 50km")

        time.sleep(0.2)  # Rate limiting

    # Save station mapping
    station_mapping_file = noaa_dir / 'station_mapping.json'
    with open(station_mapping_file, 'w') as f:
        json.dump(location_stations, f, indent=2, default=str)
    print(f"\nSaved station mapping to {station_mapping_file}")

    # Download data from stations
    print("\n" + "=" * 80)
    print("DOWNLOADING WEATHER STATION DATA")
    print("=" * 80)

    successful = 0
    failed = 0

    for loc_data in location_stations:
        location_name = loc_data['location_name']
        stations = loc_data['stations']

        if not stations:
            failed += 1
            continue

        print(f"\n[{successful + failed + 1}/{len(location_stations)}] {location_name}")

        # Try the closest station first
        station = stations[0]
        station_id = station['id']
        station_name = station['name']
        dist = station.get('distance_km', 0)

        print(f"  Using: {station_name} ({dist:.1f} km)")

        df = download_station_data(
            station_id=station_id,
            start_date=start_str,
            end_date=end_str,
            datatypes=DATA_TYPES
        )

        if df is not None and len(df) > 0:
            # Add metadata
            df.insert(0, 'location_name', location_name)
            df.insert(1, 'station_id', station_id)
            df.insert(2, 'station_name', station_name)
            df.insert(3, 'distance_km', dist)

            # Save to CSV
            safe_name = location_name.replace(' ', '_').replace('/', '_')
            csv_file = noaa_csv_dir / f"{safe_name}_{station_id}.csv"
            df.to_csv(csv_file)

            print(f"  ✓ Downloaded {len(df)} days of data")
            successful += 1

            time.sleep(0.2)  # Rate limiting
        else:
            print(f"  ✗ No data available")
            failed += 1

    print("\n" + "=" * 80)
    print("CONVERTING TO PARQUET FORMAT")
    print("=" * 80)

    # Convert all CSV files to Parquet
    csv_files = list(noaa_csv_dir.glob('*.csv'))
    print(f"\nConverting {len(csv_files)} CSV files to Parquet...")

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, index_col='date', parse_dates=True)
            parquet_file = noaa_parquet_dir / csv_file.with_suffix('.parquet').name
            df.to_parquet(parquet_file, compression='snappy', index=True)
            print(f"  ✓ {parquet_file.name}")
        except Exception as e:
            print(f"  ERROR converting {csv_file.name}: {e}")

    # Create combined dataset if we have data
    if csv_files:
        print("\n" + "=" * 80)
        print("CREATING COMBINED DATASET")
        print("=" * 80)

        all_data = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file, index_col='date', parse_dates=True)
            all_data.append(df)

        combined_df = pd.concat(all_data, ignore_index=False)
        combined_df = combined_df.sort_values(['location_name', combined_df.index.name])

        # Save combined CSV
        combined_csv = noaa_dir / 'all_noaa_station_data.csv'
        combined_df.to_csv(combined_csv)
        print(f"Saved combined CSV: {combined_csv} ({len(combined_df):,} records)")

        # Save combined Parquet
        combined_parquet = noaa_dir / 'all_noaa_station_data.parquet'
        combined_df.to_parquet(combined_parquet, compression='snappy')
        print(f"Saved combined Parquet: {combined_parquet}")

    # Summary
    print("\n" + "=" * 80)
    print("DOWNLOAD COMPLETE")
    print("=" * 80)
    print(f"\nSuccessful: {successful}/{len(location_stations)}")
    print(f"Failed: {failed}/{len(location_stations)}")
    print(f"\nData saved to:")
    print(f"  CSV files: {noaa_csv_dir}")
    print(f"  Parquet files: {noaa_parquet_dir}")
    print(f"  Station mapping: {station_mapping_file}")


if __name__ == '__main__':
    main()

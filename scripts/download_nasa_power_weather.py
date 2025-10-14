#!/usr/bin/env python3
"""
Download NASA POWER satellite weather data for Texas renewable energy sites and major cities.

NASA POWER API Documentation: https://power.larc.nasa.gov/docs/services/api/
"""

import os
import pandas as pd
import requests
from datetime import datetime, timedelta
from pathlib import Path
import time
from typing import Dict, List, Tuple
import json
from dotenv import load_dotenv

# Major Texas cities with coordinates
TEXAS_CITIES = {
    'Houston': (29.7604, -95.3698),
    'San_Antonio': (29.4241, -98.4936),
    'Dallas': (32.7767, -96.7970),
    'Austin': (30.2672, -97.7431),
    'Fort_Worth': (32.7555, -97.3308),
    'El_Paso': (31.7619, -106.4850),
    'Arlington': (32.7357, -97.1081),
    'Corpus_Christi': (27.8006, -97.3964),
    'Plano': (33.0198, -96.6989),
    'Lubbock': (33.5779, -101.8552)
}

# NASA POWER API parameters
NASA_POWER_BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

# Weather parameters to download
# See: https://power.larc.nasa.gov/docs/services/api/v1/temporal/daily/
WEATHER_PARAMS = [
    'T2M',          # Temperature at 2 Meters (°C)
    'T2M_MAX',      # Maximum Temperature at 2 Meters (°C)
    'T2M_MIN',      # Minimum Temperature at 2 Meters (°C)
    'WS10M',        # Wind Speed at 10 Meters (m/s)
    'WS50M',        # Wind Speed at 50 Meters (m/s) - important for wind farms
    'WD10M',        # Wind Direction at 10 Meters (Degrees)
    'WD50M',        # Wind Direction at 50 Meters (Degrees)
    'ALLSKY_SFC_SW_DWN',  # All Sky Surface Shortwave Downward Irradiance (kW-hr/m^2/day)
    'CLRSKY_SFC_SW_DWN',  # Clear Sky Surface Shortwave Downward Irradiance (kW-hr/m^2/day)
    'PRECTOTCORR',  # Precipitation Corrected (mm/day)
    'RH2M',         # Relative Humidity at 2 Meters (%)
    'PS',           # Surface Pressure (kPa)
]


def extract_renewable_locations(csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Extract unique wind and solar farm locations from generator file."""
    df = pd.read_csv(csv_path)

    # Filter for WIND and SOLAR, and ensure we have valid coordinates
    wind_farms = df[
        (df['Technology_Type'] == 'WIND') &
        (df['Latitude'].notna()) &
        (df['Longitude'].notna())
    ].copy()

    solar_farms = df[
        (df['Technology_Type'] == 'SOLAR') &
        (df['Latitude'].notna()) &
        (df['Longitude'].notna())
    ].copy()

    # Group by unique locations (lat/lon) and aggregate capacity
    def aggregate_by_location(df):
        return df.groupby(['Latitude', 'Longitude']).agg({
            'Decoded_Name': 'first',
            'EIA_Capacity_MW': 'sum',
            'Resource_Name': 'first'
        }).reset_index()

    wind_unique = aggregate_by_location(wind_farms)
    solar_unique = aggregate_by_location(solar_farms)

    print(f"Found {len(wind_unique)} unique wind farm locations")
    print(f"Found {len(solar_unique)} unique solar farm locations")

    # Sort by capacity and get top sites
    wind_unique = wind_unique.sort_values('EIA_Capacity_MW', ascending=False)
    solar_unique = solar_unique.sort_values('EIA_Capacity_MW', ascending=False)

    return wind_unique, solar_unique


def download_nasa_power_data(
    lat: float,
    lon: float,
    location_name: str,
    start_date: str,
    end_date: str,
    output_dir: Path
) -> bool:
    """
    Download NASA POWER weather data for a specific location.

    Args:
        lat: Latitude
        lon: Longitude
        location_name: Name of the location
        start_date: Start date in YYYYMMDD format
        end_date: End date in YYYYMMDD format
        output_dir: Directory to save CSV files

    Returns:
        True if successful, False otherwise
    """
    params = {
        'parameters': ','.join(WEATHER_PARAMS),
        'community': 'RE',  # Renewable Energy community
        'longitude': lon,
        'latitude': lat,
        'start': start_date,
        'end': end_date,
        'format': 'JSON'
    }

    try:
        print(f"  Downloading data for {location_name} ({lat:.4f}, {lon:.4f})...")
        response = requests.get(NASA_POWER_BASE_URL, params=params, timeout=60)
        response.raise_for_status()

        data = response.json()

        if 'properties' not in data or 'parameter' not in data['properties']:
            print(f"  WARNING: Unexpected response format for {location_name}")
            return False

        # Convert to DataFrame
        weather_data = data['properties']['parameter']
        df = pd.DataFrame(weather_data)

        # Convert index to datetime
        df.index = pd.to_datetime(df.index, format='%Y%m%d')
        df.index.name = 'date'

        # Add location metadata
        df.insert(0, 'location_name', location_name)
        df.insert(1, 'latitude', lat)
        df.insert(2, 'longitude', lon)

        # Save to CSV
        safe_name = location_name.replace(' ', '_').replace('/', '_')
        csv_file = output_dir / f"{safe_name}_{lat:.4f}_{lon:.4f}.csv"
        df.to_csv(csv_file)

        print(f"  ✓ Saved {len(df)} days of data to {csv_file.name}")
        return True

    except requests.exceptions.RequestException as e:
        print(f"  ERROR downloading {location_name}: {e}")
        return False
    except Exception as e:
        print(f"  ERROR processing {location_name}: {e}")
        return False


def main():
    """Main execution function."""
    # Load environment variables
    load_dotenv()
    # Setup directories
    project_root = Path(__file__).parent
    generators_file = project_root / 'ERCOT_GENERATORS_LOCATIONS_VALIDATED.csv'

    # Weather data directory from environment
    weather_dir = Path(os.getenv('WEATHER_DATA_DIR', '/pool/ssd8tb/data/weather_data'))
    weather_dir.mkdir(exist_ok=True, parents=True)

    csv_dir = weather_dir / 'csv_files'
    csv_dir.mkdir(exist_ok=True)

    parquet_dir = weather_dir / 'parquet_files'
    parquet_dir.mkdir(exist_ok=True)

    # Date range: from 2019-01-01 to present
    end_date = datetime.now()
    start_date = datetime(2019, 1, 1)

    start_str = start_date.strftime('%Y%m%d')
    end_str = end_date.strftime('%Y%m%d')

    print(f"Downloading weather data from {start_date.date()} to {end_date.date()}")
    print(f"Weather parameters: {', '.join(WEATHER_PARAMS)}\n")

    # Extract renewable energy locations
    print("=" * 80)
    print("EXTRACTING RENEWABLE ENERGY LOCATIONS")
    print("=" * 80)

    wind_farms, solar_farms = extract_renewable_locations(str(generators_file))

    # Select top wind and solar farms by capacity
    TOP_N = 30  # Top 30 wind and solar farms each
    top_wind = wind_farms.head(TOP_N)
    top_solar = solar_farms.head(TOP_N)

    print(f"\nSelected top {len(top_wind)} wind farms (total capacity: {top_wind['EIA_Capacity_MW'].sum():.1f} MW)")
    print(f"Selected top {len(top_solar)} solar farms (total capacity: {top_solar['EIA_Capacity_MW'].sum():.1f} MW)")

    # Compile all locations
    locations = []

    # Add wind farms
    for _, row in top_wind.iterrows():
        locations.append({
            'name': f"WIND_{row['Decoded_Name']}",
            'lat': row['Latitude'],
            'lon': row['Longitude'],
            'type': 'wind_farm',
            'capacity_mw': row['EIA_Capacity_MW']
        })

    # Add solar farms
    for _, row in top_solar.iterrows():
        locations.append({
            'name': f"SOLAR_{row['Decoded_Name']}",
            'lat': row['Latitude'],
            'lon': row['Longitude'],
            'type': 'solar_farm',
            'capacity_mw': row['EIA_Capacity_MW']
        })

    # Add Texas cities
    for city, (lat, lon) in TEXAS_CITIES.items():
        locations.append({
            'name': f"CITY_{city}",
            'lat': lat,
            'lon': lon,
            'type': 'city',
            'capacity_mw': None
        })

    print(f"\nTotal locations to download: {len(locations)}")

    # Save locations manifest
    locations_df = pd.DataFrame(locations)
    locations_file = weather_dir / 'weather_locations.csv'
    locations_df.to_csv(locations_file, index=False)
    print(f"Saved locations manifest to {locations_file}")

    # Download weather data
    print("\n" + "=" * 80)
    print("DOWNLOADING NASA POWER WEATHER DATA")
    print("=" * 80)

    successful = 0
    failed = 0

    for i, loc in enumerate(locations, 1):
        print(f"\n[{i}/{len(locations)}] {loc['name']}")

        success = download_nasa_power_data(
            lat=loc['lat'],
            lon=loc['lon'],
            location_name=loc['name'],
            start_date=start_str,
            end_date=end_str,
            output_dir=csv_dir
        )

        if success:
            successful += 1
        else:
            failed += 1

        # Be nice to NASA servers - rate limit
        if i < len(locations):
            time.sleep(2)  # 2 second delay between requests

    print("\n" + "=" * 80)
    print("CONVERTING TO PARQUET FORMAT")
    print("=" * 80)

    # Convert all CSV files to Parquet
    csv_files = list(csv_dir.glob('*.csv'))
    print(f"\nConverting {len(csv_files)} CSV files to Parquet...")

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, index_col='date', parse_dates=True)
            parquet_file = parquet_dir / csv_file.with_suffix('.parquet').name
            df.to_parquet(parquet_file, compression='snappy', index=True)
            print(f"  ✓ {parquet_file.name}")
        except Exception as e:
            print(f"  ERROR converting {csv_file.name}: {e}")

    # Create combined datasets
    print("\n" + "=" * 80)
    print("CREATING COMBINED DATASETS")
    print("=" * 80)

    all_data = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file, index_col='date', parse_dates=True)
        all_data.append(df)

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=False)
        combined_df = combined_df.sort_values(['location_name', 'date'])

        # Save combined CSV
        combined_csv = weather_dir / 'all_weather_data.csv'
        combined_df.to_csv(combined_csv)
        print(f"Saved combined CSV: {combined_csv} ({len(combined_df):,} records)")

        # Save combined Parquet
        combined_parquet = weather_dir / 'all_weather_data.parquet'
        combined_df.to_parquet(combined_parquet, compression='snappy')
        print(f"Saved combined Parquet: {combined_parquet}")

        # Create summary statistics
        summary = {
            'total_locations': len(locations),
            'wind_farms': len(top_wind),
            'solar_farms': len(top_solar),
            'cities': len(TEXAS_CITIES),
            'successful_downloads': successful,
            'failed_downloads': failed,
            'date_range': f"{start_date.date()} to {end_date.date()}",
            'total_records': len(combined_df),
            'parameters': WEATHER_PARAMS
        }

        summary_file = weather_dir / 'download_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Saved summary: {summary_file}")

    print("\n" + "=" * 80)
    print("DOWNLOAD COMPLETE")
    print("=" * 80)
    print(f"\nSuccessful: {successful}/{len(locations)}")
    print(f"Failed: {failed}/{len(locations)}")
    print(f"\nData saved to:")
    print(f"  CSV files: {csv_dir}")
    print(f"  Parquet files: {parquet_dir}")
    print(f"  Combined data: {weather_dir}")


if __name__ == '__main__':
    main()

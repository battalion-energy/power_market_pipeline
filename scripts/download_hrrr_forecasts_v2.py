#!/usr/bin/env python3
"""
Download NOAA HRRR (High Resolution Rapid Refresh) historical weather forecasts.
Uses Herbie package for efficient GRIB2 subset downloading.

HRRR provides:
- 3km resolution (excellent for Texas)
- Hourly forecast cycles
- Multiple forecast horizons
- Historical archive back to 2014
- Free on AWS S3

This gives us VINTAGE FORECASTS - what was actually predicted at the time,
not what actually happened. Critical for realistic energy trading analysis.

AWS Archive: s3://noaa-hrrr-bdp-pds/
Documentation: https://registry.opendata.aws/noaa-hrrr-pds/
Herbie Package: https://github.com/blaylockbk/Herbie
"""

import os
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from pathlib import Path
import time
from typing import Optional, List
import json
from herbie import Herbie
from dotenv import load_dotenv

# Forecast horizons to download (hours ahead)
FORECAST_HORIZONS = [1, 3, 6, 12, 24]  # 1-hr, 3-hr, 6-hr, 12-hr, day-ahead

# Variables to extract from HRRR
HRRR_SEARCH_STRINGS = [
    ':TMP:2 m',          # Temperature at 2m
    ':UGRD:10 m',        # U-wind component at 10m
    ':VGRD:10 m',        # V-wind component at 10m
    ':DSWRF:surface',    # Downward shortwave radiation (solar)
]


def download_hrrr_point_forecast(
    date: datetime,
    forecast_hour: int,
    forecast_horizon: int,
    lat: float,
    lon: float,
    location_name: str
) -> Optional[pd.DataFrame]:
    """
    Download HRRR forecast for a specific time and location using Herbie.

    Args:
        date: Forecast initialization date/time
        forecast_hour: Hour of day for forecast (0-23)
        forecast_horizon: Hours ahead (1-48)
        lat: Latitude
        lon: Longitude (negative for west)
        location_name: Name of location

    Returns:
        DataFrame with forecast data or None
    """
    try:
        # Create Herbie object for this forecast
        forecast_time = date.replace(hour=forecast_hour)

        H = Herbie(
            forecast_time,
            model='hrrr',
            product='sfc',  # Surface file
            fxx=forecast_horizon  # Forecast hour
        )

        # Download and read as xarray Dataset for the nearest point
        # Use Herbie's .xarray() method which efficiently subsets
        # Pass search strings as a single regex pattern
        search_pattern = '|'.join(HRRR_SEARCH_STRINGS)
        ds = H.xarray(search_pattern, remove_grib=True)

        # Handle case where cfgrib returns list of datasets (multiple hypercubes)
        # Extract data from each dataset separately to avoid coordinate conflicts
        ds_list = ds if isinstance(ds, list) else [ds]

        # Extract data
        data = {
            'forecast_time': forecast_time,
            'forecast_horizon_hours': forecast_horizon,
            'valid_time': forecast_time + timedelta(hours=forecast_horizon),
            'location_name': location_name,
        }

        # Extract from each dataset in the list
        for ds_item in ds_list:
            # Find nearest grid point for this dataset
            ds_point = ds_item.sel(latitude=lat, longitude=lon, method='nearest')

            # Store lat/lon from first dataset
            if 'latitude' not in data:
                data['latitude'] = float(ds_point.latitude.values)
                data['longitude'] = float(ds_point.longitude.values)

            # Extract variables (check what's in this dataset)
            if 't2m' in ds_point:
                data['temperature_2m_K'] = float(ds_point['t2m'].values)
                data['temperature_2m_C'] = data['temperature_2m_K'] - 273.15
            elif 't' in ds_point:
                data['temperature_2m_K'] = float(ds_point['t'].values)
                data['temperature_2m_C'] = data['temperature_2m_K'] - 273.15

            if 'u10' in ds_point:
                data['u_wind_10m'] = float(ds_point['u10'].values)
            if 'v10' in ds_point:
                data['v_wind_10m'] = float(ds_point['v10'].values)
            if 'u' in ds_point and 'u_wind_10m' not in data:
                data['u_wind_10m'] = float(ds_point['u'].values)
            if 'v' in ds_point and 'v_wind_10m' not in data:
                data['v_wind_10m'] = float(ds_point['v'].values)

            if 'dswrf' in ds_point:
                data['solar_radiation_Wm2'] = float(ds_point['dswrf'].values)

        # Calculate derived wind variables if we have components
        if 'u_wind_10m' in data and 'v_wind_10m' in data:
            u = data['u_wind_10m']
            v = data['v_wind_10m']
            data['wind_speed_10m'] = np.sqrt(u**2 + v**2)
            data['wind_direction_10m'] = (np.degrees(np.arctan2(u, v)) + 180) % 360

        return pd.DataFrame([data])

    except Exception as e:
        print(f"    ERROR: {e}")
        return None


def main():
    """Main execution function."""
    print("=" * 80)
    print("NOAA HRRR HISTORICAL FORECAST DOWNLOADER (Herbie)")
    print("=" * 80)
    print("Downloading vintage weather forecasts for energy trading analysis")
    print("Resolution: 3km | Frequency: Hourly | Source: NOAA/AWS")
    print()

    # Load environment variables and setup directories
    load_dotenv()
    project_root = Path(__file__).parent
    weather_dir = Path(os.getenv('WEATHER_DATA_DIR', '/pool/ssd8tb/data/weather_data'))

    hrrr_dir = weather_dir / 'hrrr_forecasts'
    hrrr_dir.mkdir(exist_ok=True, parents=True)

    csv_dir = hrrr_dir / 'csv_files'
    csv_dir.mkdir(exist_ok=True)

    parquet_dir = hrrr_dir / 'parquet_files'
    parquet_dir.mkdir(exist_ok=True)

    # Load locations
    locations_file = weather_dir / 'weather_locations.csv'
    if not locations_file.exists():
        print(f"ERROR: Locations file not found: {locations_file}")
        return

    locations_df = pd.read_csv(locations_file)

    # Date range - BESS analysis period
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2024, 12, 31)

    print(f"Date range: {start_date.date()} to {end_date.date()}")
    print(f"Forecast horizons: {FORECAST_HORIZONS} hours")
    print(f"Forecast cycles per day: 4 (00, 06, 12, 18 UTC)")
    print()

    # Test with Houston first
    print("=" * 80)
    print("PHASE 1: TESTING WITH SAMPLE DATA")
    print("=" * 80)
    print("Downloading 1 day of forecasts for Houston to test...")
    print()

    test_location = locations_df[locations_df['name'] == 'CITY_Houston'].iloc[0]
    test_date = datetime(2022, 7, 15)  # Summer day

    print(f"Test location: {test_location['name']}")
    print(f"  Coordinates: ({test_location['lat']:.4f}, {test_location['lon']:.4f})")
    print(f"Test date: {test_date.date()}")
    print()

    # Download forecasts for one day, all horizons
    all_forecasts = []

    for cycle_hour in [0, 6, 12, 18]:  # 4 forecast cycles per day
        for horizon in FORECAST_HORIZONS:
            print(f"  {test_date.date()} {cycle_hour:02d}Z + {horizon}h... ", end='', flush=True)

            df = download_hrrr_point_forecast(
                date=test_date,
                forecast_hour=cycle_hour,
                forecast_horizon=horizon,
                lat=test_location['lat'],
                lon=test_location['lon'],
                location_name=test_location['name']
            )

            if df is not None:
                all_forecasts.append(df)
                print("✓")
            else:
                print("✗")

            time.sleep(1)  # Be nice to the service

    if all_forecasts:
        # Combine and save test data
        test_df = pd.concat(all_forecasts, ignore_index=True)
        test_file = hrrr_dir / 'test_houston_forecast_herbie.csv'
        test_df.to_csv(test_file, index=False)

        # Also save as parquet
        test_parquet = hrrr_dir / 'test_houston_forecast_herbie.parquet'
        test_df.to_parquet(test_parquet, compression='snappy')

        print(f"\n✓ Test successful! Downloaded {len(test_df)} forecast records")
        print(f"  CSV: {test_file}")
        print(f"  Parquet: {test_parquet}")
        print(f"\nSample data:")
        print(test_df[['forecast_time', 'forecast_horizon_hours', 'valid_time',
                       'temperature_2m_C', 'wind_speed_10m', 'solar_radiation_Wm2']].head(10))

        print("\n" + "=" * 80)
        print("PHASE 2: FULL DOWNLOAD PLAN")
        print("=" * 80)
        print("Test successful! Ready to download full dataset.")
        print()
        print("Recommended strategy:")
        print("  1. Download by month for better organization")
        print("  2. Focus on main forecast cycles (00Z, 12Z) initially")
        print("  3. Select priority locations (cities + largest farms)")
        print("  4. Use 1-hr and 12-hr horizons (intraday + day-ahead)")
        print()

        # Estimate size
        records_per_location_day = len(FORECAST_HORIZONS) * 4  # horizons × cycles
        days = (end_date - start_date).days
        total_records = records_per_location_day * days * len(locations_df)

        print(f"Estimated full dataset size:")
        print(f"  - Days: {days}")
        print(f"  - Locations: {len(locations_df)}")
        print(f"  - Total records: {total_records:,}")
        print(f"  - Estimated time: {total_records * 1.5 / 3600:.1f} hours")
        print(f"  - Estimated storage: {total_records * 200 / 1e9:.1f} GB")
        print()
        print("To proceed with full download, uncomment the full download section in the script.")

        # Save plan
        plan = {
            'test_completed': True,
            'test_date': str(test_date.date()),
            'test_location': test_location['name'],
            'records_downloaded': len(test_df),
            'variables_extracted': list(test_df.columns),
            'full_dataset_estimate': {
                'total_records': total_records,
                'estimated_hours': round(total_records * 1.5 / 3600, 1),
                'estimated_gb': round(total_records * 200 / 1e9, 1)
            }
        }

        plan_file = hrrr_dir / 'download_plan_herbie.json'
        with open(plan_file, 'w') as f:
            json.dump(plan, f, indent=2, default=str)

        print(f"Download plan saved to: {plan_file}")

    else:
        print("\n✗ Test failed - no data downloaded")
        print("Possible issues:")
        print("  - Date outside HRRR archive (2014-07-30 to present)")
        print("  - AWS S3 access issues")
        print("  - Herbie configuration issues")


if __name__ == '__main__':
    main()

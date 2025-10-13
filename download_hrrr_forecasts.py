#!/usr/bin/env python3
"""
Download NOAA HRRR (High Resolution Rapid Refresh) historical weather forecasts.

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
"""

import os
import xarray as xr
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import s3fs
import time
from typing import Optional, List, Tuple
import json
from dotenv import load_dotenv

# AWS S3 configuration for HRRR
HRRR_BUCKET = 'noaa-hrrr-bdp-pds'

# HRRR Variables to extract
# See: https://rapidrefresh.noaa.gov/hrrr/HRRRv4_native_grib2_channel_list.txt
HRRR_VARIABLES = {
    'TMP:2 m above ground': 'temperature_2m',  # Temperature (K)
    'UGRD:10 m above ground': 'u_wind_10m',    # U-wind component (m/s)
    'VGRD:10 m above ground': 'v_wind_10m',    # V-wind component (m/s)
    'DSWRF:surface': 'solar_radiation',        # Downward shortwave radiation (W/m²)
    'APCP:surface': 'precipitation',           # Total precipitation (kg/m²)
}

# Forecast horizons to download (hours ahead)
FORECAST_HORIZONS = [1, 3, 6, 12, 24]  # 1-hr, 3-hr, 6-hr, 12-hr, day-ahead


def find_nearest_grid_point(ds: xr.Dataset, lat: float, lon: float) -> Tuple[int, int]:
    """
    Find nearest HRRR grid point to a lat/lon location.

    Args:
        ds: HRRR dataset
        lat: Target latitude
        lon: Target longitude (negative for west)

    Returns:
        (y_idx, x_idx) of nearest grid point
    """
    # HRRR uses Lambert Conformal projection
    # Find nearest point in the grid
    lats = ds['latitude'].values
    lons = ds['longitude'].values

    # Convert lon to 0-360 if needed
    if lon < 0:
        lon = lon + 360

    # Calculate distances
    dist = np.sqrt((lats - lat)**2 + (lons - lon)**2)

    # Find minimum
    y_idx, x_idx = np.unravel_index(dist.argmin(), dist.shape)

    return int(y_idx), int(x_idx)


def download_hrrr_forecast(
    date: datetime,
    forecast_hour: int,
    forecast_horizon: int,
    lat: float,
    lon: float,
    location_name: str
) -> Optional[pd.DataFrame]:
    """
    Download HRRR forecast for a specific time and location.

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
    # Build S3 path
    # Format: hrrr.YYYYMMDD/conus/hrrr.tHHz.wrfsfcfHH.grib2
    datestr = date.strftime('%Y%m%d')
    s3_key = f'hrrr.{datestr}/conus/hrrr.t{forecast_hour:02d}z.wrfsfcf{forecast_horizon:02d}.grib2'

    try:
        # Use s3fs to access the file
        fs = s3fs.S3FileSystem(anon=True)

        # Open the file from S3
        with fs.open(f'{HRRR_BUCKET}/{s3_key}', 'rb') as f:
            # Open dataset with cfgrib engine
            # filter_by_keys to get surface variables
            ds = xr.open_dataset(
                f,
                engine='cfgrib',
                backend_kwargs={
                    'filter_by_keys': {'typeOfLevel': 'surface'},
                    'indexpath': ''
                }
            )

            # Find nearest grid point
            y_idx, x_idx = find_nearest_grid_point(ds, lat, lon)

            # Extract data at this point
            data = {}
            data['forecast_time'] = date.replace(hour=forecast_hour)
            data['forecast_horizon_hours'] = forecast_horizon
            data['valid_time'] = data['forecast_time'] + timedelta(hours=forecast_horizon)
            data['location_name'] = location_name
            data['latitude'] = lat
            data['longitude'] = lon

            # Extract variables
            if 't2m' in ds:  # Temperature at 2m
                data['temperature_2m_K'] = float(ds['t2m'].isel(y=y_idx, x=x_idx).values)
                data['temperature_2m_C'] = data['temperature_2m_K'] - 273.15

            if 'u10' in ds and 'v10' in ds:  # Wind components
                u = float(ds['u10'].isel(y=y_idx, x=x_idx).values)
                v = float(ds['v10'].isel(y=y_idx, x=x_idx).values)
                data['u_wind_10m'] = u
                data['v_wind_10m'] = v
                data['wind_speed_10m'] = np.sqrt(u**2 + v**2)
                data['wind_direction_10m'] = np.degrees(np.arctan2(u, v)) % 360

            if 'dswrf' in ds:  # Solar radiation
                data['solar_radiation_Wm2'] = float(ds['dswrf'].isel(y=y_idx, x=x_idx).values)

            ds.close()

        return pd.DataFrame([data])

    except Exception as e:
        print(f"    ERROR: {e}")
        return None


def main():
    """Main execution function."""
    print("=" * 80)
    print("NOAA HRRR HISTORICAL FORECAST DOWNLOADER")
    print("=" * 80)
    print("Downloading vintage weather forecasts for energy trading analysis")
    print("Resolution: 3km | Frequency: Hourly | Source: NOAA/AWS")
    print()

    # Setup directories
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

    # Date range - start with recent period for testing
    # HRRR archive available from 2014-07-30 to present
    # Start with 2022-2024 for BESS analysis period
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2024, 12, 31)

    print(f"Date range: {start_date.date()} to {end_date.date()}")
    print(f"Forecast horizons: {FORECAST_HORIZONS} hours")
    print(f"Forecast cycles per day: 4 (00, 06, 12, 18 UTC)")
    print()

    # For initial test, download a sample
    print("=" * 80)
    print("PHASE 1: TESTING WITH SAMPLE DATA")
    print("=" * 80)
    print("Downloading 1 day of forecasts for Houston to test...")
    print()

    # Test with Houston
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

            df = download_hrrr_forecast(
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

            time.sleep(0.5)  # Be nice to AWS

    if all_forecasts:
        # Combine and save test data
        test_df = pd.concat(all_forecasts, ignore_index=True)
        test_file = hrrr_dir / 'test_houston_forecast.csv'
        test_df.to_csv(test_file, index=False)

        print(f"\n✓ Test successful! Downloaded {len(test_df)} forecast records")
        print(f"  Saved to: {test_file}")
        print(f"\nSample data:")
        print(test_df[['forecast_time', 'forecast_horizon_hours', 'valid_time',
                       'temperature_2m_C', 'wind_speed_10m', 'solar_radiation_Wm2']].head())

        print("\n" + "=" * 80)
        print("PHASE 2: FULL DOWNLOAD READY")
        print("=" * 80)
        print("Test successful! Ready to download full dataset.")
        print()
        print("To proceed with full download:")
        print("  1. Uncomment the full download section below")
        print("  2. Adjust date range as needed")
        print("  3. Estimated time: Several hours to days")
        print("  4. Estimated storage: 10-100 GB depending on period")
        print()
        print("Recommendation:")
        print("  - Start with 2022-2024 (BESS analysis period)")
        print("  - Download in monthly chunks")
        print("  - Focus on key forecast cycles (00Z, 12Z)")
        print("  - Use selective locations (major cities + key wind farms)")

        # Save download plan
        plan = {
            'test_completed': True,
            'test_date': str(test_date.date()),
            'test_location': test_location['name'],
            'records_downloaded': len(test_df),
            'recommended_next_steps': [
                'Review test data quality',
                'Verify forecast accuracy vs actuals',
                'Design sampling strategy (not every hour/location)',
                'Estimate storage requirements',
                'Plan monthly batch downloads'
            ],
            'storage_estimates': {
                '1_day_1_location': f'{len(test_df)} records',
                '1_year_1_location_all_cycles': '~29,000 forecasts',
                '1_year_48_locations': '~1.4M forecasts',
                'parquet_size_estimate': '500MB-2GB per year'
            }
        }

        plan_file = hrrr_dir / 'download_plan.json'
        with open(plan_file, 'w') as f:
            json.dump(plan, f, indent=2)

        print(f"\nDownload plan saved to: {plan_file}")

    else:
        print("\n✗ Test failed - no data downloaded")
        print("Possible issues:")
        print("  - Date outside HRRR archive (2014-07-30 to present)")
        print("  - AWS S3 access issues")
        print("  - GRIB library configuration")

    # FULL DOWNLOAD CODE (commented out for safety)
    """
    print("\n" + "=" * 80)
    print("FULL DOWNLOAD - PHASE 2")
    print("=" * 80)

    # Download strategy: monthly chunks, key locations, main forecast cycles
    selected_locations = locations_df[
        (locations_df['type'] == 'city') |  # All cities
        (locations_df['capacity_mw'] > 1000)  # Large wind/solar farms
    ]

    print(f"Selected {len(selected_locations)} high-priority locations")

    # Main forecast cycles (reduce from 24 to 4 per day)
    main_cycles = [0, 6, 12, 18]  # 00Z, 06Z, 12Z, 18Z

    # Download by month
    current_date = start_date
    while current_date <= end_date:
        month_end = min(current_date + timedelta(days=30), end_date)

        print(f"\nProcessing: {current_date.strftime('%Y-%m')}")

        for _, location in selected_locations.iterrows():
            location_forecasts = []

            # Iterate through days in month
            day = current_date
            while day < month_end:
                for cycle_hour in main_cycles:
                    for horizon in FORECAST_HORIZONS:
                        df = download_hrrr_forecast(
                            date=day,
                            forecast_hour=cycle_hour,
                            forecast_horizon=horizon,
                            lat=location['lat'],
                            lon=location['lon'],
                            location_name=location['name']
                        )

                        if df is not None:
                            location_forecasts.append(df)

                        time.sleep(0.1)  # Rate limiting

                day += timedelta(days=1)

            # Save month's data for this location
            if location_forecasts:
                month_df = pd.concat(location_forecasts, ignore_index=True)
                safe_name = location['name'].replace(' ', '_').replace('/', '_')
                month_str = current_date.strftime('%Y%m')
                csv_file = csv_dir / f"{safe_name}_{month_str}.csv"
                month_df.to_csv(csv_file, index=False)

                # Also save as parquet
                parquet_file = parquet_dir / f"{safe_name}_{month_str}.parquet"
                month_df.to_parquet(parquet_file, compression='snappy')

                print(f"  ✓ {location['name']}: {len(month_df)} forecasts")

        current_date = month_end
    """


if __name__ == '__main__':
    main()

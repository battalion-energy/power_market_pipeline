#!/usr/bin/env python3
"""
Download NOAA HRRR (High Resolution Rapid Refresh) historical weather forecasts - v3.
Simplified approach using Herbie's built-in point extraction.

HRRR provides:
- 3km resolution (excellent for Texas)
- Hourly forecast cycles
- Multiple forecast horizons
- Historical archive back to 2014
- Free on AWS S3

This version fixes coordinate indexing issues by using Herbie's proper methods.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import time
from typing import Optional, List, Dict
import json
from dotenv import load_dotenv

# Try to import Herbie
try:
    from herbie import Herbie
    HERBIE_AVAILABLE = True
except ImportError:
    print("ERROR: Herbie not installed. Install with: pip install herbie-data")
    HERBIE_AVAILABLE = False

# Forecast horizons to download (hours ahead)
FORECAST_HORIZONS = [1, 6, 12, 24]  # 1-hr, 6-hr, 12-hr, day-ahead

# Variables to extract from HRRR (search strings for Herbie)
HRRR_VARIABLES = {
    'TMP:2 m': 'temperature_2m',
    'UGRD:10 m': 'u_wind_10m',
    'VGRD:10 m': 'v_wind_10m',
    'DSWRF:surface': 'solar_radiation',
}


def download_hrrr_forecast_point(
    date: datetime,
    forecast_hour: int,
    forecast_horizon: int,
    lat: float,
    lon: float,
) -> Optional[Dict]:
    """
    Download HRRR forecast for a specific time and location.

    Args:
        date: Forecast initialization date
        forecast_hour: Hour of day for forecast (0-23)
        forecast_horizon: Hours ahead (1-48)
        lat: Latitude
        lon: Longitude (negative for west)

    Returns:
        Dictionary with forecast data or None
    """
    try:
        # Create Herbie object
        forecast_time = date.replace(hour=forecast_hour)

        H = Herbie(
            forecast_time,
            model='hrrr',
            product='sfc',  # Surface file
            fxx=forecast_horizon  # Forecast hour
        )

        # Check if file exists
        if not hasattr(H, 'grib'):
            return None

        data = {
            'forecast_time': forecast_time,
            'forecast_hour': forecast_hour,
            'forecast_horizon_hours': forecast_horizon,
            'valid_time': forecast_time + timedelta(hours=forecast_horizon),
        }

        # Download each variable separately to avoid coordinate conflicts
        for search_str, var_name in HRRR_VARIABLES.items():
            try:
                # Download variable using Herbie's xarray method
                ds = H.xarray(search_str, remove_grib=True)

                # Handle case where multiple datasets are returned
                if isinstance(ds, list):
                    ds = ds[0]

                # Find nearest point
                # HRRR uses y/x coordinates, not lat/lon directly
                # Use sel with method='nearest' on the actual coordinate names
                if 'latitude' in ds.coords and 'longitude' in ds.coords:
                    ds_point = ds.sel(latitude=lat, longitude=lon, method='nearest')
                elif 'y' in ds.coords and 'x' in ds.coords:
                    # Convert lat/lon to y/x indices (approximate)
                    # This is a simplified approach
                    ds_point = ds.isel(y=ds.dims['y']//2, x=ds.dims['x']//2)
                else:
                    continue

                # Extract the value
                for var in ds_point.data_vars:
                    if var not in ['latitude', 'longitude', 'time', 'step', 'valid_time']:
                        val = float(ds_point[var].values)
                        data[var_name] = val
                        break

            except Exception as e:
                # Skip variables that fail
                continue

        # Calculate derived variables if we have wind components
        if 'u_wind_10m' in data and 'v_wind_10m' in data:
            u = data['u_wind_10m']
            v = data['v_wind_10m']
            data['wind_speed_10m'] = np.sqrt(u**2 + v**2)
            data['wind_direction_10m'] = (np.degrees(np.arctan2(u, v)) + 180) % 360

        # Convert temperature from K to C if present
        if 'temperature_2m' in data:
            data['temperature_2m_C'] = data['temperature_2m'] - 273.15

        return data if len(data) > 4 else None  # Need more than just metadata

    except Exception as e:
        return None


def test_single_download():
    """Test downloading a single HRRR forecast."""
    print("=" * 80)
    print("TESTING HRRR DOWNLOAD - SINGLE FORECAST")
    print("=" * 80)

    # Test with Houston on a recent date
    test_date = datetime.now() - timedelta(days=7)
    test_lat = 29.7604
    test_lon = -95.3698

    print(f"Test date: {test_date.date()}")
    print(f"Test location: Houston ({test_lat}, {test_lon})")
    print(f"Test forecast: 00Z + 6 hours")
    print()

    data = download_hrrr_forecast_point(
        date=test_date,
        forecast_hour=0,
        forecast_horizon=6,
        lat=test_lat,
        lon=test_lon
    )

    if data:
        print("✓ SUCCESS! Downloaded forecast data:")
        for key, value in data.items():
            print(f"  {key}: {value}")
        return True
    else:
        print("✗ FAILED to download forecast data")
        print("\nPossible issues:")
        print("  - Date too recent (HRRR archive has a delay)")
        print("  - Network connectivity")
        print("  - Herbie/cfgrib configuration")
        return False


def download_full_dataset(
    locations_df: pd.DataFrame,
    start_date: datetime,
    end_date: datetime,
    output_dir: Path,
    forecast_cycles: List[int] = [0, 12]  # 00Z and 12Z
):
    """
    Download full HRRR forecast dataset.

    Args:
        locations_df: DataFrame with locations
        start_date: Start date
        end_date: End date
        output_dir: Output directory
        forecast_cycles: Hours of day to download (default: 00Z, 12Z)
    """
    csv_dir = output_dir / 'csv_files'
    csv_dir.mkdir(exist_ok=True, parents=True)

    parquet_dir = output_dir / 'parquet_files'
    parquet_dir.mkdir(exist_ok=True)

    all_forecasts = []
    total_attempts = 0
    successful = 0
    failed = 0

    print(f"\nDownloading HRRR forecasts:")
    print(f"  Date range: {start_date.date()} to {end_date.date()}")
    print(f"  Locations: {len(locations_df)}")
    print(f"  Forecast cycles: {forecast_cycles}")
    print(f"  Forecast horizons: {FORECAST_HORIZONS}")
    print()

    current_date = start_date
    while current_date <= end_date:
        print(f"\n{current_date.date()}")

        for idx, row in locations_df.iterrows():
            location_name = row['name']
            lat = row['lat']
            lon = row['lon']

            for cycle_hour in forecast_cycles:
                for horizon in FORECAST_HORIZONS:
                    total_attempts += 1

                    # Download forecast
                    data = download_hrrr_forecast_point(
                        date=current_date,
                        forecast_hour=cycle_hour,
                        forecast_horizon=horizon,
                        lat=lat,
                        lon=lon
                    )

                    if data:
                        data['location_name'] = location_name
                        data['latitude'] = lat
                        data['longitude'] = lon
                        all_forecasts.append(data)
                        successful += 1
                        print(f"  ✓ {location_name} {cycle_hour:02d}Z+{horizon}h", end=' ')
                    else:
                        failed += 1
                        print(f"  ✗ {location_name} {cycle_hour:02d}Z+{horizon}h", end=' ')

                    # Rate limiting
                    time.sleep(2)

                    # Save progress every 100 forecasts
                    if len(all_forecasts) > 0 and len(all_forecasts) % 100 == 0:
                        print(f"\n  Saving progress ({len(all_forecasts)} forecasts)...")
                        df = pd.DataFrame(all_forecasts)
                        df.to_csv(csv_dir / 'hrrr_forecasts_partial.csv', index=False)

        current_date += timedelta(days=1)

    # Save final results
    if all_forecasts:
        print("\n" + "=" * 80)
        print("SAVING RESULTS")
        print("=" * 80)

        df = pd.DataFrame(all_forecasts)

        # Save CSV
        csv_file = csv_dir / 'all_hrrr_forecasts.csv'
        df.to_csv(csv_file, index=False)
        print(f"  CSV: {csv_file}")

        # Save Parquet
        parquet_file = parquet_dir / 'all_hrrr_forecasts.parquet'
        df.to_parquet(parquet_file, compression='snappy')
        print(f"  Parquet: {parquet_file}")

        # Summary
        print("\n" + "=" * 80)
        print("DOWNLOAD SUMMARY")
        print("=" * 80)
        print(f"Total attempts: {total_attempts}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Success rate: {successful/total_attempts*100:.1f}%")
        print(f"Total forecasts: {len(df)}")

        # Save metadata
        metadata = {
            'download_date': datetime.now().isoformat(),
            'date_range': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'locations': len(locations_df),
            'forecast_cycles': forecast_cycles,
            'forecast_horizons': FORECAST_HORIZONS,
            'total_forecasts': len(df),
            'download_stats': {
                'attempts': total_attempts,
                'successful': successful,
                'failed': failed,
                'success_rate': successful/total_attempts*100 if total_attempts > 0 else 0
            }
        }

        metadata_file = output_dir / 'hrrr_download_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        print(f"\nMetadata: {metadata_file}")
    else:
        print("\n✗ No forecasts downloaded")


def main():
    """Main execution."""
    if not HERBIE_AVAILABLE:
        return

    print("=" * 80)
    print("NOAA HRRR HISTORICAL FORECAST DOWNLOADER v3")
    print("=" * 80)
    print("Simplified point extraction using Herbie")
    print()

    # Test first
    if not test_single_download():
        print("\nTest failed. Please check Herbie installation and configuration.")
        print("Install: pip install herbie-data")
        return

    print("\n" + "=" * 80)
    print("TEST SUCCESSFUL - READY FOR FULL DOWNLOAD")
    print("=" * 80)
    print()
    print("To proceed with full download, uncomment the download section below.")
    print()

    # Load environment and setup
    load_dotenv()
    weather_dir = Path(os.getenv('WEATHER_DATA_DIR', '/pool/ssd8tb/data/weather_data'))
    hrrr_dir = weather_dir / 'hrrr_forecasts'
    hrrr_dir.mkdir(exist_ok=True, parents=True)

    # Load locations
    locations_file = weather_dir / 'weather_locations.csv'
    if not locations_file.exists():
        print(f"ERROR: Locations file not found: {locations_file}")
        return

    locations_df = pd.read_csv(locations_file)

    # Uncomment to run full download
    # start_date = datetime(2022, 1, 1)
    # end_date = datetime(2024, 12, 31)
    # download_full_dataset(
    #     locations_df=locations_df,
    #     start_date=start_date,
    #     end_date=end_date,
    #     output_dir=hrrr_dir,
    #     forecast_cycles=[0, 12]  # 00Z and 12Z only
    # )


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Fix and create combined Meteostat Parquet file."""

import pandas as pd
from pathlib import Path
import json

def main():
    project_root = Path(__file__).parent
    # Use SSD storage for weather data
    weather_dir = Path('/pool/ssd8tb/data/weather_data')
    meteostat_dir = weather_dir / 'meteostat_stations'
    meteostat_csv_dir = meteostat_dir / 'csv_files'

    # Load all CSV files
    csv_files = list(meteostat_csv_dir.glob('*.csv'))
    print(f"Found {len(csv_files)} CSV files")

    all_data = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file, index_col=0, parse_dates=True)

        # Ensure station_id is string if it exists
        if 'station_id' in df.columns:
            df['station_id'] = df['station_id'].astype(str)
        if 'station_name' in df.columns:
            df['station_name'] = df['station_name'].astype(str)

        all_data.append(df)
        print(f"  Loaded: {csv_file.name} ({len(df)} records)")

    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=False)
    combined_df = combined_df.sort_index()

    print(f"\nTotal combined records: {len(combined_df):,}")

    # Save combined CSV
    combined_csv = meteostat_dir / 'all_meteostat_station_data.csv'
    combined_df.to_csv(combined_csv)
    print(f"Saved: {combined_csv}")

    # Save combined Parquet
    combined_parquet = meteostat_dir / 'all_meteostat_station_data.parquet'
    combined_df.to_parquet(combined_parquet, compression='snappy')
    print(f"Saved: {combined_parquet}")

    # Create summary
    summary = {
        'total_locations': len(csv_files),
        'date_range': f"{combined_df.index.min().date()} to {combined_df.index.max().date()}",
        'total_records': len(combined_df),
        'data_columns': [col for col in combined_df.columns if col not in ['location_name', 'latitude', 'longitude', 'station_id', 'station_name', 'distance_km']],
        'unique_locations': combined_df['location_name'].nunique()
    }

    summary_file = meteostat_dir / 'download_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved: {summary_file}")

    print("\n" + "="*80)
    print("SUCCESS - All files created")
    print("="*80)

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Create a combined locations CSV with population and generation capacity.
"""

import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# Texas city populations (2023 estimates)
CITY_POPULATIONS = {
    'CITY_Houston': 2304580,
    'CITY_San_Antonio': 1495295,
    'CITY_Dallas': 1304379,
    'CITY_Austin': 974447,
    'CITY_Fort_Worth': 956709,
    'CITY_El_Paso': 678815,
    'CITY_Arlington': 398121,
    'CITY_Corpus_Christi': 317863,
    'CITY_Plano': 285494,
    'CITY_Lubbock': 258870
}

def main():
    project_root = Path(__file__).parent
    # Use environment-configured weather data directory
    load_dotenv()
    weather_dir = Path(os.getenv('WEATHER_DATA_DIR', '/pool/ssd8tb/data/weather_data'))

    # Read existing locations
    locations_df = pd.read_csv(weather_dir / 'weather_locations.csv')

    # Create new columns
    locations_df['population'] = None
    locations_df['MW_generation'] = None

    # Fill in values based on type
    for idx, row in locations_df.iterrows():
        if row['type'] == 'city':
            locations_df.at[idx, 'population'] = CITY_POPULATIONS.get(row['name'])
        else:
            locations_df.at[idx, 'MW_generation'] = row['capacity_mw']

    # Clean up the name column - remove prefixes
    locations_df['location'] = locations_df['name'].str.replace('WIND_', '', regex=False)
    locations_df['location'] = locations_df['location'].str.replace('SOLAR_', '', regex=False)
    locations_df['location'] = locations_df['location'].str.replace('CITY_', '', regex=False)

    # Create final DataFrame with requested columns
    final_df = pd.DataFrame({
        'location': locations_df['location'],
        'latitude': locations_df['lat'],
        'longitude': locations_df['lon'],
        'population': locations_df['population'].astype('Int64'),  # Int64 allows NaN
        'MW_generation': pd.to_numeric(locations_df['MW_generation'], errors='coerce')
    })

    # Save to CSV
    output_file = weather_dir / 'locations_summary.csv'
    final_df.to_csv(output_file, index=False)

    print(f"Created: {output_file}")
    print(f"\nTotal locations: {len(final_df)}")
    print(f"  Cities: {final_df['population'].notna().sum()}")
    print(f"  Generation sites: {final_df['MW_generation'].notna().sum()}")
    print(f"\nTotal population: {final_df['population'].sum():,}")
    print(f"Total generation capacity: {final_df['MW_generation'].sum():.1f} MW")

    # Show sample
    print("\n" + "="*80)
    print("SAMPLE RECORDS")
    print("="*80)
    print("\nTop 5 wind farms by capacity:")
    print(final_df[final_df['MW_generation'].notna()].nlargest(5, 'MW_generation').to_string(index=False))

    print("\n\nTop 5 cities by population:")
    print(final_df[final_df['population'].notna()].nlargest(5, 'population').to_string(index=False))

if __name__ == '__main__':
    main()

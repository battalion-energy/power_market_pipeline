#!/usr/bin/env python3
"""
Create weather location lists for all major ISOs in the US.

For each ISO:
- Top 10 cities by population
- Top wind farms by capacity
- Top solar farms by capacity

Output:
- CSV file with all locations
- JSON file mapping locations to ISOs (for ML training)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

# Top cities by ISO region (2023 population estimates)
ISO_CITIES = {
    'ERCOT': [
        ('Houston', 'TX', 29.7604, -95.3698, 2304580),
        ('San Antonio', 'TX', 29.4241, -98.4936, 1495295),
        ('Dallas', 'TX', 32.7767, -96.7970, 1304379),
        ('Austin', 'TX', 30.2672, -97.7431, 974447),
        ('Fort Worth', 'TX', 32.7555, -97.3308, 956709),
        ('El Paso', 'TX', 31.7619, -106.4850, 678815),
        ('Arlington', 'TX', 32.7357, -97.1081, 398121),
        ('Corpus Christi', 'TX', 27.8006, -97.3964, 317863),
        ('Plano', 'TX', 33.0198, -96.6989, 285494),
        ('Lubbock', 'TX', 33.5779, -101.8552, 258870),
    ],
    'CAISO': [
        ('Los Angeles', 'CA', 34.0522, -118.2437, 3898747),
        ('San Diego', 'CA', 32.7157, -117.1611, 1386932),
        ('San Jose', 'CA', 37.3382, -121.8863, 1013240),
        ('San Francisco', 'CA', 37.7749, -122.4194, 873965),
        ('Fresno', 'CA', 36.7378, -119.7871, 542107),
        ('Sacramento', 'CA', 38.5816, -121.4944, 524943),
        ('Long Beach', 'CA', 33.7701, -118.1937, 466742),
        ('Oakland', 'CA', 37.8044, -122.2712, 440646),
        ('Bakersfield', 'CA', 35.3733, -119.0187, 403455),
        ('Anaheim', 'CA', 33.8366, -117.9143, 346824),
    ],
    'PJM': [
        ('Philadelphia', 'PA', 39.9526, -75.1652, 1584064),
        ('Columbus', 'OH', 39.9612, -82.9988, 905748),
        ('Charlotte', 'NC', 35.2271, -80.8431, 897720),
        ('Indianapolis', 'IN', 39.7684, -86.1581, 879293),
        ('Baltimore', 'MD', 39.2904, -76.6122, 585708),
        ('Washington', 'DC', 38.9072, -77.0369, 689545),
        ('Nashville', 'TN', 36.1627, -86.7816, 689447),
        ('Memphis', 'TN', 35.1495, -90.0490, 633104),
        ('Louisville', 'KY', 38.2527, -85.7585, 624444),
        ('Richmond', 'VA', 37.5407, -77.4360, 226610),
    ],
    'MISO': [
        ('Chicago', 'IL', 41.8781, -87.6298, 2746388),
        ('Detroit', 'MI', 42.3314, -83.0458, 639111),
        ('Milwaukee', 'WI', 43.0389, -87.9065, 577222),
        ('Minneapolis', 'MN', 44.9778, -93.2650, 429954),
        ('St. Louis', 'MO', 38.6270, -90.1994, 301578),
        ('New Orleans', 'LA', 29.9511, -90.0715, 383997),
        ('Kansas City', 'MO', 39.0997, -94.5786, 508090),
        ('Omaha', 'NE', 41.2565, -95.9345, 486051),
        ('Baton Rouge', 'LA', 30.4515, -91.1871, 227470),
        ('Des Moines', 'IA', 41.5868, -93.6250, 214133),
    ],
    'NYISO': [
        ('New York', 'NY', 40.7128, -74.0060, 8336817),
        ('Buffalo', 'NY', 42.8864, -78.8784, 276807),
        ('Rochester', 'NY', 43.1566, -77.6088, 211328),
        ('Yonkers', 'NY', 40.9312, -73.8987, 211569),
        ('Syracuse', 'NY', 43.0481, -76.1474, 148620),
        ('Albany', 'NY', 42.6526, -73.7562, 99224),
        ('New Rochelle', 'NY', 40.9115, -73.7823, 79726),
        ('Mount Vernon', 'NY', 40.9126, -73.8370, 73893),
        ('Schenectady', 'NY', 42.8142, -73.9396, 67047),
        ('Utica', 'NY', 43.1009, -75.2327, 65283),
    ],
    'ISONE': [
        ('Boston', 'MA', 42.3601, -71.0589, 675647),
        ('Worcester', 'MA', 42.2626, -71.8023, 206518),
        ('Providence', 'RI', 41.8240, -71.4128, 190934),
        ('Springfield', 'MA', 42.1015, -72.5898, 155929),
        ('Bridgeport', 'CT', 41.1865, -73.1952, 148654),
        ('Hartford', 'CT', 41.7658, -72.6734, 121054),
        ('Manchester', 'NH', 42.9956, -71.4548, 115644),
        ('Portland', 'ME', 43.6591, -70.2568, 68408),
        ('Cambridge', 'MA', 42.3736, -71.1097, 118403),
        ('New Haven', 'CT', 41.3083, -72.9279, 134023),
    ],
    'SPP': [
        ('Oklahoma City', 'OK', 35.4676, -97.5164, 681054),
        ('Tulsa', 'OK', 36.1540, -95.9928, 413066),
        ('Wichita', 'KS', 37.6872, -97.3301, 397532),
        ('Lincoln', 'NE', 40.8136, -96.7026, 291082),
        ('Fargo', 'ND', 46.8772, -96.7898, 125990),
        ('Sioux Falls', 'SD', 43.5460, -96.7313, 192517),
        ('Amarillo', 'TX', 35.2220, -101.8313, 200393),
        ('Topeka', 'KS', 39.0558, -95.6890, 126587),
        ('Overland Park', 'KS', 38.9822, -94.6708, 197238),
        ('Norman', 'OK', 35.2226, -97.4395, 128026),
    ],
}

# Balancing Authority to ISO mapping
BA_TO_ISO = {
    # ERCOT
    'ERCO': 'ERCOT',

    # CAISO
    'CISO': 'CAISO',
    'TIDC': 'CAISO',  # Turlock Irrigation District
    'BANC': 'CAISO',  # Balancing Authority of Northern California

    # PJM
    'PJM': 'PJM',

    # MISO
    'MISO': 'MISO',

    # NYISO
    'NYIS': 'NYISO',

    # ISO-NE
    'ISNE': 'ISONE',

    # SPP
    'SWPP': 'SPP',
    'WFEC': 'SPP',  # Western Farmers Electric Cooperative
    'OKGE': 'SPP',  # Oklahoma Gas and Electric
}


def load_eia_generators():
    """Load EIA-860 generator data."""
    eia_file = '/home/enrico/experiments/ERCOT_SCED/pypsa-usa/workflow/repo_data/plants/eia860_ads_merged.csv'

    print(f"Loading EIA-860 data from: {eia_file}")

    # Read only columns we need
    cols = [
        'plant_code', 'plant_name', 'generator_id',
        'nameplate_capacity_mw', 'technology', 'status',
        'balancing_authority_code', 'state', 'latitude', 'longitude'
    ]

    df = pd.read_csv(eia_file, usecols=cols)

    # Filter to operating generators
    df = df[df['status'].isin(['OP', 'Operating', 'V'])]  # Operating or planned

    # Map BA codes to ISOs
    df['iso'] = df['balancing_authority_code'].map(BA_TO_ISO)

    # Remove generators without valid coordinates
    df = df.dropna(subset=['latitude', 'longitude'])

    # Remove invalid coordinates (0, 0 or outside continental US)
    df = df[
        (df['latitude'] >= 24) & (df['latitude'] <= 50) &
        (df['longitude'] >= -125) & (df['longitude'] <= -66)
    ]

    print(f"Loaded {len(df):,} generators")
    print(f"ISOs found: {df['iso'].dropna().unique()}")

    return df


def get_top_generators_by_iso(df, technology_filter, n_per_iso=30):
    """Get top N generators by capacity for each ISO."""

    # Filter by technology
    tech_df = df[df['technology'].str.contains(technology_filter, case=False, na=False)]

    # Group by plant and sum capacity (multiple generators per plant)
    plant_df = tech_df.groupby(
        ['plant_code', 'plant_name', 'iso', 'state', 'latitude', 'longitude']
    )['nameplate_capacity_mw'].sum().reset_index()

    # Get top N per ISO
    top_plants = []
    for iso in plant_df['iso'].dropna().unique():
        iso_plants = plant_df[plant_df['iso'] == iso].nlargest(n_per_iso, 'nameplate_capacity_mw')
        top_plants.append(iso_plants)

    result = pd.concat(top_plants, ignore_index=True)

    print(f"\n{technology_filter}: {len(result)} plants across {result['iso'].nunique()} ISOs")
    for iso in sorted(result['iso'].unique()):
        count = len(result[result['iso'] == iso])
        total_mw = result[result['iso'] == iso]['nameplate_capacity_mw'].sum()
        print(f"  {iso}: {count} plants, {total_mw:,.0f} MW")

    return result


def create_location_dataframe():
    """Create comprehensive location DataFrame for all ISOs."""

    # Load EIA data
    eia_df = load_eia_generators()

    # Get top wind farms
    print("\n" + "="*80)
    print("EXTRACTING WIND FARMS")
    print("="*80)
    wind_df = get_top_generators_by_iso(eia_df, 'Wind', n_per_iso=30)
    wind_df['type'] = 'wind'

    # Get top solar farms
    print("\n" + "="*80)
    print("EXTRACTING SOLAR FARMS")
    print("="*80)
    solar_df = get_top_generators_by_iso(eia_df, 'Solar', n_per_iso=15)
    solar_df['type'] = 'solar'

    # Prepare generator data
    gen_df = pd.concat([wind_df, solar_df], ignore_index=True)
    gen_df['name'] = gen_df.apply(
        lambda row: f"{row['type'].upper()}_{row['plant_name'].replace(' ', '_').replace('/', '_')[:40]}",
        axis=1
    )
    gen_df['population'] = None
    gen_df['capacity_mw'] = gen_df['nameplate_capacity_mw']

    # Prepare city data
    print("\n" + "="*80)
    print("ADDING CITIES")
    print("="*80)

    city_rows = []
    for iso, cities in ISO_CITIES.items():
        for city_name, state, lat, lon, pop in cities:
            city_rows.append({
                'name': f'CITY_{city_name.replace(" ", "_")}',
                'plant_name': city_name,
                'iso': iso,
                'state': state,
                'latitude': lat,
                'longitude': lon,
                'type': 'city',
                'population': pop,
                'capacity_mw': None,
            })
        print(f"  {iso}: {len(cities)} cities")

    city_df = pd.DataFrame(city_rows)

    # Combine all locations
    locations_df = pd.concat([
        gen_df[['name', 'plant_name', 'iso', 'state', 'latitude', 'longitude', 'type', 'population', 'capacity_mw']],
        city_df
    ], ignore_index=True)

    # Rename columns for consistency
    locations_df = locations_df.rename(columns={
        'latitude': 'lat',
        'longitude': 'lon'
    })

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total locations: {len(locations_df)}")
    print(f"\nBy ISO:")
    for iso in sorted(locations_df['iso'].unique()):
        iso_locs = locations_df[locations_df['iso'] == iso]
        n_cities = len(iso_locs[iso_locs['type'] == 'city'])
        n_wind = len(iso_locs[iso_locs['type'] == 'wind'])
        n_solar = len(iso_locs[iso_locs['type'] == 'solar'])
        print(f"  {iso}: {len(iso_locs)} total ({n_cities} cities, {n_wind} wind, {n_solar} solar)")

    print(f"\nBy type:")
    for type_name in ['city', 'wind', 'solar']:
        count = len(locations_df[locations_df['type'] == type_name])
        print(f"  {type_name}: {count}")

    return locations_df


def create_iso_mapping_json(locations_df, output_file):
    """Create JSON file mapping locations to ISOs for ML training."""

    iso_mapping = {}

    for iso in sorted(locations_df['iso'].unique()):
        iso_locs = locations_df[locations_df['iso'] == iso]

        iso_mapping[iso] = {
            'name': iso,
            'total_locations': len(iso_locs),
            'cities': iso_locs[iso_locs['type'] == 'city']['name'].tolist(),
            'wind_farms': iso_locs[iso_locs['type'] == 'wind']['name'].tolist(),
            'solar_farms': iso_locs[iso_locs['type'] == 'solar']['name'].tolist(),
            'location_details': []
        }

        for _, row in iso_locs.iterrows():
            detail = {
                'name': row['name'],
                'display_name': row['plant_name'],
                'type': row['type'],
                'state': row['state'],
                'lat': round(row['lat'], 4),
                'lon': round(row['lon'], 4),
            }

            if row['type'] == 'city':
                detail['population'] = int(row['population']) if pd.notna(row['population']) else None
            else:
                detail['capacity_mw'] = round(row['capacity_mw'], 1) if pd.notna(row['capacity_mw']) else None

            iso_mapping[iso]['location_details'].append(detail)

    # Add metadata
    metadata = {
        'created_date': pd.Timestamp.now().isoformat(),
        'total_locations': len(locations_df),
        'total_isos': len(iso_mapping),
        'iso_list': list(iso_mapping.keys()),
        'data_sources': [
            'EIA-860 (generators)',
            'US Census (population)',
            'ISO boundaries'
        ],
        'coverage': {
            'date_range': '2019-01-01 to present',
            'update_frequency': 'daily (automated)',
            'weather_sources': ['NASA POWER', 'Meteostat']
        }
    }

    full_mapping = {
        'metadata': metadata,
        'isos': iso_mapping
    }

    # Save JSON
    with open(output_file, 'w') as f:
        json.dump(full_mapping, f, indent=2)

    print(f"\nISO mapping JSON saved to: {output_file}")
    print(f"  {len(iso_mapping)} ISOs")
    print(f"  {len(locations_df)} locations")


def main():
    """Main execution."""
    print("="*80)
    print("MULTI-ISO WEATHER LOCATION GENERATOR")
    print("="*80)
    print()

    # Setup output directory
    weather_dir = Path('/pool/ssd8tb/data/weather_data')
    weather_dir.mkdir(exist_ok=True, parents=True)

    # Create locations DataFrame
    locations_df = create_location_dataframe()

    # Save locations CSV
    csv_file = weather_dir / 'weather_locations_all_isos.csv'
    locations_df.to_csv(csv_file, index=False)
    print(f"\nLocations CSV saved to: {csv_file}")

    # Create ISO mapping JSON
    json_file = weather_dir / 'iso_location_mapping.json'
    create_iso_mapping_json(locations_df, json_file)

    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print("\nFiles created:")
    print(f"  1. {csv_file}")
    print(f"  2. {json_file}")
    print("\nNext steps:")
    print("  1. Run download_nasa_power_weather.py with --incremental flag")
    print("  2. Run download_meteostat_weather.py with --incremental flag")
    print("  3. Set up cron job for daily updates")


if __name__ == '__main__':
    main()

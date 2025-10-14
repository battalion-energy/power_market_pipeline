#!/usr/bin/env python3
"""
Create comprehensive BESS mapping with all IQ and EIA data plus coordinates
Includes geocoding via Google Places API and county center fallbacks
"""

import pandas as pd
import numpy as np
import requests
import json
import os
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv('/home/enrico/projects/battalion-platform/.env')

def load_all_iq_data():
    """Load all columns from interconnection queue data"""
    print("📂 Loading complete IQ data...")
    
    # Load the CSVs with all columns
    coloc_op = pd.read_csv('/home/enrico/projects/power_market_pipeline/interconnection_queue_clean/co_located_operational.csv')
    standalone = pd.read_csv('/home/enrico/projects/power_market_pipeline/interconnection_queue_clean/stand_alone.csv')
    solar_coloc = pd.read_csv('/home/enrico/projects/power_market_pipeline/interconnection_queue_clean/co_located_with_solar.csv')
    wind_coloc = pd.read_csv('/home/enrico/projects/power_market_pipeline/interconnection_queue_clean/co_located_with_wind.csv')
    
    # Combine all IQ data
    all_iq = pd.concat([
        coloc_op.assign(IQ_Sheet='Co-located Operational'),
        standalone.assign(IQ_Sheet='Stand-Alone'),
        solar_coloc.assign(IQ_Sheet='Co-located with Solar'),
        wind_coloc.assign(IQ_Sheet='Co-located with Wind')
    ], ignore_index=True)
    
    # Rename columns to be clear they're from IQ
    iq_columns = {}
    for col in all_iq.columns:
        if col not in ['IQ_Sheet']:
            iq_columns[col] = f'IQ_{col}'
    all_iq = all_iq.rename(columns=iq_columns)
    
    print(f"  Loaded {len(all_iq)} total IQ projects with {len(all_iq.columns)} columns")
    return all_iq

def load_all_eia_data():
    """Load all columns from EIA generator data including lat/long"""
    print("📂 Loading complete EIA data with coordinates...")
    
    # Load EIA data with all columns
    eia_operating = pd.read_excel(
        '/home/enrico/projects/battalion-platform/data/EIA/generators/EIA_generators_latest.xlsx',
        sheet_name='Operating',
        header=2
    )
    
    eia_planned = pd.read_excel(
        '/home/enrico/projects/battalion-platform/data/EIA/generators/EIA_generators_latest.xlsx',
        sheet_name='Planned',
        header=2
    )
    
    # Filter for Texas battery storage
    def filter_tx_battery(df):
        return df[
            (df['Plant State'] == 'TX') &
            ((df['Technology'].str.contains('Battery', na=False)) |
             (df['Energy Source Code'].str.contains('MWH', na=False)) |
             (df['Prime Mover Code'].str.contains('BA', na=False)))
        ].copy()
    
    eia_op_tx = filter_tx_battery(eia_operating).assign(EIA_Sheet='Operating')
    eia_plan_tx = filter_tx_battery(eia_planned).assign(EIA_Sheet='Planned')
    
    # Combine
    all_eia = pd.concat([eia_op_tx, eia_plan_tx], ignore_index=True)
    
    # Rename columns to be clear they're from EIA
    eia_columns = {}
    for col in all_eia.columns:
        if col not in ['EIA_Sheet'] and not col.startswith('EIA_'):
            eia_columns[col] = f'EIA_{col}'
    all_eia = all_eia.rename(columns=eia_columns)
    
    print(f"  Loaded {len(all_eia)} TX battery facilities with {len(all_eia.columns)} columns")
    
    # Check for lat/long columns
    lat_cols = [c for c in all_eia.columns if 'Lat' in c]
    lon_cols = [c for c in all_eia.columns if 'Lon' in c]
    print(f"  Found lat/long columns: {lat_cols}, {lon_cols}")
    
    return all_eia

def geocode_with_google_places(location_name, county, state="Texas"):
    """Geocode a location using Google Places API"""
    api_key = os.getenv('GOOGLE_MAPS_KEY')
    if not api_key:
        return None, None, "No API key"
    
    # Try searching for the substation
    query = f"{location_name} substation {county} County {state}"
    
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {
        'query': query,
        'key': api_key,
        'region': 'us'
    }
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        if data.get('status') == 'OK' and data.get('results'):
            # Take the first result
            result = data['results'][0]
            location = result['geometry']['location']
            return location['lat'], location['lng'], f"Google Places: {result.get('name', 'Unknown')}"
        else:
            return None, None, f"No results for: {query}"
    except Exception as e:
        return None, None, f"API error: {str(e)}"

def get_texas_county_centers():
    """Get center coordinates for Texas counties"""
    # Major Texas counties with their approximate centers
    county_centers = {
        'HARRIS': (29.7604, -95.3698),
        'DALLAS': (32.7767, -96.7970),
        'TARRANT': (32.7555, -97.3308),
        'BEXAR': (29.4241, -98.4936),
        'TRAVIS': (30.2672, -97.7431),
        'FORT BEND': (29.5694, -95.7676),
        'WILLIAMSON': (30.6321, -97.6780),
        'NUECES': (27.8006, -97.3964),
        'DENTON': (33.2148, -97.1331),
        'COLLIN': (33.1795, -96.4930),
        'BELL': (31.0595, -97.4977),
        'BRAZORIA': (29.1694, -95.4185),
        'GALVESTON': (29.3013, -94.7977),
        'HIDALGO': (26.1004, -98.2630),
        'MONTGOMERY': (30.3072, -95.4955),
        'BRAZOS': (30.6280, -96.3344),
        'WEBB': (27.5306, -99.4803),
        'JEFFERSON': (29.8499, -94.1951),
        'CAMERON': (26.1224, -97.6355),
        'GUADALUPE': (29.5729, -97.9478),
        'ECTOR': (31.8673, -102.5406),
        'REEVES': (31.4199, -103.4814),
        'WARD': (31.4885, -103.1394),
        'PECOS': (30.8823, -102.2882),
        'MAVERICK': (28.7086, -100.4837),
        'ZAPATA': (26.9073, -99.2717),
        'STARR': (26.5540, -98.7319),
        'VAL VERDE': (29.3709, -100.8959),
        'LA SALLE': (28.3408, -99.0952),
        'DIMMIT': (28.4199, -99.7573),
        'HALE': (34.0731, -101.8238),
        'YOUNG': (33.1771, -98.6989),
        'KIMBLE': (30.4863, -99.7428),
        'MATAGORDA': (28.8055, -95.9669),
        'ANGELINA': (31.2546, -94.6088),
        'PALO PINTO': (32.7679, -98.2976),
        'HASKELL': (33.1576, -99.7337),
        'MASON': (30.7488, -99.2303),
        'REAGAN': (31.3482, -101.5268),
        'UPTON': (31.3682, -102.0779),
        'COKE': (31.8857, -100.5321),
        'HILL': (32.0085, -97.1253),
        'LEON': (31.4365, -95.9966),
        'BASTROP': (30.1105, -97.3151),
        'DELTA': (33.3829, -95.6655),
        'WISE': (33.2601, -97.6364),
        'BROOKS': (27.0200, -98.2211),
        'GRIMES': (30.5238, -95.9880),
        'FALLS': (31.2460, -96.9280),
        'KINNEY': (29.3566, -100.4201),
        'EASTLAND': (32.3324, -98.8256),
        'JIM HOGG': (27.0458, -98.6975),
        'SAN PATRICIO': (27.9958, -97.5169),
        'HENDERSON': (32.1532, -95.8513),
        'AUSTIN': (29.8831, -96.2772)
    }
    
    return county_centers

def create_comprehensive_mapping():
    """Create mapping with all IQ, EIA data and coordinates"""
    
    print("="*70)
    print("CREATING COMPREHENSIVE BESS MAPPING WITH COORDINATES")
    print("="*70)
    
    # Start with current unified mapping
    print("\n1️⃣ Loading base unified mapping...")
    unified = pd.read_csv('/home/enrico/projects/power_market_pipeline/BESS_UNIFIED_MAPPING_V3_CLARIFIED.csv')
    
    # Load all IQ data
    print("\n2️⃣ Loading complete IQ data...")
    all_iq = load_all_iq_data()
    
    # Match BESS to IQ projects (using existing matches from unified)
    # For each BESS, find its full IQ record
    iq_matches = []
    for _, row in unified.iterrows():
        bess_name = row['BESS_Gen_Resource']
        
        # Try to match by Unit Code or Project Name
        if pd.notna(row.get('IQ_Unit_Code')):
            match = all_iq[all_iq['IQ_Unit Code'] == row['IQ_Unit_Code']]
        elif pd.notna(row.get('IQ_Project_Name')):
            match = all_iq[all_iq['IQ_Project Name'] == row['IQ_Project_Name']]
        else:
            match = pd.DataFrame()
        
        if not match.empty:
            # Take first match and add BESS name
            match_row = match.iloc[0].to_dict()
            match_row['BESS_Gen_Resource'] = bess_name
            iq_matches.append(match_row)
        else:
            # No match, just keep BESS name
            iq_matches.append({'BESS_Gen_Resource': bess_name})
    
    iq_matched_df = pd.DataFrame(iq_matches)
    
    # Merge IQ data with unified
    unified_with_iq = unified.merge(
        iq_matched_df,
        on='BESS_Gen_Resource',
        how='left',
        suffixes=('', '_full')
    )
    
    # Load all EIA data
    print("\n3️⃣ Loading complete EIA data with coordinates...")
    all_eia = load_all_eia_data()
    
    # Match BESS to EIA projects
    eia_matches = []
    for _, row in unified.iterrows():
        bess_name = row['BESS_Gen_Resource']
        
        # Try to match by Plant Name or Generator ID
        if pd.notna(row.get('EIA_Plant_Name')):
            match = all_eia[all_eia['EIA_Plant Name'] == row['EIA_Plant_Name']]
        elif pd.notna(row.get('EIA_Generator_ID')):
            match = all_eia[all_eia['EIA_Generator ID'] == row['EIA_Generator_ID']]
        else:
            match = pd.DataFrame()
        
        if not match.empty:
            # Take first match and add BESS name
            match_row = match.iloc[0].to_dict()
            match_row['BESS_Gen_Resource'] = bess_name
            eia_matches.append(match_row)
        else:
            # No match, just keep BESS name
            eia_matches.append({'BESS_Gen_Resource': bess_name})
    
    eia_matched_df = pd.DataFrame(eia_matches)
    
    # Merge EIA data with unified+IQ
    comprehensive = unified_with_iq.merge(
        eia_matched_df,
        on='BESS_Gen_Resource',
        how='left',
        suffixes=('', '_eia_full')
    )
    
    print(f"\n✅ Created comprehensive mapping with {len(comprehensive.columns)} columns")
    
    # Save intermediate result
    comprehensive.to_csv('/home/enrico/projects/power_market_pipeline/BESS_COMPREHENSIVE_TEMP.csv', index=False)
    
    return comprehensive

def add_coordinates(df):
    """Add lat/long coordinates using multiple methods"""
    
    print("\n4️⃣ Adding coordinates...")
    
    # Initialize coordinate columns
    df['Latitude'] = np.nan
    df['Longitude'] = np.nan
    df['Coordinate_Source'] = ''
    
    # Get county centers
    county_centers = get_texas_county_centers()
    
    # Method 1: Use EIA coordinates if available
    if 'EIA_Latitude' in df.columns:
        has_eia_coords = df['EIA_Latitude'].notna() & df['EIA_Longitude'].notna()
        df.loc[has_eia_coords, 'Latitude'] = df.loc[has_eia_coords, 'EIA_Latitude']
        df.loc[has_eia_coords, 'Longitude'] = df.loc[has_eia_coords, 'EIA_Longitude']
        df.loc[has_eia_coords, 'Coordinate_Source'] = 'EIA Data'
        print(f"  ✅ Found {has_eia_coords.sum()} coordinates from EIA data")
    
    # Method 2: Geocode substations using Google Places API
    needs_geocoding = df['Latitude'].isna() & df['Substation'].notna()
    geocoded_count = 0
    
    print(f"  🌍 Geocoding {needs_geocoding.sum()} substations...")
    
    for idx in df[needs_geocoding].index[:20]:  # Limit to 20 to avoid API limits
        substation = df.loc[idx, 'Substation']
        county = df.loc[idx, 'IQ_County'] if 'IQ_County' in df.columns else None
        
        if pd.notna(substation):
            lat, lng, source = geocode_with_google_places(substation, county)
            
            if lat and lng:
                df.loc[idx, 'Latitude'] = lat
                df.loc[idx, 'Longitude'] = lng
                df.loc[idx, 'Coordinate_Source'] = source
                geocoded_count += 1
                print(f"    Geocoded {substation}: {lat:.4f}, {lng:.4f}")
            
            time.sleep(0.2)  # Rate limiting
    
    print(f"  ✅ Geocoded {geocoded_count} substations")
    
    # Method 3: Use county centers as fallback
    still_missing = df['Latitude'].isna()
    
    for idx in df[still_missing].index:
        # Try IQ County first, then EIA County
        county = None
        if 'IQ_County' in df.columns:
            county = df.loc[idx, 'IQ_County']
        if pd.isna(county) and 'EIA_County' in df.columns:
            county = df.loc[idx, 'EIA_County']
        
        if pd.notna(county):
            county_upper = str(county).upper().replace(' COUNTY', '').strip()
            if county_upper in county_centers:
                lat, lng = county_centers[county_upper]
                df.loc[idx, 'Latitude'] = lat
                df.loc[idx, 'Longitude'] = lng
                df.loc[idx, 'Coordinate_Source'] = f'County Center: {county}'
    
    # Summary
    has_coords = df['Latitude'].notna()
    print(f"\n📍 Coordinate Summary:")
    print(f"  Total BESS: {len(df)}")
    print(f"  With coordinates: {has_coords.sum()} ({100*has_coords.mean():.1f}%)")
    
    if 'Coordinate_Source' in df.columns:
        print("\n  By source:")
        for source, count in df['Coordinate_Source'].value_counts().items():
            if source:
                print(f"    {source}: {count}")
    
    return df

if __name__ == '__main__':
    # Create comprehensive mapping
    comprehensive_df = create_comprehensive_mapping()
    
    # Add coordinates
    final_df = add_coordinates(comprehensive_df)
    
    # Save final result
    output_file = '/home/enrico/projects/power_market_pipeline/BESS_COMPREHENSIVE_WITH_COORDINATES.csv'
    final_df.to_csv(output_file, index=False)
    
    print(f"\n✅ Saved comprehensive mapping with coordinates to:")
    print(f"   {output_file}")
    
    # Show sample
    print("\n📋 Sample records with coordinates:")
    sample_cols = ['BESS_Gen_Resource', 'Substation', 'Latitude', 'Longitude', 'Coordinate_Source']
    available_cols = [c for c in sample_cols if c in final_df.columns]
    print(final_df[final_df['Latitude'].notna()][available_cols].head(10).to_string(index=False))
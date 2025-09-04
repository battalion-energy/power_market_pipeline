#!/usr/bin/env python3
"""
Create comprehensive BESS mapping with improved location logic and load zone validation
Version 2 - Enhanced with:
1. EIA coordinates as first priority
2. LLM validation of Google Places results
3. Distance validation between substation and county
4. Improved load zone mapping logic
"""

import pandas as pd
import numpy as np
import requests
import json
import os
from dotenv import load_dotenv
import time
from geopy.distance import geodesic
from typing import Tuple, Optional
from openai import OpenAI

# Load environment variables
load_dotenv('/home/enrico/projects/battalion-platform/.env')

# Initialize OpenAI client for LLM validation (optional)
try:
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        client = OpenAI(api_key=api_key)
    else:
        client = None
        print("⚠️ OpenAI API key not found - LLM validation will be skipped")
except Exception as e:
    client = None
    print(f"⚠️ Could not initialize OpenAI client: {e}")

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

def validate_google_places_with_llm(location_name: str, google_result: dict, county: str) -> bool:
    """Use LLM to validate if Google Places result is a valid electric substation"""
    # If no OpenAI client, accept all results
    if not client:
        return True
    
    try:
        # Extract relevant info from Google result
        result_name = google_result.get('name', '')
        result_address = google_result.get('formatted_address', '')
        result_types = google_result.get('types', [])
        
        # Create validation prompt
        prompt = f"""You are validating whether a Google Places search result is a valid electric substation location.

Search query was for: "{location_name} substation {county} County Texas"

Google Places returned:
- Name: {result_name}
- Address: {result_address}
- Types: {', '.join(result_types)}

Is this likely to be an actual electric substation or BESS facility location? Consider:
1. Does the name contain words like "substation", "switching station", "electric", "power", "energy", "BESS", "battery"?
2. Is it in the correct county/area?
3. Are the place types consistent with an industrial/utility facility?

Respond with only "VALID" if this is likely a correct substation/BESS location, or "INVALID" if not."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=10
        )
        
        result = response.choices[0].message.content.strip().upper()
        return result == "VALID"
        
    except Exception as e:
        print(f"    ⚠️ LLM validation error: {str(e)}")
        # Default to accepting if LLM fails
        return True

def geocode_with_google_places(location_name: str, county: str, state: str = "Texas") -> Tuple[Optional[float], Optional[float], str]:
    """Geocode a location using Google Places API with LLM validation"""
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
            # Get first result
            result = data['results'][0]
            
            # Validate with LLM
            if validate_google_places_with_llm(location_name, result, county):
                location = result['geometry']['location']
                return location['lat'], location['lng'], f"Google Places (LLM validated): {result.get('name', 'Unknown')}"
            else:
                print(f"    ❌ LLM rejected Google result for {location_name}")
                return None, None, f"Google result rejected by LLM"
        else:
            return None, None, f"No results for: {query}"
    except Exception as e:
        return None, None, f"API error: {str(e)}"

def get_texas_county_centers():
    """Get center coordinates for Texas counties - expanded list"""
    county_centers = {
        # Major counties
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
        
        # West Texas counties
        'ECTOR': (31.8673, -102.5406),
        'MIDLAND': (31.9973, -102.0779),
        'REEVES': (31.4199, -103.4814),
        'WARD': (31.4885, -103.1394),
        'PECOS': (30.8823, -102.2882),
        'ANDREWS': (32.3048, -102.6379),
        'CRANE': (31.3976, -102.3569),
        'WINKLER': (31.8529, -103.0816),
        
        # South Texas
        'MAVERICK': (28.7086, -100.4837),
        'ZAPATA': (26.9073, -99.2717),
        'STARR': (26.5540, -98.7319),
        'VAL VERDE': (29.3709, -100.8959),
        'LA SALLE': (28.3408, -99.0952),
        'DIMMIT': (28.4199, -99.7573),
        'BROOKS': (27.0200, -98.2211),
        'JIM HOGG': (27.0458, -98.6975),
        'SAN PATRICIO': (27.9958, -97.5169),
        
        # North/Panhandle
        'HALE': (34.0731, -101.8238),
        'POTTER': (35.2220, -101.8313),
        'RANDALL': (34.9637, -101.9188),
        'LUBBOCK': (33.5779, -101.8552),
        'FLOYD': (34.1834, -101.3251),
        'SWISHER': (34.5184, -101.7571),
        
        # Central Texas
        'HILL': (32.0085, -97.1253),
        'MCLENNAN': (31.5493, -97.1467),
        'FALLS': (31.2460, -96.9280),
        'LIMESTONE': (31.5293, -96.5761),
        'LEON': (31.4365, -95.9966),
        'BASTROP': (30.1105, -97.3151),
        'GRIMES': (30.5238, -95.9880),
        
        # East Texas
        'HENDERSON': (32.1532, -95.8513),
        'ANGELINA': (31.2546, -94.6088),
        'NACOGDOCHES': (31.6035, -94.6535),
        'RUSK': (32.0959, -94.7691),
        'CHEROKEE': (31.9177, -95.1705),
        
        # Additional counties
        'YOUNG': (33.1771, -98.6989),
        'KIMBLE': (30.4863, -99.7428),
        'MATAGORDA': (28.8055, -95.9669),
        'PALO PINTO': (32.7679, -98.2976),
        'HASKELL': (33.1576, -99.7337),
        'MASON': (30.7488, -99.2303),
        'REAGAN': (31.3482, -101.5268),
        'UPTON': (31.3682, -102.0779),
        'COKE': (31.8857, -100.5321),
        'DELTA': (33.3829, -95.6655),
        'WISE': (33.2601, -97.6364),
        'KINNEY': (29.3566, -100.4201),
        'EASTLAND': (32.3324, -98.8256),
        'AUSTIN': (29.8831, -96.2772),
        'WHARTON': (29.3116, -96.1027),
        'LIBERTY': (30.0577, -94.7955),
        'CHAMBERS': (29.7239, -94.6819),
    }
    
    return county_centers

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two coordinates in miles"""
    try:
        distance = geodesic((lat1, lon1), (lat2, lon2)).miles
        return distance
    except:
        return float('inf')

def determine_load_zone(lat: float, lon: float) -> str:
    """Determine ERCOT load zone based on coordinates"""
    # Define approximate load zone boundaries
    # These are simplified polygons - in production use more precise boundaries
    
    if pd.isna(lat) or pd.isna(lon):
        return "Unknown"
    
    # Houston zone - Harris and surrounding counties
    if (28.5 <= lat <= 30.5) and (-96.0 <= lon <= -94.5):
        return "LZ_HOUSTON"
    
    # North zone - DFW metroplex and north
    elif lat >= 32.0 and lon >= -98.5:
        return "LZ_NORTH"
    
    # West zone - West Texas including Permian Basin
    elif lon <= -99.5:
        return "LZ_WEST"
    
    # South zone - Austin, San Antonio, and south
    elif lat <= 32.0:
        return "LZ_SOUTH"
    
    else:
        # Default based on more specific checks
        if lon <= -100.0:
            return "LZ_WEST"
        elif lat >= 31.5:
            return "LZ_NORTH"
        else:
            return "LZ_SOUTH"

def add_coordinates_improved(df):
    """Add lat/long coordinates using improved prioritization logic"""
    
    print("\n4️⃣ Adding coordinates with improved logic...")
    
    # Initialize coordinate columns
    df['Latitude'] = np.nan
    df['Longitude'] = np.nan
    df['Coordinate_Source'] = ''
    df['Distance_Validation'] = ''
    
    # Get county centers
    county_centers = get_texas_county_centers()
    
    # Priority 1: Use EIA coordinates if available
    print("\n  📍 Priority 1: EIA coordinates")
    if 'EIA_Latitude' in df.columns:
        has_eia_coords = df['EIA_Latitude'].notna() & df['EIA_Longitude'].notna()
        df.loc[has_eia_coords, 'Latitude'] = df.loc[has_eia_coords, 'EIA_Latitude']
        df.loc[has_eia_coords, 'Longitude'] = df.loc[has_eia_coords, 'EIA_Longitude']
        df.loc[has_eia_coords, 'Coordinate_Source'] = 'EIA Data'
        print(f"    ✅ Found {has_eia_coords.sum()} coordinates from EIA data")
    
    # Priority 2: Geocode substations with LLM validation
    print("\n  📍 Priority 2: Google Places with LLM validation")
    needs_geocoding = df['Latitude'].isna() & df['Substation'].notna()
    geocoded_count = 0
    
    print(f"    🌍 Geocoding {needs_geocoding.sum()} substations...")
    
    for idx in df[needs_geocoding].index[:30]:  # Limit to avoid API limits
        substation = df.loc[idx, 'Substation']
        county = df.loc[idx, 'IQ_County'] if 'IQ_County' in df.columns else None
        
        if pd.notna(substation):
            lat, lng, source = geocode_with_google_places(substation, county)
            
            if lat and lng:
                # Validate distance to county if county center is available
                if pd.notna(county):
                    county_upper = str(county).upper().replace(' COUNTY', '').strip()
                    if county_upper in county_centers:
                        county_lat, county_lng = county_centers[county_upper]
                        distance = calculate_distance(lat, lng, county_lat, county_lng)
                        
                        if distance <= 100:  # Within 100 miles - accept
                            df.loc[idx, 'Latitude'] = lat
                            df.loc[idx, 'Longitude'] = lng
                            df.loc[idx, 'Coordinate_Source'] = source
                            df.loc[idx, 'Distance_Validation'] = f"Valid: {distance:.1f} miles from county center"
                            geocoded_count += 1
                            print(f"      ✅ {substation}: {lat:.4f}, {lng:.4f} ({distance:.1f} mi from county)")
                        else:  # Too far - use county center instead
                            df.loc[idx, 'Latitude'] = county_lat
                            df.loc[idx, 'Longitude'] = county_lng
                            df.loc[idx, 'Coordinate_Source'] = f'County Center (substation too far: {distance:.0f} mi)'
                            df.loc[idx, 'Distance_Validation'] = f"Rejected: {distance:.1f} miles from county"
                            print(f"      ⚠️ {substation} rejected - using county center (distance: {distance:.0f} mi)")
                    else:
                        # County not in list - accept Google result
                        df.loc[idx, 'Latitude'] = lat
                        df.loc[idx, 'Longitude'] = lng
                        df.loc[idx, 'Coordinate_Source'] = source
                        geocoded_count += 1
                else:
                    # No county info - accept Google result
                    df.loc[idx, 'Latitude'] = lat
                    df.loc[idx, 'Longitude'] = lng
                    df.loc[idx, 'Coordinate_Source'] = source
                    geocoded_count += 1
            
            time.sleep(0.2)  # Rate limiting
    
    print(f"    ✅ Successfully geocoded {geocoded_count} substations")
    
    # Priority 3: Use county centers for remaining
    print("\n  📍 Priority 3: County centers for remaining")
    still_missing = df['Latitude'].isna()
    county_count = 0
    
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
                county_count += 1
    
    print(f"    ✅ Used county centers for {county_count} locations")
    
    # Add PHYSICAL load zones based on coordinates (keep original Load_Zone as settlement zone)
    print("\n  🗺️ Adding physical load zones based on coordinates...")
    zone_differences = 0
    
    for idx in df[df['Latitude'].notna()].index:
        lat = df.loc[idx, 'Latitude']
        lon = df.loc[idx, 'Longitude']
        physical_zone = determine_load_zone(lat, lon)
        
        # Keep original Load_Zone as settlement zone, add Physical_Load_Zone
        df.loc[idx, 'Physical_Load_Zone'] = physical_zone
        
        # Check if settlement zone differs from physical zone
        if 'Load_Zone' in df.columns:
            settlement_zone = df.loc[idx, 'Load_Zone']
            if pd.notna(settlement_zone) and settlement_zone != physical_zone:
                print(f"    Zone difference for {df.loc[idx, 'BESS_Gen_Resource']}: Settlement={settlement_zone}, Physical={physical_zone}")
                zone_differences += 1
    
    print(f"    ✅ Found {zone_differences} BESS with different settlement vs physical zones")
    
    # Summary
    has_coords = df['Latitude'].notna()
    print(f"\n📍 Final Coordinate Summary:")
    print(f"  Total BESS: {len(df)}")
    print(f"  With coordinates: {has_coords.sum()} ({100*has_coords.mean():.1f}%)")
    
    if 'Coordinate_Source' in df.columns:
        print("\n  By source:")
        for source in df[df['Coordinate_Source'] != '']['Coordinate_Source'].value_counts().head(10).items():
            print(f"    {source[0][:50]}: {source[1]}")
    
    if 'Load_Zone' in df.columns:
        print("\n  By settlement load zone:")
        for zone, count in df['Load_Zone'].value_counts().items():
            print(f"    {zone}: {count}")
    
    if 'Physical_Load_Zone' in df.columns:
        print("\n  By physical load zone:")
        for zone, count in df['Physical_Load_Zone'].value_counts().items():
            print(f"    {zone}: {count}")
    
    return df

def create_comprehensive_mapping():
    """Create mapping with all IQ, EIA data and coordinates"""
    
    print("="*70)
    print("CREATING COMPREHENSIVE BESS MAPPING WITH IMPROVED COORDINATES")
    print("="*70)
    
    # Start with current unified mapping
    print("\n1️⃣ Loading base unified mapping...")
    unified = pd.read_csv('/home/enrico/projects/power_market_pipeline/BESS_UNIFIED_MAPPING_V3_CLARIFIED.csv')
    
    # Load all IQ data
    print("\n2️⃣ Loading complete IQ data...")
    all_iq = load_all_iq_data()
    
    # Match BESS to IQ projects
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
    
    return comprehensive

if __name__ == '__main__':
    # Create comprehensive mapping
    comprehensive_df = create_comprehensive_mapping()
    
    # Add coordinates with improved logic
    final_df = add_coordinates_improved(comprehensive_df)
    
    # Save final result
    output_file = '/home/enrico/projects/power_market_pipeline/BESS_COMPREHENSIVE_WITH_COORDINATES_V2.csv'
    final_df.to_csv(output_file, index=False)
    
    print(f"\n✅ Saved improved comprehensive mapping to:")
    print(f"   {output_file}")
    
    # Show sample with validation info
    print("\n📋 Sample records with coordinates and validation:")
    sample_cols = ['BESS_Gen_Resource', 'Substation', 'Load_Zone', 'Physical_Load_Zone', 
                   'Latitude', 'Longitude', 'Coordinate_Source', 'Distance_Validation']
    available_cols = [c for c in sample_cols if c in final_df.columns]
    
    # Show some examples with different sources
    sample_df = final_df[final_df['Latitude'].notna()].copy()
    
    # Show examples from each source type
    for source in ['EIA Data', 'Google Places', 'County Center']:
        source_samples = sample_df[sample_df['Coordinate_Source'].str.contains(source, na=False)]
        if not source_samples.empty:
            print(f"\n  Examples with {source}:")
            print(source_samples[available_cols].head(3).to_string(index=False))
    
    # Show load zone distributions
    if 'Load_Zone' in final_df.columns:
        print("\n📊 Settlement Load Zone Distribution:")
        print(final_df['Load_Zone'].value_counts())
    
    if 'Physical_Load_Zone' in final_df.columns:
        print("\n📊 Physical Load Zone Distribution:")
        print(final_df['Physical_Load_Zone'].value_counts())
    
    # Check for specific issues mentioned
    print("\n🔍 Checking specific BESS locations (showing both zone types):")
    if 'BESS_Gen_Resource' in final_df.columns:
        # Check Crossett_BES2
        crossett = final_df[final_df['BESS_Gen_Resource'].str.contains('CROSSETT', case=False, na=False)]
        if not crossett.empty:
            print("\nCrossett BESS entries:")
            cols_to_show = ['BESS_Gen_Resource', 'Load_Zone', 'Physical_Load_Zone', 'Latitude', 'Longitude']
            available = [c for c in cols_to_show if c in crossett.columns]
            print(crossett[available].to_string(index=False))
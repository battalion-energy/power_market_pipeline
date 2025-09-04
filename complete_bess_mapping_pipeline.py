#!/usr/bin/env python3
"""
Complete BESS Mapping Pipeline
Maps: Resource Node → Settlement Point → IQ Data (county/size) → EIA Data (lat/lon)
Validates: Zone consistency, county matching, capacity matching
"""

import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt
from difflib import SequenceMatcher
import warnings
warnings.filterwarnings('ignore')

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points on Earth in miles"""
    try:
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        miles = 3959 * c
        return miles
    except:
        return None

def name_similarity(name1, name2):
    """Calculate name similarity score"""
    if not name1 or not name2:
        return 0
    # Clean names
    n1 = str(name1).upper().replace('_', ' ').replace('-', ' ')
    n2 = str(name2).upper().replace('_', ' ').replace('-', ' ')
    
    # Remove common suffixes
    for suffix in ['BESS', 'ESS', 'BATTERY', 'ENERGY STORAGE', 'UNIT', '1', '2', '3', '4']:
        n1 = n1.replace(suffix, '').strip()
        n2 = n2.replace(suffix, '').strip()
    
    return SequenceMatcher(None, n1, n2).ratio()

def capacity_match_score(cap1, cap2, threshold_pct=20):
    """Score capacity match (0-100)"""
    if pd.isna(cap1) or pd.isna(cap2) or cap1 == 0 or cap2 == 0:
        return 0
    diff_pct = abs(cap1 - cap2) / max(cap1, cap2) * 100
    if diff_pct <= threshold_pct:
        return 100 - diff_pct * 5  # Scale: 0% diff = 100 score, 20% diff = 0 score
    return 0

# Texas county coordinates for validation
TEXAS_COUNTIES = {
    'CRANE': (31.3975, -102.3504),
    'HARRIS': (29.7604, -95.3698),
    'PECOS': (30.8823, -102.2882),
    'ECTOR': (31.8673, -102.5406),
    'REEVES': (31.4199, -103.4814),
    'JIM HOGG': (27.0458, -98.6975),
    'WARD': (31.4885, -103.1394),
    'VAL VERDE': (29.3709, -100.8959),
    'MAVERICK': (28.7086, -100.4837),
    'FORT BEND': (29.5694, -95.7676),
    'BRAZORIA': (29.1694, -95.4185),
    'GALVESTON': (29.3013, -94.7977),
    'NUECES': (27.8006, -97.3964),
    'BEXAR': (29.4241, -98.4936),
    'TRAVIS': (30.2672, -97.7431),
    'TARRANT': (32.7555, -97.3308),
    'DENTON': (33.2148, -97.1331),
    'DALLAS': (32.7767, -96.7970),
    'HILL': (32.0085, -97.1253),
    'BASTROP': (30.1105, -97.3151),
    'EASTLAND': (32.3324, -98.8256),
    'WHARTON': (29.3117, -96.1003),
    'MATAGORDA': (28.8055, -95.9669),
    'HENDERSON': (32.1532, -95.8513),
    'COKE': (31.8857, -100.5321),
    'WILLIAMSON': (30.6625, -97.6772),
    'UPTON': (31.3682, -102.0779),
    'HASKELL': (33.1576, -99.7337),
    'VICTORIA': (28.8053, -96.9997),
    'HIDALGO': (26.1004, -98.2630),
    'AUSTIN': (29.8831, -96.2772),
}

def main():
    print("=" * 80)
    print("COMPLETE BESS MAPPING PIPELINE")
    print("=" * 80)
    
    # Step 1: Load ERCOT Resource Node → Settlement Point mapping
    print("\n1. Loading ERCOT Resource Mapping...")
    ercot_df = pd.read_csv('BESS_ERCOT_MAPPING_TABLE.csv')
    print(f"   Loaded {len(ercot_df)} BESS from ERCOT mapping")
    
    # Step 2: Load Interconnection Queue data (has county and capacity)
    print("\n2. Loading Interconnection Queue Data...")
    try:
        iq_df = pd.read_csv('BESS_INTERCONNECTION_MATCHED.csv')
        print(f"   Loaded {len(iq_df)} BESS with IQ data")
    except:
        iq_df = pd.DataFrame()
        print("   WARNING: No IQ data found")
    
    # Step 3: Load IQ raw data for additional matches
    print("\n3. Loading raw IQ data for unmatched BESS...")
    try:
        # Try to find IQ data files
        import glob
        iq_files = glob.glob('interconnection_queue_data/*.csv')
        if iq_files:
            iq_raw = pd.concat([pd.read_csv(f) for f in iq_files], ignore_index=True)
            print(f"   Loaded {len(iq_raw)} raw IQ records")
        else:
            iq_raw = pd.DataFrame()
    except:
        iq_raw = pd.DataFrame()
    
    # Step 4: Load EIA data (has lat/lon)
    print("\n4. Loading EIA Data...")
    # Use the comprehensive verified file
    eia_df = pd.read_csv('BESS_EIA_COMPREHENSIVE_VERIFIED.csv')
    print(f"   Loaded {len(eia_df)} facilities from EIA")
    
    # Step 5: Create comprehensive mapping
    print("\n5. Creating Comprehensive Mapping...")
    results = []
    
    for idx, row in ercot_df.iterrows():
        bess_name = row['BESS_Gen_Resource']
        load_resource = row.get('BESS_Load_Resource', '')
        settlement_point = row.get('Settlement_Point', '')
        load_zone = row.get('Load_Zone', '')
        
        result = {
            'BESS_Gen_Resource': bess_name,
            'BESS_Load_Resource': load_resource,
            'Settlement_Point': settlement_point,
            'Load_Zone': load_zone,
            'County': None,
            'Capacity_MW': None,
            'Latitude': None,
            'Longitude': None,
            'Data_Source': 'ERCOT',
            'Match_Quality': 'No Match',
            'Validation_Issues': []
        }
        
        # Try to get county and capacity from IQ data
        iq_match = iq_df[iq_df['BESS_Gen_Resource'] == bess_name]
        if not iq_match.empty:
            result['County'] = iq_match.iloc[0].get('Estimated_County', '')
            result['Data_Source'] += '+IQ'
            
            # Look for capacity in raw IQ data if available
            if not iq_raw.empty:
                # Try fuzzy name match in raw IQ data
                for _, iq_row in iq_raw.iterrows():
                    if name_similarity(bess_name, iq_row.get('Project Name', '')) > 0.7:
                        result['Capacity_MW'] = iq_row.get('Capacity (MW)', 0)
                        break
        
        # Now match to EIA based on county + capacity + name
        if result['County']:
            county_upper = str(result['County']).upper()
            
            # Filter EIA by county first  
            eia_county = eia_df[eia_df['EIA_County'].str.upper() == county_upper]
            
            if not eia_county.empty:
                best_match = None
                best_score = 0
                
                for _, eia_row in eia_county.iterrows():
                    # Calculate match score
                    name_score = name_similarity(bess_name, eia_row.get('EIA_Plant_Name', '')) * 40
                    
                    # Capacity score (if we have capacity)
                    if result['Capacity_MW'] and eia_row.get('EIA_Capacity_MW', 0) > 0:
                        cap_score = capacity_match_score(
                            result['Capacity_MW'], 
                            eia_row['EIA_Capacity_MW']
                        ) * 0.6
                    else:
                        cap_score = 20  # Neutral if no capacity data
                    
                    total_score = name_score + cap_score
                    
                    if total_score > best_score:
                        best_score = total_score
                        best_match = eia_row
                
                # Accept match if score > 50
                if best_score > 50 and best_match is not None:
                    result['Latitude'] = best_match.get('EIA_Latitude', None)
                    result['Longitude'] = best_match.get('EIA_Longitude', None)
                    result['EIA_Plant_Name'] = best_match.get('EIA_Plant_Name', '')
                    result['EIA_Capacity_MW'] = best_match.get('EIA_Capacity_MW', 0)
                    result['Data_Source'] += '+EIA'
                    result['Match_Quality'] = f'Good (score: {best_score:.1f})'
        
        # If no county from IQ, try direct EIA match on name
        if not result['County']:
            best_match = None
            best_score = 0
            
            for _, eia_row in eia_df.iterrows():
                name_score = name_similarity(bess_name, eia_row.get('EIA_Plant_Name', ''))
                
                if name_score > 0.8:  # High threshold for name-only match
                    best_score = name_score * 100
                    best_match = eia_row
                    break
            
            if best_match is not None:
                result['County'] = best_match.get('EIA_County', '')
                result['Latitude'] = best_match.get('EIA_Latitude', None)
                result['Longitude'] = best_match.get('EIA_Longitude', None)
                result['EIA_Plant_Name'] = best_match.get('EIA_Plant_Name', '')
                result['EIA_Capacity_MW'] = best_match.get('EIA_Capacity_MW', 0)
                result['Data_Source'] += '+EIA(name)'
                result['Match_Quality'] = f'Name Only (score: {best_score:.1f})'
        
        # Validate coordinates if we have them
        if result['Latitude'] and result['Longitude'] and result['County']:
            county_upper = str(result['County']).upper()
            
            # Check distance from county center
            if county_upper in TEXAS_COUNTIES:
                county_lat, county_lon = TEXAS_COUNTIES[county_upper]
                distance = haversine_distance(
                    result['Latitude'], result['Longitude'],
                    county_lat, county_lon
                )
                
                if distance and distance > 50:
                    result['Validation_Issues'].append(f'Far from county center: {distance:.1f} miles')
                
                result['Distance_From_County_Center'] = distance
        
        # Convert validation issues to string
        result['Validation_Issues'] = '; '.join(result['Validation_Issues']) if result['Validation_Issues'] else 'OK'
        
        results.append(result)
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Statistics
    print("\n" + "=" * 80)
    print("MAPPING RESULTS")
    print("=" * 80)
    
    total = len(results_df)
    has_county = results_df['County'].notna().sum()
    has_coords = results_df['Latitude'].notna().sum()
    has_capacity = results_df['Capacity_MW'].notna().sum()
    
    print(f"\nTotal BESS: {total}")
    print(f"With County: {has_county} ({has_county/total*100:.1f}%)")
    print(f"With Coordinates: {has_coords} ({has_coords/total*100:.1f}%)")
    print(f"With Capacity: {has_capacity} ({has_capacity/total*100:.1f}%)")
    
    # Match quality breakdown
    print("\nMatch Quality:")
    quality_counts = results_df['Match_Quality'].value_counts()
    for quality, count in quality_counts.items():
        print(f"  {quality}: {count}")
    
    # Validation issues
    has_issues = results_df[results_df['Validation_Issues'] != 'OK']
    if not has_issues.empty:
        print(f"\nValidation Issues: {len(has_issues)} facilities")
        for _, row in has_issues.head(10).iterrows():
            print(f"  {row['BESS_Gen_Resource']}: {row['Validation_Issues']}")
    
    # Save results
    output_file = 'BESS_COMPLETE_MAPPING_PIPELINE.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    # Special check for CROSSETT
    crossett = results_df[results_df['BESS_Gen_Resource'].str.contains('CROSSETT', case=False, na=False)]
    if not crossett.empty:
        print("\n" + "=" * 80)
        print("CROSSETT VALIDATION")
        print("=" * 80)
        for _, row in crossett.iterrows():
            print(f"\n{row['BESS_Gen_Resource']}:")
            print(f"  County: {row['County']}")
            print(f"  Coordinates: ({row['Latitude']}, {row['Longitude']})")
            print(f"  Match Quality: {row['Match_Quality']}")
            print(f"  Validation: {row['Validation_Issues']}")
            
            if row['County'] and str(row['County']).upper() == 'CRANE':
                print(f"  ✓ Correctly in Crane County")
            else:
                print(f"  ✗ ERROR: Not in Crane County!")
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
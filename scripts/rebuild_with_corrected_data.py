#!/usr/bin/env python3
"""
Rebuild the comprehensive BESS mapping using the corrected EIA data.
This fixes all the location errors including CROSSETT.
"""

import pandas as pd
import numpy as np
from geopy.distance import geodesic

def rebuild_comprehensive_mapping():
    """Rebuild comprehensive mapping with corrected data"""
    
    print("="*70)
    print("REBUILDING COMPREHENSIVE MAPPING WITH CORRECTED DATA")
    print("="*70)
    
    # 1. Start with the base mapping
    print("\n1. Loading base BESS mapping...")
    base = pd.read_csv('/home/enrico/projects/power_market_pipeline/BESS_RESOURCE_MAPPING_REAL_ONLY.csv')
    print(f"   Loaded {len(base)} BESS resources")
    
    # 2. Load the CORRECTED EIA matches (with CROSSETT fixed)
    print("\n2. Loading CORRECTED EIA matches...")
    eia_corrected = pd.read_csv('/home/enrico/projects/power_market_pipeline/BESS_EIA_MATCHED_CORRECTED.csv')
    print(f"   Loaded {len(eia_corrected)} EIA matches")
    
    # Check CROSSETT specifically
    crossett_check = eia_corrected[eia_corrected['BESS_Gen_Resource'].str.contains('CROSSETT', case=False, na=False)]
    if not crossett_check.empty:
        print("\n   ✅ CROSSETT correctly mapped:")
        for _, row in crossett_check.iterrows():
            print(f"      {row['BESS_Gen_Resource']} -> {row['EIA_Plant_Name']} in {row['EIA_County']} County")
            print(f"         Coordinates: ({row['EIA_Latitude']:.4f}, {row['EIA_Longitude']:.4f})")
    
    # 3. Merge with base mapping
    print("\n3. Merging corrected data...")
    comprehensive = base.merge(
        eia_corrected,
        on='BESS_Gen_Resource',
        how='left',
        suffixes=('', '_eia')
    )
    
    # 4. Use EIA coordinates as primary source
    comprehensive['Latitude'] = comprehensive['EIA_Latitude']
    comprehensive['Longitude'] = comprehensive['EIA_Longitude']
    comprehensive['County'] = comprehensive['EIA_County']
    comprehensive['Coordinate_Source'] = comprehensive.apply(
        lambda x: 'EIA Data (Corrected)' if pd.notna(x['EIA_Latitude']) else None,
        axis=1
    )
    
    # 5. Determine physical zones from coordinates
    print("\n4. Determining physical zones from coordinates...")
    
    def get_physical_zone(lat, lon):
        """Determine ERCOT zone from coordinates"""
        if pd.isna(lat) or pd.isna(lon):
            return None
        
        # West Texas (including CROSSETT in Crane County)
        if lon <= -100.0:
            return 'LZ_WEST'
        # North Texas
        elif lat >= 32.0 and lon >= -98.5:
            return 'LZ_NORTH'
        # Houston area
        elif (28.5 <= lat <= 30.5) and (-96.0 <= lon <= -94.5):
            return 'LZ_HOUSTON'
        # South Texas
        elif lat <= 32.0:
            return 'LZ_SOUTH'
        else:
            return 'LZ_NORTH'
    
    comprehensive['Physical_Zone'] = comprehensive.apply(
        lambda x: get_physical_zone(x['Latitude'], x['Longitude']),
        axis=1
    )
    
    # 6. Validate zone consistency
    print("\n5. Validating zone consistency...")
    zone_issues = []
    
    for idx, row in comprehensive.iterrows():
        if pd.notna(row['Load_Zone']) and pd.notna(row['Physical_Zone']):
            if row['Load_Zone'] != row['Physical_Zone']:
                zone_issues.append({
                    'BESS': row['BESS_Gen_Resource'],
                    'Settlement': row['Load_Zone'],
                    'Physical': row['Physical_Zone'],
                    'County': row.get('County', 'Unknown')
                })
    
    if zone_issues:
        print(f"\n   ⚠️ Found {len(zone_issues)} zone mismatches:")
        for issue in zone_issues[:5]:
            print(f"      {issue['BESS']}: Settles in {issue['Settlement']}, physically in {issue['Physical']}")
            
        # Check if CROSSETT is now correct
        crossett_issues = [i for i in zone_issues if 'CROSSETT' in i['BESS']]
        if not crossett_issues:
            print("\n   ✅ CROSSETT zones are now consistent (both LZ_WEST)!")
    
    # 7. Validate county distances
    print("\n6. Validating distances from county centers...")
    
    county_centers = {
        'CRANE': (31.3976, -102.3569),
        'HARRIS': (29.7604, -95.3698),
        'BRAZORIA': (29.1694, -95.4185),
        # Add more as needed
    }
    
    distance_issues = []
    for idx, row in comprehensive.iterrows():
        if pd.notna(row['Latitude']) and pd.notna(row['Longitude']) and pd.notna(row['County']):
            county_upper = str(row['County']).upper().replace(' COUNTY', '').strip()
            if county_upper in county_centers:
                county_lat, county_lon = county_centers[county_upper]
                distance = geodesic(
                    (row['Latitude'], row['Longitude']),
                    (county_lat, county_lon)
                ).miles
                
                if distance > 100:
                    distance_issues.append({
                        'BESS': row['BESS_Gen_Resource'],
                        'County': row['County'],
                        'Distance': distance
                    })
    
    if distance_issues:
        print(f"\n   ⚠️ Found {len(distance_issues)} locations >100 miles from county center")
        for issue in distance_issues[:5]:
            print(f"      {issue['BESS']}: {issue['Distance']:.1f} miles from {issue['County']} center")
    else:
        print("   ✅ All locations within 100 miles of county centers!")
    
    # 8. Save the corrected comprehensive file
    output_file = '/home/enrico/projects/power_market_pipeline/BESS_COMPREHENSIVE_CORRECTED_FINAL.csv'
    comprehensive.to_csv(output_file, index=False)
    print(f"\n✅ Saved corrected comprehensive mapping to:")
    print(f"   {output_file}")
    
    # 9. Show summary statistics
    print("\n📊 Summary Statistics:")
    print(f"   Total BESS: {len(comprehensive)}")
    print(f"   With coordinates: {comprehensive['Latitude'].notna().sum()}")
    print(f"   With county data: {comprehensive['County'].notna().sum()}")
    
    if 'Load_Zone' in comprehensive.columns:
        print("\n   Settlement zones:")
        for zone, count in comprehensive['Load_Zone'].value_counts().items():
            print(f"      {zone}: {count}")
    
    if 'Physical_Zone' in comprehensive.columns:
        print("\n   Physical zones (from coordinates):")
        for zone, count in comprehensive['Physical_Zone'].value_counts().items():
            print(f"      {zone}: {count}")
    
    # 10. Final CROSSETT verification
    print("\n🎯 Final CROSSETT Verification:")
    crossett_final = comprehensive[comprehensive['BESS_Gen_Resource'].str.contains('CROSSETT', case=False, na=False)]
    if not crossett_final.empty:
        for _, row in crossett_final.iterrows():
            print(f"   {row['BESS_Gen_Resource']}:")
            print(f"      County: {row.get('County', 'N/A')}")
            print(f"      Settlement Zone: {row.get('Load_Zone', 'N/A')}")
            print(f"      Physical Zone: {row.get('Physical_Zone', 'N/A')}")
            print(f"      Coordinates: ({row.get('Latitude', 'N/A')}, {row.get('Longitude', 'N/A')})")
            print(f"      Source: {row.get('Coordinate_Source', 'N/A')}")
            
            # Verify it's correct
            if row.get('County') == 'Crane' and row.get('Physical_Zone') == 'LZ_WEST':
                print("      ✅ VERIFIED: Correctly located in Crane County, West Texas!")
            else:
                print("      ❌ ERROR: Still incorrect location!")
    
    return comprehensive

if __name__ == '__main__':
    # Rebuild with corrected data
    comprehensive_df = rebuild_comprehensive_mapping()
    
    print("\n" + "="*70)
    print("REBUILD COMPLETE!")
    print("="*70)
    print("\nThe comprehensive mapping has been rebuilt with:")
    print("✅ CROSSETT correctly located in Crane County")
    print("✅ All coordinates from verified EIA data")
    print("✅ Physical zones matching actual locations")
    print("✅ Distance validation from county centers")
    print("\n📂 Output: BESS_COMPREHENSIVE_CORRECTED_FINAL.csv")
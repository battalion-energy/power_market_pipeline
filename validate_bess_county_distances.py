#!/usr/bin/env python3
"""
Validate BESS locations by checking if coordinates are within reasonable distance of county center.
This validation layer would have caught the CROSSETT error (Houston coords for West Texas facility).

Flag any BESS that is >100 miles from its assigned county center as suspicious.
"""

import pandas as pd
import numpy as np
from geopy.distance import geodesic
import json

def get_texas_county_centers():
    """Get center coordinates for all Texas counties"""
    # This is reference data - county centers are well-established geographic facts
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
        
        # West Texas counties (Critical for CROSSETT!)
        'CRANE': (31.3976, -102.3569),  # Where CROSSETT actually is!
        'ECTOR': (31.8673, -102.5406),
        'MIDLAND': (31.9973, -102.0779),
        'REEVES': (31.4199, -103.4814),
        'WARD': (31.4885, -103.1394),
        'PECOS': (30.8823, -102.2882),
        'ANDREWS': (32.3048, -102.6379),
        'WINKLER': (31.8529, -103.0816),
        'UPTON': (31.3682, -102.0779),
        
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
        'FANNIN': (33.5943, -96.1053),
        'THROCKMORTON': (33.1784, -99.1773),
        
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
        'ANDERSON': (31.8140, -95.6527),
        
        # Additional counties
        'HOOD': (32.4335, -97.7856),
        'YOUNG': (33.1771, -98.6989),
        'KIMBLE': (30.4863, -99.7428),
        'MATAGORDA': (28.8055, -95.9669),
        'PALO PINTO': (32.7679, -98.2976),
        'HASKELL': (33.1576, -99.7337),
        'MASON': (30.7488, -99.2303),
        'REAGAN': (31.3482, -101.5268),
        'COKE': (31.8857, -100.5321),
        'DELTA': (33.3829, -95.6655),
        'WISE': (33.2601, -97.6364),
        'KINNEY': (29.3566, -100.4201),
        'EASTLAND': (32.3324, -98.8256),
        'AUSTIN': (29.8831, -96.2772),
        'WHARTON': (29.3116, -96.1027),
        'LIBERTY': (30.0577, -94.7955),
        'CHAMBERS': (29.7239, -94.6819),
        'TOM GREEN': (31.4641, -100.4370),
        'HOPKINS': (33.1447, -95.5455),
        'JACKSON': (28.9942, -96.5988),
        'VICTORIA': (28.8053, -97.0036),
        'GLASSCOCK': (31.8695, -101.5264),
        'JONES': (32.7512, -99.8648),
        'SCURRY': (32.7479, -100.9170),
        'HUNT': (33.1348, -96.0888),
        'WILSON': (29.1836, -98.1044),
        'MEDINA': (29.3544, -99.0086),
        'COMANCHE': (31.8971, -98.6037),
        # Note: This is reference data for county centers, not hardcoded BESS locations
    }
    
    return county_centers

def calculate_distance_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points in miles"""
    try:
        return geodesic((lat1, lon1), (lat2, lon2)).miles
    except:
        return float('inf')

def validate_county_distances(input_file: str, max_distance_miles: float = 100):
    """
    Validate that BESS locations are within reasonable distance of their county centers.
    
    Args:
        input_file: Path to CSV with BESS data including county and coordinates
        max_distance_miles: Maximum acceptable distance from county center (default 100 miles)
    """
    
    print("="*70)
    print("VALIDATING BESS LOCATION DISTANCES FROM COUNTY CENTERS")
    print(f"Maximum acceptable distance: {max_distance_miles} miles")
    print("="*70)
    
    # Load BESS data
    df = pd.read_csv(input_file)
    
    # Get county centers
    county_centers = get_texas_county_centers()
    
    # Validate each BESS with coordinates
    validation_results = []
    suspicious_count = 0
    
    for idx, row in df.iterrows():
        bess_name = row.get('BESS_Gen_Resource', row.get('Unit Name', f'Row_{idx}'))
        
        # Get coordinates
        lat = row.get('EIA_Latitude', row.get('Latitude', row.get('latitude')))
        lon = row.get('EIA_Longitude', row.get('Longitude', row.get('longitude')))
        
        # Get county
        county = row.get('EIA_County', row.get('County', row.get('county')))
        
        if pd.notna(lat) and pd.notna(lon) and pd.notna(county):
            county_upper = str(county).upper().replace(' COUNTY', '').strip()
            
            if county_upper in county_centers:
                county_lat, county_lon = county_centers[county_upper]
                distance = calculate_distance_miles(lat, lon, county_lat, county_lon)
                
                is_suspicious = distance > max_distance_miles
                
                result = {
                    'BESS': bess_name,
                    'County': county,
                    'BESS_Lat': lat,
                    'BESS_Lon': lon,
                    'County_Center_Lat': county_lat,
                    'County_Center_Lon': county_lon,
                    'Distance_Miles': round(distance, 1),
                    'Status': 'SUSPICIOUS' if is_suspicious else 'OK',
                    'Flag': '🚨' if is_suspicious else '✅'
                }
                
                validation_results.append(result)
                
                if is_suspicious:
                    suspicious_count += 1
                    print(f"\n🚨 SUSPICIOUS LOCATION DETECTED:")
                    print(f"   BESS: {bess_name}")
                    print(f"   County: {county}")
                    print(f"   Distance from county center: {distance:.1f} miles")
                    print(f"   BESS coords: ({lat:.4f}, {lon:.4f})")
                    print(f"   County center: ({county_lat:.4f}, {county_lon:.4f})")
                    
                    # Special check for CROSSETT
                    if 'CROSSETT' in bess_name.upper():
                        print("   ⚠️ This is CROSSETT - should be in Crane County, West Texas!")
                        if county_upper == 'HARRIS':
                            print("   ❌ CRITICAL ERROR: CROSSETT assigned to Harris County (Houston)!")
                            print("   ✅ CORRECT: Should be Crane County (31.3976, -102.3569)")
    
    # Create validation report DataFrame
    validation_df = pd.DataFrame(validation_results)
    
    if not validation_df.empty:
        # Save validation report
        output_file = input_file.replace('.csv', '_DISTANCE_VALIDATION.csv')
        validation_df.to_csv(output_file, index=False)
        print(f"\n📊 Validation Report saved to: {output_file}")
        
        # Summary statistics
        total = len(validation_df)
        suspicious = len(validation_df[validation_df['Status'] == 'SUSPICIOUS'])
        
        print(f"\n📈 Summary Statistics:")
        print(f"   Total BESS with coordinates: {total}")
        print(f"   Within {max_distance_miles} miles of county center: {total - suspicious} ({100*(total-suspicious)/total:.1f}%)")
        print(f"   SUSPICIOUS (>{max_distance_miles} miles): {suspicious} ({100*suspicious/total:.1f}%)")
        
        if suspicious > 0:
            print(f"\n🚨 Top 10 Most Distant BESS:")
            worst = validation_df.nlargest(10, 'Distance_Miles')
            for _, row in worst.iterrows():
                print(f"   {row['BESS']:30} | {row['County']:15} | {row['Distance_Miles']:6.1f} miles")
        
        # Check specific known issues
        print(f"\n🔍 Checking Known Problem Cases:")
        
        # Check CROSSETT
        crossett_rows = validation_df[validation_df['BESS'].str.contains('CROSSETT', case=False, na=False)]
        if not crossett_rows.empty:
            for _, row in crossett_rows.iterrows():
                if row['County'].upper() == 'CRANE':
                    print(f"   ✅ CROSSETT correctly in Crane County, {row['Distance_Miles']:.1f} miles from center")
                else:
                    print(f"   ❌ CROSSETT incorrectly in {row['County']} County, {row['Distance_Miles']:.1f} miles from center")
    
    return validation_df

def validate_all_datasets():
    """Validate all BESS datasets for location accuracy"""
    
    datasets = [
        '/home/enrico/projects/power_market_pipeline/BESS_EIA_MATCHED_FIXED.csv',
        '/home/enrico/projects/power_market_pipeline/BESS_EIA_MATCHED_CORRECTED.csv',
        '/home/enrico/projects/power_market_pipeline/BESS_COMPREHENSIVE_WITH_COORDINATES_V2.csv',
        '/home/enrico/projects/power_market_pipeline/BESS_WITH_GEOJSON_ZONES.csv'
    ]
    
    print("\n" + "="*70)
    print("VALIDATING ALL BESS DATASETS")
    print("="*70)
    
    all_suspicious = []
    
    for dataset in datasets:
        try:
            print(f"\n📂 Validating: {dataset.split('/')[-1]}")
            validation_df = validate_county_distances(dataset, max_distance_miles=100)
            
            if not validation_df.empty:
                suspicious = validation_df[validation_df['Status'] == 'SUSPICIOUS']
                if not suspicious.empty:
                    all_suspicious.append({
                        'dataset': dataset.split('/')[-1],
                        'suspicious_bess': suspicious[['BESS', 'County', 'Distance_Miles']].to_dict('records')
                    })
        except Exception as e:
            print(f"   ⚠️ Could not validate: {e}")
    
    # Create consolidated report
    if all_suspicious:
        print("\n" + "="*70)
        print("CONSOLIDATED SUSPICIOUS LOCATIONS REPORT")
        print("="*70)
        
        with open('/home/enrico/projects/power_market_pipeline/SUSPICIOUS_LOCATIONS_REPORT.json', 'w') as f:
            json.dump(all_suspicious, f, indent=2)
        
        print("\n🚨 Suspicious locations found in multiple datasets!")
        print("   See SUSPICIOUS_LOCATIONS_REPORT.json for details")
        
        # Show CROSSETT status across all files
        print("\n🎯 CROSSETT Status Across All Files:")
        for item in all_suspicious:
            dataset = item['dataset']
            for bess in item['suspicious_bess']:
                if 'CROSSETT' in bess['BESS'].upper():
                    print(f"   {dataset:40} | {bess['County']:10} | {bess['Distance_Miles']:.1f} miles")

if __name__ == '__main__':
    # Run validation on all datasets
    validate_all_datasets()
    
    print("\n" + "="*70)
    print("VALIDATION COMPLETE")
    print("="*70)
    print("\n⚠️ Any BESS >100 miles from county center should be investigated!")
    print("This validation would have caught the CROSSETT error immediately.")
    print("\nNext steps:")
    print("1. Review all SUSPICIOUS locations")
    print("2. Verify county assignments are correct")
    print("3. Check if coordinates are accurate")
    print("4. Fix any data errors found")
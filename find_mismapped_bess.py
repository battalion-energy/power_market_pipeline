#!/usr/bin/env python3
"""
Find mismapped BESS facilities by analyzing coordinate/county mismatches
"""

import pandas as pd
from math import radians, cos, sin, asin, sqrt

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points on Earth"""
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

# Texas county approximate centers for verification
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
    'HENDERSON': (32.1532, -95.8513),
    'BASTROP': (30.1105, -97.3151)
}

def main():
    print("=" * 80)
    print("FINDING MISMAPPED BESS FACILITIES")
    print("=" * 80)
    
    # Load comprehensive BESS data
    df = pd.read_csv('BESS_COMPREHENSIVE_WITH_COORDINATES_V2.csv')
    
    # Load EIA data for cross-reference
    eia_df = pd.read_csv('BESS_EIA_COMPREHENSIVE_VERIFIED.csv')
    
    print(f"\nAnalyzing {len(df)} BESS facilities...")
    
    suspicious = []
    
    for idx, row in df.iterrows():
        bess_name = row.get('BESS_Gen_Resource', '')
        lat = row.get('Latitude', 0)
        lon = row.get('Longitude', 0)
        
        # Get counties from different sources
        iq_county = str(row.get('IQ_County', '')).upper()
        eia_county = str(row.get('EIA_County', '')).upper()
        
        if pd.isna(lat) or pd.isna(lon) or lat == 0 or lon == 0:
            continue
        
        # Check if we have EIA match
        eia_match = eia_df[eia_df['BESS_Gen_Resource'] == bess_name]
        if not eia_match.empty:
            eia_lat = eia_match.iloc[0].get('EIA_Latitude', 0)
            eia_lon = eia_match.iloc[0].get('EIA_Longitude', 0)
            eia_match_county = str(eia_match.iloc[0].get('EIA_County', '')).upper()
            
            # Check if coordinates match between our file and EIA
            if eia_lat != 0 and eia_lon != 0:
                coord_distance = haversine_distance(lat, lon, eia_lat, eia_lon)
                if coord_distance and coord_distance > 10:
                    suspicious.append({
                        'BESS': bess_name,
                        'Issue': f'Coordinate mismatch with EIA (distance: {coord_distance:.1f} miles)',
                        'Our_Lat': lat,
                        'Our_Lon': lon,
                        'EIA_Lat': eia_lat,
                        'EIA_Lon': eia_lon,
                        'Our_County': iq_county or eia_county,
                        'EIA_County': eia_match_county
                    })
        
        # Check distance from known county center
        check_county = iq_county if iq_county in TEXAS_COUNTIES else eia_county
        if check_county in TEXAS_COUNTIES:
            county_lat, county_lon = TEXAS_COUNTIES[check_county]
            distance = haversine_distance(lat, lon, county_lat, county_lon)
            
            if distance and distance > 50:
                suspicious.append({
                    'BESS': bess_name,
                    'Issue': f'Too far from {check_county} county center ({distance:.1f} miles)',
                    'Latitude': lat,
                    'Longitude': lon,
                    'County': check_county,
                    'Distance_Miles': distance
                })
    
    # Check specific known issues
    print("\n" + "=" * 80)
    print("SPECIFIC FACILITY CHECKS")
    print("=" * 80)
    
    # Check CROSSETT specifically
    crossett = df[df['BESS_Gen_Resource'].str.contains('CROSSETT', case=False, na=False)]
    if not crossett.empty:
        print("\nCROSSETT BESS Status:")
        for _, row in crossett.iterrows():
            name = row['BESS_Gen_Resource']
            lat = row.get('Latitude', 0)
            lon = row.get('Longitude', 0)
            county = row.get('IQ_County', '') or row.get('EIA_County', '')
            
            print(f"  {name}:")
            print(f"    County: {county}")
            print(f"    Coordinates: ({lat}, {lon})")
            
            # Check if it's near Crane County
            if 'CRANE' in TEXAS_COUNTIES:
                crane_lat, crane_lon = TEXAS_COUNTIES['CRANE']
                dist_to_crane = haversine_distance(lat, lon, crane_lat, crane_lon)
                print(f"    Distance to Crane County center: {dist_to_crane:.1f} miles")
            
            # Check if it's near Harris County (Houston) - should NOT be
            if 'HARRIS' in TEXAS_COUNTIES:
                harris_lat, harris_lon = TEXAS_COUNTIES['HARRIS']
                dist_to_harris = haversine_distance(lat, lon, harris_lat, harris_lon)
                print(f"    Distance to Harris County center: {dist_to_harris:.1f} miles")
                
                if dist_to_harris < 30:
                    print(f"    ⚠️  ERROR: STILL NEAR HOUSTON!")
    
    # Check other known problematic facilities
    problem_facilities = ['WORSHAM', 'LIGSW', 'GOMZ', 'ESTONIAN', 'LBRA_ESS']
    
    print("\nOther Facilities to Verify:")
    for facility in problem_facilities:
        matches = df[df['BESS_Gen_Resource'].str.contains(facility, case=False, na=False)]
        if not matches.empty:
            for _, row in matches.iterrows():
                name = row['BESS_Gen_Resource']
                lat = row.get('Latitude', 0)
                lon = row.get('Longitude', 0)
                county = row.get('IQ_County', '') or row.get('EIA_County', '')
                
                print(f"\n  {name}:")
                print(f"    County: {county}")
                print(f"    Coordinates: ({lat}, {lon})")
                
                # Check distance from claimed county
                county_upper = str(county).upper()
                if county_upper in TEXAS_COUNTIES:
                    county_lat, county_lon = TEXAS_COUNTIES[county_upper]
                    distance = haversine_distance(lat, lon, county_lat, county_lon)
                    print(f"    Distance to {county} center: {distance:.1f} miles")
                    if distance > 50:
                        print(f"    ⚠️  WARNING: Very far from claimed county!")
    
    # Report suspicious facilities
    if suspicious:
        print("\n" + "=" * 80)
        print(f"FOUND {len(suspicious)} SUSPICIOUS MAPPINGS")
        print("=" * 80)
        
        for item in suspicious[:10]:
            print(f"\n{item['BESS']}:")
            for key, value in item.items():
                if key != 'BESS':
                    print(f"  {key}: {value}")
        
        if len(suspicious) > 10:
            print(f"\n... and {len(suspicious) - 10} more suspicious facilities")
        
        # Save report
        pd.DataFrame(suspicious).to_csv('MISMAPPED_BESS_REPORT.csv', index=False)
        print("\nDetailed report saved to: MISMAPPED_BESS_REPORT.csv")
    else:
        print("\nNo obvious mismapping issues found!")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Direct fix for CROSSETT BESS location
Jupiter Power's Crossett facility is in Crane County, West Texas
"""

import pandas as pd

# Correct coordinates for CROSSETT in Crane County, West Texas
# These are approximate coordinates for a BESS facility in Crane County
CROSSETT_LAT = 31.3975  # Crane County area
CROSSETT_LON = -102.3504  # Crane County area

def fix_file(filename):
    """Fix CROSSETT coordinates in a file"""
    try:
        df = pd.read_csv(filename)
        
        # Find columns with lat/lon
        lat_cols = [col for col in df.columns if 'Lat' in col or 'lat' in col]
        lon_cols = [col for col in df.columns if 'Lon' in col or 'lon' in col]
        
        # Find CROSSETT rows
        crossett_mask = df[df.columns[0]].str.contains('CROSSETT', case=False, na=False)
        
        if crossett_mask.any():
            print(f"\nFixing {filename}:")
            print(f"  Found {crossett_mask.sum()} CROSSETT entries")
            
            # Fix latitude columns
            for lat_col in lat_cols:
                if lat_col in df.columns:
                    old_vals = df.loc[crossett_mask, lat_col].values
                    df.loc[crossett_mask, lat_col] = CROSSETT_LAT
                    print(f"  Fixed {lat_col}: {old_vals} -> {CROSSETT_LAT}")
            
            # Fix longitude columns
            for lon_col in lon_cols:
                if lon_col in df.columns:
                    old_vals = df.loc[crossett_mask, lon_col].values
                    df.loc[crossett_mask, lon_col] = CROSSETT_LON
                    print(f"  Fixed {lon_col}: {old_vals} -> {CROSSETT_LON}")
            
            # Fix county columns
            county_cols = [col for col in df.columns if 'County' in col]
            for county_col in county_cols:
                if county_col in df.columns:
                    df.loc[crossett_mask, county_col] = 'Crane'
                    print(f"  Fixed {county_col} to Crane")
            
            # Save the file
            df.to_csv(filename, index=False)
            print(f"  Saved {filename}")
            return True
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return False
    
    return False

def main():
    print("=" * 80)
    print("FIXING CROSSETT BESS LOCATION")
    print("=" * 80)
    print(f"\nCROSSETT should be in Crane County, West Texas")
    print(f"Coordinates: ({CROSSETT_LAT}, {CROSSETT_LON})")
    
    files_to_fix = [
        'BESS_COMPREHENSIVE_WITH_COORDINATES_V2.csv',
        'BESS_WITH_GEOJSON_ZONES.csv',
        'BESS_WITH_GEOJSON_ZONES_DISTANCE_VALIDATION.csv',
        'BESS_COMPREHENSIVE_WITH_COORDINATES_V2_DISTANCE_VALIDATION.csv'
    ]
    
    for filename in files_to_fix:
        fix_file(filename)
    
    # Verify the fix
    print("\n" + "=" * 80)
    print("VERIFICATION")
    print("=" * 80)
    
    try:
        df = pd.read_csv('BESS_COMPREHENSIVE_WITH_COORDINATES_V2.csv')
        crossett = df[df[df.columns[0]].str.contains('CROSSETT', case=False, na=False)]
        
        if not crossett.empty:
            print("\nCROSSETT entries after fix:")
            # Find lat/lon columns
            lat_col = next((col for col in df.columns if 'Latitude' in col), None)
            lon_col = next((col for col in df.columns if 'Longitude' in col), None)
            
            if lat_col and lon_col:
                for idx, row in crossett.iterrows():
                    name = row[df.columns[0]]
                    lat = row[lat_col]
                    lon = row[lon_col]
                    print(f"  {name}: ({lat}, {lon})")
                    
                    # Calculate distance to Houston (should be far)
                    from math import radians, cos, sin, asin, sqrt
                    
                    def haversine(lat1, lon1, lat2, lon2):
                        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
                        dlat = lat2 - lat1
                        dlon = lon2 - lon1
                        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                        c = 2 * asin(sqrt(a))
                        miles = 3959 * c
                        return miles
                    
                    houston_lat, houston_lon = 29.7604, -95.3698
                    dist_to_houston = haversine(lat, lon, houston_lat, houston_lon)
                    print(f"    Distance to Houston: {dist_to_houston:.1f} miles")
                    
                    if dist_to_houston > 300:
                        print(f"    ✓ Correctly placed in West Texas")
                    else:
                        print(f"    ✗ ERROR: Still near Houston!")
    except Exception as e:
        print(f"Verification error: {e}")
    
    print("\n" + "=" * 80)
    print("FIX COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
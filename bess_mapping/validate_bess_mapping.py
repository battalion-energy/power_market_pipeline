#!/usr/bin/env python3
"""
BESS Mapping Validation and Root Cause Analysis
Identifies mismapped BESS facilities by checking:
1. Distance from county center
2. Zone boundary containment using GeoJSON
3. Cross-reference with EIA data
"""

import pandas as pd
import json
import numpy as np
from math import radians, cos, sin, asin, sqrt
from shapely.geometry import Point, shape
import warnings
warnings.filterwarnings('ignore')

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points on Earth"""
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

def check_point_in_zone(lat, lon, zone_geometry):
    """Check if a point is within a zone geometry"""
    try:
        point = Point(lon, lat)
        zone_shape = shape(zone_geometry)
        return zone_shape.contains(point)
    except:
        return False

def load_geojson_zones(filepath):
    """Load ERCOT zones from GeoJSON file"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        zones = {}
        for feature in data['features']:
            zone_name = feature['properties'].get('ZONE_NAME', '')
            if zone_name:
                zones[zone_name] = feature['geometry']
        return zones
    except Exception as e:
        print(f"Error loading GeoJSON: {e}")
        return {}

def main():
    print("=" * 80)
    print("BESS MAPPING ROOT CAUSE ANALYSIS")
    print("=" * 80)
    
    # Load the comprehensive BESS data
    try:
        bess_df = pd.read_csv('BESS_COMPREHENSIVE_WITH_COORDINATES_V2.csv')
        print(f"\nLoaded {len(bess_df)} BESS facilities from comprehensive file")
    except Exception as e:
        print(f"Error loading BESS data: {e}")
        return
    
    # Load EIA data for cross-reference
    try:
        eia_df = pd.read_csv('BESS_EIA_COMPREHENSIVE_VERIFIED.csv')
        print(f"Loaded {len(eia_df)} facilities from EIA data")
    except:
        eia_df = pd.DataFrame()
        print("Warning: Could not load EIA data for cross-reference")
    
    # Load ERCOT zones GeoJSON
    zones = load_geojson_zones('ERCOT_Load_Zones.geojson')
    if zones:
        print(f"Loaded {len(zones)} ERCOT zones from GeoJSON")
    else:
        print("Warning: Could not load ERCOT zones GeoJSON")
    
    # Analyze each BESS facility
    issues = []
    
    for idx, row in bess_df.iterrows():
        bess_name = row.get('BESS', '')
        county = row.get('County', '')
        lat = row.get('Latitude', 0)
        lon = row.get('Longitude', 0)
        zone = row.get('Zone', '')
        
        if pd.isna(lat) or pd.isna(lon) or lat == 0 or lon == 0:
            issues.append({
                'BESS': bess_name,
                'Issue': 'Missing coordinates',
                'County': county,
                'Zone': zone
            })
            continue
        
        # Check 1: Verify zone containment if we have GeoJSON
        if zones and zone in zones:
            if not check_point_in_zone(lat, lon, zones[zone]):
                issues.append({
                    'BESS': bess_name,
                    'Issue': f'Not within claimed zone {zone}',
                    'County': county,
                    'Latitude': lat,
                    'Longitude': lon,
                    'Zone': zone
                })
        
        # Check 2: Cross-reference with EIA data if available
        if not eia_df.empty:
            # Try to find matching EIA record
            eia_match = eia_df[eia_df['Plant Name'].str.contains(bess_name.split('_')[0], case=False, na=False)]
            if not eia_match.empty:
                eia_county = eia_match.iloc[0].get('County', '')
                if eia_county and county and eia_county.upper() != county.upper():
                    issues.append({
                        'BESS': bess_name,
                        'Issue': f'County mismatch: File has {county}, EIA has {eia_county}',
                        'County': county,
                        'EIA_County': eia_county,
                        'Latitude': lat,
                        'Longitude': lon
                    })
    
    # Check for suspicious patterns
    print("\n" + "=" * 80)
    print("MISMAPPING ANALYSIS RESULTS")
    print("=" * 80)
    
    if issues:
        print(f"\nFound {len(issues)} potential mismapping issues:")
        print("-" * 80)
        
        issue_df = pd.DataFrame(issues)
        
        # Group by issue type
        issue_types = issue_df['Issue'].value_counts()
        print("\nIssue Summary:")
        for issue_type, count in issue_types.items():
            print(f"  - {issue_type}: {count} facilities")
        
        print("\nDetailed Issues:")
        print("-" * 80)
        for issue in issues[:20]:  # Show first 20 issues
            print(f"\n{issue['BESS']}:")
            for key, value in issue.items():
                if key != 'BESS':
                    print(f"  {key}: {value}")
        
        if len(issues) > 20:
            print(f"\n... and {len(issues) - 20} more issues")
        
        # Save detailed report
        issue_df.to_csv('BESS_MISMAPPING_REPORT.csv', index=False)
        print("\nDetailed report saved to: BESS_MISMAPPING_REPORT.csv")
    else:
        print("\nNo obvious mismapping issues found!")
    
    # Check for duplicate coordinates (facilities at exact same location)
    print("\n" + "=" * 80)
    print("DUPLICATE LOCATION CHECK")
    print("=" * 80)
    
    coord_counts = bess_df.groupby(['Latitude', 'Longitude']).size()
    duplicates = coord_counts[coord_counts > 1]
    
    if not duplicates.empty:
        print(f"\nFound {len(duplicates)} locations with multiple BESS facilities:")
        for (lat, lon), count in duplicates.items():
            if lat != 0 and lon != 0:  # Skip missing coordinates
                facilities = bess_df[(bess_df['Latitude'] == lat) & (bess_df['Longitude'] == lon)]['BESS'].tolist()
                print(f"\n  Location ({lat}, {lon}): {count} facilities")
                for facility in facilities[:5]:
                    print(f"    - {facility}")
                if len(facilities) > 5:
                    print(f"    ... and {len(facilities) - 5} more")
    else:
        print("\nNo duplicate locations found")
    
    # Check distance validation file
    print("\n" + "=" * 80)
    print("DISTANCE VALIDATION CHECK")
    print("=" * 80)
    
    try:
        dist_df = pd.read_csv('BESS_WITH_GEOJSON_ZONES_DISTANCE_VALIDATION.csv')
        high_dist = dist_df[dist_df['Distance_Miles'] > 30]
        
        if not high_dist.empty:
            print(f"\nFacilities more than 30 miles from county center:")
            print("-" * 80)
            for _, row in high_dist.iterrows():
                print(f"{row['BESS']:30} {row['County']:15} {row['Distance_Miles']:.1f} miles")
                
                # Check if this is actually wrong or just a large county
                if row['Distance_Miles'] > 50:
                    print(f"  ⚠️  WARNING: Very high distance - possible mismapping!")
    except:
        print("Could not load distance validation file")
    
    print("\n" + "=" * 80)
    print("ROOT CAUSE ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
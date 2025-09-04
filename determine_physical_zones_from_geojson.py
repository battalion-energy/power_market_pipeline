#!/usr/bin/env python3
"""
Determine physical ERCOT zones for BESS locations using actual ERCOT GeoJSON boundaries
"""

import pandas as pd
import json
from shapely.geometry import Point, shape
import numpy as np

def load_ercot_zones():
    """Load ERCOT zone polygons from GeoJSON"""
    print("Loading ERCOT zones from GeoJSON...")
    
    geojson_path = '/home/enrico/projects/battalion-platform/apps/neoweb/public/geojson/RTO_Regions.geojson'
    
    with open(geojson_path, 'r') as f:
        data = json.load(f)
    
    # Extract ERCOT zones
    ercot_zones = []
    for feature in data['features']:
        if feature['properties'].get('RTO_ISO') == 'ERCOT':
            zone_name = feature['properties']['NAME']
            zone_polygon = shape(feature['geometry'])
            ercot_zones.append({
                'name': zone_name,
                'polygon': zone_polygon
            })
    
    print(f"  Found {len(ercot_zones)} ERCOT zones")
    zone_names = sorted(set(z['name'] for z in ercot_zones))
    print(f"  Zones: {', '.join(zone_names)}")
    
    return ercot_zones

def determine_zone_for_point(lat, lon, ercot_zones):
    """Determine which ERCOT zone a point falls within"""
    if pd.isna(lat) or pd.isna(lon):
        return None
    
    point = Point(lon, lat)  # Note: shapely uses (lon, lat) order
    
    for zone in ercot_zones:
        if zone['polygon'].contains(point):
            return zone['name']
    
    # If point doesn't fall within any zone, find nearest
    min_distance = float('inf')
    nearest_zone = None
    
    for zone in ercot_zones:
        distance = zone['polygon'].distance(point)
        if distance < min_distance:
            min_distance = distance
            nearest_zone = zone['name']
    
    if nearest_zone:
        return f"{nearest_zone} (nearest)"
    
    return None

def update_bess_physical_zones():
    """Update BESS CSV with accurate physical zones from GeoJSON"""
    
    print("\n" + "="*70)
    print("DETERMINING PHYSICAL ERCOT ZONES FROM GEOJSON")
    print("="*70)
    
    # Load ERCOT zones
    ercot_zones = load_ercot_zones()
    
    # Load BESS data
    print("\nLoading BESS data...")
    df = pd.read_csv('/home/enrico/projects/power_market_pipeline/BESS_COMPREHENSIVE_WITH_COORDINATES_V2.csv')
    print(f"  Loaded {len(df)} BESS records")
    
    # Add new physical zone column based on GeoJSON
    print("\nDetermining physical zones from GeoJSON boundaries...")
    df['Physical_Zone_GeoJSON'] = df.apply(
        lambda row: determine_zone_for_point(row['Latitude'], row['Longitude'], ercot_zones),
        axis=1
    )
    
    # Compare with existing physical zones and settlement zones
    print("\nComparison of zone assignments:")
    
    # Show distribution
    print("\n📊 Physical Zone Distribution (from GeoJSON):")
    for zone, count in df['Physical_Zone_GeoJSON'].value_counts().items():
        if zone:
            print(f"  {zone}: {count}")
    
    # Find differences between settlement and GeoJSON physical
    if 'Load_Zone' in df.columns:
        print("\n🔍 Comparing Settlement vs GeoJSON Physical Zones:")
        
        # Create mapping from detailed to simple zones
        zone_mapping = {
            'Houston': 'LZ_HOUSTON',
            'North': 'LZ_NORTH',
            'North Central': 'LZ_NORTH',
            'North LR': 'LZ_NORTH',
            'South': 'LZ_SOUTH',
            'South Central': 'LZ_SOUTH',
            'Southern': 'LZ_SOUTH',
            'Austin': 'LZ_SOUTH',  # Austin is typically in South zone
            'West': 'LZ_WEST',
            'Far West': 'LZ_WEST',
            'West LR': 'LZ_WEST',
            'Coast': 'LZ_SOUTH',  # Coast is typically South
            'East LR': 'LZ_SOUTH',
            'CPS energy': 'LZ_SOUTH'  # CPS Energy (San Antonio) is South
        }
        
        df['Physical_Zone_Simple'] = df['Physical_Zone_GeoJSON'].apply(
            lambda x: zone_mapping.get(x.replace(' (nearest)', '') if x else None, 'Unknown') if x else 'Unknown'
        )
        
        # Count differences
        differences = df[
            (df['Load_Zone'].notna()) & 
            (df['Physical_Zone_Simple'].notna()) &
            (df['Load_Zone'] != df['Physical_Zone_Simple'])
        ]
        
        print(f"\n  Found {len(differences)} BESS where settlement ≠ GeoJSON physical zone")
        
        if len(differences) > 0:
            print("\n  Examples of differences:")
            for _, row in differences.head(10).iterrows():
                print(f"    {row['BESS_Gen_Resource']}:")
                print(f"      Settlement: {row['Load_Zone']}")
                print(f"      GeoJSON Physical: {row['Physical_Zone_GeoJSON']}")
                print(f"      Coordinates: ({row['Latitude']:.4f}, {row['Longitude']:.4f})")
    
    # Check Crossett specifically
    crossett = df[df['BESS_Gen_Resource'].str.contains('CROSSETT', case=False, na=False)]
    if not crossett.empty:
        print("\n✅ Crossett BESS zone determination:")
        for _, row in crossett.iterrows():
            print(f"  {row['BESS_Gen_Resource']}:")
            print(f"    Settlement Zone: {row.get('Load_Zone', 'N/A')}")
            print(f"    GeoJSON Physical Zone: {row['Physical_Zone_GeoJSON']}")
            print(f"    Coordinates: ({row['Latitude']:.4f}, {row['Longitude']:.4f})")
    
    # Save updated file
    output_file = '/home/enrico/projects/power_market_pipeline/BESS_WITH_GEOJSON_ZONES.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✅ Saved file with GeoJSON zones to: {output_file}")
    
    return df

if __name__ == '__main__':
    # Install shapely if needed
    try:
        from shapely.geometry import Point, shape
    except ImportError:
        print("Installing shapely...")
        import subprocess
        subprocess.run(['pip', 'install', 'shapely'])
        from shapely.geometry import Point, shape
    
    update_bess_physical_zones()
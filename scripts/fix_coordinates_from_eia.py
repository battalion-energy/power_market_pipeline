#!/usr/bin/env python3
"""
Fix coordinates in comprehensive file using EIA data as authoritative source
"""

import pandas as pd
import numpy as np

def main():
    print("=" * 80)
    print("FIXING COORDINATE MISMATCHES USING EIA DATA")
    print("=" * 80)
    
    # Load files
    comp_df = pd.read_csv('BESS_COMPREHENSIVE_WITH_COORDINATES_V2.csv')
    eia_df = pd.read_csv('BESS_EIA_COMPREHENSIVE_VERIFIED.csv')
    
    print(f"Loaded {len(comp_df)} records from comprehensive file")
    print(f"Loaded {len(eia_df)} records from EIA verified file")
    
    fixes_made = []
    
    # For each BESS in comprehensive file
    for idx, row in comp_df.iterrows():
        bess_name = row['BESS_Gen_Resource']
        current_lat = row.get('Latitude', 0)
        current_lon = row.get('Longitude', 0)
        
        # Find matching EIA record
        eia_match = eia_df[eia_df['BESS_Gen_Resource'] == bess_name]
        
        if not eia_match.empty:
            eia_lat = eia_match.iloc[0].get('EIA_Latitude', 0)
            eia_lon = eia_match.iloc[0].get('EIA_Longitude', 0)
            eia_county = eia_match.iloc[0].get('EIA_County', '')
            
            # If EIA has valid coordinates and they differ from current
            if eia_lat != 0 and eia_lon != 0:
                if abs(current_lat - eia_lat) > 0.01 or abs(current_lon - eia_lon) > 0.01:
                    fixes_made.append({
                        'BESS': bess_name,
                        'Old_Lat': current_lat,
                        'Old_Lon': current_lon,
                        'New_Lat': eia_lat,
                        'New_Lon': eia_lon,
                        'County': eia_county
                    })
                    
                    # Update coordinates
                    comp_df.at[idx, 'Latitude'] = eia_lat
                    comp_df.at[idx, 'Longitude'] = eia_lon
                    comp_df.at[idx, 'Coordinate_Source'] = 'EIA_Verified'
                    
                    # Also update the EIA coordinate columns if they exist
                    if 'EIA_Latitude' in comp_df.columns:
                        comp_df.at[idx, 'EIA_Latitude'] = eia_lat
                    if 'EIA_Longitude' in comp_df.columns:
                        comp_df.at[idx, 'EIA_Longitude'] = eia_lon
                    
                    # Update county to match EIA
                    if 'EIA_County' in comp_df.columns and eia_county:
                        comp_df.at[idx, 'EIA_County'] = eia_county
    
    print(f"\nFixed coordinates for {len(fixes_made)} facilities")
    
    if fixes_made:
        print("\nCoordinate fixes applied:")
        print("-" * 80)
        for fix in fixes_made[:10]:
            print(f"{fix['BESS']}:")
            print(f"  Old: ({fix['Old_Lat']:.6f}, {fix['Old_Lon']:.6f})")
            print(f"  New: ({fix['New_Lat']:.6f}, {fix['New_Lon']:.6f})")
            print(f"  County: {fix['County']}")
        
        if len(fixes_made) > 10:
            print(f"\n... and {len(fixes_made) - 10} more fixes")
        
        # Save fixes report
        pd.DataFrame(fixes_made).to_csv('COORDINATE_FIXES_APPLIED.csv', index=False)
        print(f"\nFixes report saved to: COORDINATE_FIXES_APPLIED.csv")
    
    # Save updated comprehensive file
    comp_df.to_csv('BESS_COMPREHENSIVE_WITH_COORDINATES_V2_FIXED.csv', index=False)
    print(f"\nUpdated file saved to: BESS_COMPREHENSIVE_WITH_COORDINATES_V2_FIXED.csv")
    
    # Also update the main file that the map uses
    comp_df.to_csv('BESS_COMPREHENSIVE_WITH_COORDINATES_V2.csv', index=False)
    print(f"Overwrote original file: BESS_COMPREHENSIVE_WITH_COORDINATES_V2.csv")
    
    # Update the simplified map file too
    print("\nUpdating BESS_WITH_GEOJSON_ZONES.csv...")
    
    try:
        map_df = pd.read_csv('BESS_WITH_GEOJSON_ZONES.csv')
        
        for idx, row in map_df.iterrows():
            bess_name = row['BESS']
            
            # Find the fixed coordinates
            fixed_match = comp_df[comp_df['BESS_Gen_Resource'] == bess_name]
            if not fixed_match.empty:
                new_lat = fixed_match.iloc[0]['Latitude']
                new_lon = fixed_match.iloc[0]['Longitude']
                
                if new_lat != 0 and new_lon != 0:
                    map_df.at[idx, 'BESS_Lat'] = new_lat
                    map_df.at[idx, 'BESS_Lon'] = new_lon
        
        map_df.to_csv('BESS_WITH_GEOJSON_ZONES.csv', index=False)
        print("Updated BESS_WITH_GEOJSON_ZONES.csv with fixed coordinates")
    except Exception as e:
        print(f"Could not update BESS_WITH_GEOJSON_ZONES.csv: {e}")
    
    print("\n" + "=" * 80)
    print("COORDINATE FIX COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
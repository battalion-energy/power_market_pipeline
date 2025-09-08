#!/usr/bin/env python3
"""
Complete BESS Mapping Pipeline V2
Uses the comprehensive mapping file that already has all the data
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def main():
    print("=" * 80)
    print("COMPLETE BESS MAPPING PIPELINE V2")
    print("=" * 80)
    
    # Load the comprehensive file that already has everything
    print("\n1. Loading comprehensive BESS mapping...")
    comp_df = pd.read_csv('BESS_COMPREHENSIVE_WITH_COORDINATES_V2.csv')
    print(f"   Loaded {len(comp_df)} BESS resources")
    
    # Create simplified output with key fields
    print("\n2. Creating simplified mapping output...")
    
    output_df = comp_df[[
        'BESS_Gen_Resource',
        'BESS_Load_Resource', 
        'Settlement_Point',
        'Load_Zone',
        'IQ_County',
        'IQ_Capacity_MW',
        'EIA_Plant_Name',
        'EIA_Capacity_MW',
        'EIA_County',
        'Latitude',
        'Longitude',
        'True_Operational_Status',
        'Data_Completeness_%'
    ]].copy()
    
    # Rename for clarity
    output_df.columns = [
        'BESS_Gen_Resource',
        'BESS_Load_Resource',
        'Settlement_Point',
        'Load_Zone',
        'County',
        'IQ_Capacity_MW',
        'EIA_Plant_Name',
        'EIA_Capacity_MW',
        'EIA_County',
        'Latitude',
        'Longitude',
        'Operational_Status',
        'Data_Completeness'
    ]
    
    # Use EIA county if IQ county is missing
    output_df['County'] = output_df['County'].fillna(output_df['EIA_County'])
    
    # Use EIA capacity if IQ capacity is missing
    output_df['Capacity_MW'] = output_df['IQ_Capacity_MW'].fillna(output_df['EIA_Capacity_MW'])
    
    # Add match quality indicator
    output_df['Has_Coordinates'] = output_df['Latitude'].notna() & output_df['Longitude'].notna()
    output_df['Has_County'] = output_df['County'].notna()
    output_df['Has_Capacity'] = output_df['Capacity_MW'].notna()
    
    output_df['Match_Quality'] = 'No Match'
    output_df.loc[output_df['Has_County'], 'Match_Quality'] = 'County Only'
    output_df.loc[output_df['Has_County'] & output_df['Has_Capacity'], 'Match_Quality'] = 'County+Capacity'
    output_df.loc[output_df['Has_Coordinates'], 'Match_Quality'] = 'Full Match'
    
    # Statistics
    print("\n" + "=" * 80)
    print("MAPPING RESULTS")
    print("=" * 80)
    
    total = len(output_df)
    has_county = output_df['Has_County'].sum()
    has_coords = output_df['Has_Coordinates'].sum()
    has_capacity = output_df['Has_Capacity'].sum()
    
    print(f"\nTotal BESS: {total}")
    print(f"With County: {has_county} ({has_county/total*100:.1f}%)")
    print(f"With Coordinates: {has_coords} ({has_coords/total*100:.1f}%)")
    print(f"With Capacity: {has_capacity} ({has_capacity/total*100:.1f}%)")
    
    # Match quality breakdown
    print("\nMatch Quality:")
    quality_counts = output_df['Match_Quality'].value_counts()
    for quality, count in quality_counts.items():
        print(f"  {quality}: {count} ({count/total*100:.1f}%)")
    
    # Operational status breakdown
    print("\nOperational Status:")
    if 'Operational_Status' in output_df.columns:
        status_counts = output_df['Operational_Status'].value_counts()
        for status, count in status_counts.head(5).items():
            print(f"  {status[:50]}: {count}")
    
    # Save results
    output_file = 'BESS_COMPLETE_MAPPING_PIPELINE.csv'
    output_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    # Special check for CROSSETT
    crossett = output_df[output_df['BESS_Gen_Resource'].str.contains('CROSSETT', case=False, na=False)]
    if not crossett.empty:
        print("\n" + "=" * 80)
        print("CROSSETT VALIDATION")
        print("=" * 80)
        for _, row in crossett.iterrows():
            print(f"\n{row['BESS_Gen_Resource']}:")
            print(f"  County: {row['County']}")
            print(f"  Coordinates: ({row['Latitude']}, {row['Longitude']})")
            print(f"  Match Quality: {row['Match_Quality']}")
            print(f"  Capacity: {row['Capacity_MW']} MW")
            
            if row['County'] and str(row['County']).upper() == 'CRANE':
                print(f"  ✓ Correctly in Crane County")
            elif pd.notna(row['Latitude']) and pd.notna(row['Longitude']):
                # Check if coordinates are in West Texas (Crane County area)
                if row['Longitude'] < -101 and row['Longitude'] > -103 and row['Latitude'] > 31 and row['Latitude'] < 32:
                    print(f"  ✓ Coordinates appear to be in Crane County area")
                else:
                    print(f"  ✗ WARNING: Coordinates don't match Crane County location!")
            else:
                print(f"  ✗ ERROR: Not properly located in Crane County!")
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
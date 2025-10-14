#!/usr/bin/env python3
"""
FIXED VERSION: Cross-reference BESS resources with EIA Generator data
CRITICAL FIX: County matching is now REQUIRED, not optional
"""

import pandas as pd
import numpy as np
from rapidfuzz import fuzz, process
import json
import os
from dotenv import load_dotenv

# Load environment
load_dotenv('/home/enrico/projects/battalion-platform/.env')

def load_eia_data(sheet_name='Operating'):
    """Load EIA generator data from Excel"""
    file_path = '/home/enrico/projects/battalion-platform/data/EIA/generators/EIA_generators_latest.xlsx'
    
    # Read with correct header row (row 2)
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=2)
    
    # Filter for Texas battery storage
    df_tx = df[df['Plant State'] == 'TX'].copy()
    
    # Filter for battery/energy storage
    df_bess = df_tx[
        (df_tx['Technology'].str.contains('Battery', na=False)) |
        (df_tx['Technology'].str.contains('Energy Storage', na=False)) |
        (df_tx['Energy Source Code'].str.contains('MWH', na=False)) |
        (df_tx['Prime Mover Code'].str.contains('BA', na=False))
    ].copy()
    
    print(f"Loaded {len(df_bess)} TX battery storage facilities from {sheet_name} tab")
    
    # Clean up names
    df_bess['Plant Name Clean'] = df_bess['Plant Name'].str.upper().str.replace(' ', '_')
    df_bess['Generator ID Clean'] = df_bess['Generator ID'].astype(str).str.upper()
    
    return df_bess

def load_bess_resources():
    """Load BESS resources from improved matched data"""
    df = pd.read_csv('/home/enrico/projects/power_market_pipeline/BESS_IMPROVED_MATCHED.csv')
    
    # Also load the resource mapping to get more info
    mapping_df = pd.read_csv('/home/enrico/projects/power_market_pipeline/BESS_RESOURCE_MAPPING_REAL_ONLY.csv')
    
    # Merge to get substation and load zone info
    df = df.merge(
        mapping_df[['BESS_Gen_Resource', 'Substation', 'Load_Zone']],
        on='BESS_Gen_Resource',
        how='left',
        suffixes=('', '_mapping')
    )
    
    # Use mapping data to fill missing values
    for col in ['Substation', 'Load_Zone']:
        if f'{col}_mapping' in df.columns:
            df[col] = df[col].fillna(df[f'{col}_mapping'])
            df = df.drop(columns=[f'{col}_mapping'])
    
    return df

def normalize_county_name(county):
    """Normalize county names for matching"""
    if pd.isna(county):
        return ''
    
    # Remove 'County' suffix and standardize
    county = str(county).upper().replace(' COUNTY', '').replace(' CO.', '').strip()
    return county

def heuristic_match_fixed(bess_df, eia_df, pass_name="Heuristic"):
    """
    FIXED: Heuristic matching that REQUIRES county match
    County mismatch will PREVENT matching regardless of name similarity
    """
    matches = []
    
    for _, bess_row in bess_df.iterrows():
        bess_name = bess_row['BESS_Gen_Resource']
        bess_county = normalize_county_name(bess_row.get('County'))
        
        # Skip if BESS has no county data
        if not bess_county:
            print(f"  ⚠️ Skipping {bess_name} - no county data")
            continue
        
        best_match = None
        best_score = 0
        best_reason = ""
        
        # ONLY consider EIA entries from the SAME COUNTY
        for _, eia_row in eia_df.iterrows():
            eia_county = normalize_county_name(eia_row.get('County'))
            
            # CRITICAL FIX: Skip if counties don't match
            if not eia_county or eia_county != bess_county:
                continue
            
            eia_plant = eia_row['Plant Name Clean']
            eia_gen_id = eia_row['Generator ID Clean']
            
            # Calculate name similarity
            plant_similarity = fuzz.ratio(bess_name, eia_plant) / 100
            gen_similarity = fuzz.ratio(bess_name, eia_gen_id) / 100
            name_similarity = max(plant_similarity, gen_similarity)
            
            # Only consider if name similarity is reasonable
            if name_similarity < 0.3:  # Too different
                continue
            
            # Score based on name similarity (county already verified)
            score = name_similarity * 100
            reason = f"County: {bess_county}, Name similarity: {name_similarity:.2f}"
            
            if score > best_score:
                best_score = score
                best_match = eia_row
                best_reason = reason
        
        if best_score >= 30:  # Minimum threshold (30% name similarity with county match)
            matches.append({
                'BESS_Gen_Resource': bess_name,
                'EIA_Plant_Name': best_match['Plant Name'],
                'EIA_Generator_ID': best_match['Generator ID'],
                'EIA_County': best_match.get('County'),
                'EIA_Technology': best_match.get('Technology'),
                'EIA_Capacity_MW': best_match.get('Nameplate Capacity (MW)'),
                'EIA_Operating_Year': best_match.get('Operating Year'),
                'EIA_Latitude': best_match.get('Latitude'),
                'EIA_Longitude': best_match.get('Longitude'),
                'match_score': best_score,
                'match_reason': best_reason,
                'Pass': pass_name,
                'Source': 'EIA'
            })
    
    return pd.DataFrame(matches)

def main():
    """Run the fixed matching process"""
    print("="*70)
    print("FIXED EIA MATCHING - COUNTY VERIFICATION REQUIRED")
    print("="*70)
    
    # Load BESS resources
    print("\n1. Loading BESS resources...")
    bess_df = load_bess_resources()
    print(f"   Loaded {len(bess_df)} BESS resources")
    
    # Check how many have county data
    has_county = bess_df['County'].notna()
    print(f"   {has_county.sum()} have county data ({100*has_county.mean():.1f}%)")
    
    # Load EIA data
    print("\n2. Loading EIA data...")
    eia_operating = load_eia_data('Operating')
    eia_planned = load_eia_data('Planned')
    
    # Run matching with FIXED algorithm
    print("\n3. Running FIXED matching (county required)...")
    
    # Pass 1: Operating
    print("\n   Pass 1: Operating facilities")
    matches_op = heuristic_match_fixed(bess_df, eia_operating, "Pass 1 (Operating)")
    print(f"   Matched {len(matches_op)} BESS to operating facilities")
    
    # Pass 2: Planned (for remaining unmatched)
    unmatched_bess = bess_df[~bess_df['BESS_Gen_Resource'].isin(matches_op['BESS_Gen_Resource'])]
    print(f"\n   Pass 2: Planned facilities ({len(unmatched_bess)} remaining)")
    matches_plan = heuristic_match_fixed(unmatched_bess, eia_planned, "Pass 2 (Planned)")
    print(f"   Matched {len(matches_plan)} BESS to planned facilities")
    
    # Combine results
    all_matches = pd.concat([matches_op, matches_plan], ignore_index=True)
    
    # Check specific cases
    print("\n4. Checking specific cases...")
    
    # Check CROSSETT
    crossett_matches = all_matches[all_matches['BESS_Gen_Resource'].str.contains('CROSSETT', case=False, na=False)]
    if not crossett_matches.empty:
        print("\n   CROSSETT matches:")
        for _, row in crossett_matches.iterrows():
            print(f"     {row['BESS_Gen_Resource']} -> {row['EIA_Plant_Name']} ({row['EIA_County']} County)")
            print(f"       Coordinates: ({row.get('EIA_Latitude', 'N/A')}, {row.get('EIA_Longitude', 'N/A')})")
    else:
        print("\n   CROSSETT: No matches found")
        
        # Check if Crossett Power Management exists in EIA
        crossett_eia = eia_operating[eia_operating['Plant Name'].str.contains('Crossett', case=False, na=False)]
        if not crossett_eia.empty:
            print("   But found in EIA data:")
            for _, row in crossett_eia.iterrows():
                print(f"     {row['Plant Name']} - {row['County']} County")
                print(f"       Coordinates: ({row.get('Latitude', 'N/A')}, {row.get('Longitude', 'N/A')})")
    
    # Save results
    output_file = '/home/enrico/projects/power_market_pipeline/BESS_EIA_MATCHED_FIXED.csv'
    all_matches.to_csv(output_file, index=False)
    print(f"\n✅ Saved FIXED matches to: {output_file}")
    
    # Summary statistics
    print("\n5. Summary Statistics:")
    print(f"   Total BESS: {len(bess_df)}")
    print(f"   Matched: {len(all_matches)} ({100*len(all_matches)/len(bess_df):.1f}%)")
    print(f"   Unmatched: {len(bess_df) - len(all_matches)}")
    
    # Check for suspicious cross-county matches (should be none now)
    print("\n6. Verifying no cross-county matches...")
    for _, match in all_matches.iterrows():
        bess_info = bess_df[bess_df['BESS_Gen_Resource'] == match['BESS_Gen_Resource']].iloc[0]
        bess_county = normalize_county_name(bess_info.get('County'))
        eia_county = normalize_county_name(match['EIA_County'])
        
        if bess_county and eia_county and bess_county != eia_county:
            print(f"   ⚠️ MISMATCH: {match['BESS_Gen_Resource']}")
            print(f"      BESS County: {bess_county}")
            print(f"      EIA County: {eia_county}")
    
    print("\n✅ FIXED matching complete!")
    return all_matches

if __name__ == '__main__':
    main()
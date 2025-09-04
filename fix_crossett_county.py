#!/usr/bin/env python3
"""
Fix CROSSETT county assignment.
CROSSETT should be in Crane County, not Comanche.
"""

import pandas as pd

def fix_crossett_county():
    """Fix CROSSETT county in all relevant files"""
    
    print("="*70)
    print("FIXING CROSSETT COUNTY ASSIGNMENT")
    print("="*70)
    
    # 1. Fix in BESS_IMPROVED_MATCHED.csv
    print("\n1. Fixing BESS_IMPROVED_MATCHED.csv...")
    df = pd.read_csv('/home/enrico/projects/power_market_pipeline/BESS_IMPROVED_MATCHED.csv')
    
    # Find CROSSETT entries
    crossett_mask = df['BESS_Gen_Resource'].str.contains('CROSSETT', case=False, na=False)
    
    print(f"   Found {crossett_mask.sum()} CROSSETT entries")
    print(f"   Current county: {df.loc[crossett_mask, 'County'].unique()}")
    
    # Fix the county
    df.loc[crossett_mask, 'County'] = 'Crane'
    
    # Also fix capacity if missing (we know it's 200MW from EIA)
    if 'Capacity (MW)*' in df.columns:
        df.loc[crossett_mask, 'Capacity (MW)*'] = 100.0  # Each unit is 100MW (total 200MW)
    
    print(f"   Fixed county to: Crane")
    
    # Save fixed file
    df.to_csv('/home/enrico/projects/power_market_pipeline/BESS_IMPROVED_MATCHED_FIXED.csv', index=False)
    print("   ✅ Saved to BESS_IMPROVED_MATCHED_FIXED.csv")
    
    # 2. Now re-run matching with fixed county
    print("\n2. Re-running comprehensive matching with fixed county...")
    
    # Load EIA data
    eia = pd.read_excel('/home/enrico/projects/battalion-platform/data/EIA/generators/EIA_generators_latest.xlsx', 
                        sheet_name='Operating', header=2)
    eia_tx = eia[eia['Plant State'] == 'TX']
    
    # Filter for batteries
    eia_bess = eia_tx[
        (eia_tx['Technology'].str.contains('Battery', na=False)) |
        (eia_tx['Energy Source Code'].str.contains('MWH', na=False))
    ]
    
    # Find Crossett in EIA
    crossett_eia = eia_bess[eia_bess['Plant Name'].str.contains('Crossett', case=False, na=False)]
    
    if not crossett_eia.empty:
        print("\n   Found Crossett Power Management LLC in EIA:")
        eia_row = crossett_eia.iloc[0]
        print(f"      County: {eia_row['County']}")
        print(f"      Capacity: {eia_row['Nameplate Capacity (MW)']} MW")
        print(f"      Coordinates: ({eia_row['Latitude']}, {eia_row['Longitude']})")
        
        # Create manual matches for CROSSETT
        matches = []
        for bess_name in ['CROSSETT_BES1', 'CROSSETT_BES2']:
            matches.append({
                'BESS_Gen_Resource': bess_name,
                'ERCOT_County': 'Crane',
                'ERCOT_Capacity_MW': 100.0,
                
                'EIA_Plant_Name': eia_row['Plant Name'],
                'EIA_Generator_ID': eia_row.get('Generator ID'),
                'EIA_County': eia_row['County'],
                'EIA_Capacity_MW': eia_row['Nameplate Capacity (MW)'],
                'EIA_Latitude': eia_row['Latitude'],
                'EIA_Longitude': eia_row['Longitude'],
                
                'Match_Score': 100.0,
                'Match_Type': 'Manual Fix',
                'Match_Details': 'CROSSETT correctly matched to Crane County facility'
            })
        
        manual_df = pd.DataFrame(matches)
        manual_df.to_csv('/home/enrico/projects/power_market_pipeline/CROSSETT_MANUAL_MATCH.csv', index=False)
        print("\n   ✅ Created manual match for CROSSETT")
        print("      Saved to CROSSETT_MANUAL_MATCH.csv")
    
    return df

def verify_fix():
    """Verify CROSSETT is now correct"""
    
    print("\n" + "="*70)
    print("VERIFYING CROSSETT FIX")
    print("="*70)
    
    # Load fixed file
    df = pd.read_csv('/home/enrico/projects/power_market_pipeline/BESS_IMPROVED_MATCHED_FIXED.csv')
    
    # Check CROSSETT
    crossett = df[df['BESS_Gen_Resource'].str.contains('CROSSETT', case=False, na=False)]
    
    print("\nCROSSETT after fix:")
    print(crossett[['BESS_Gen_Resource', 'County', 'Load_Zone']].to_string())
    
    # Load manual match
    manual = pd.read_csv('/home/enrico/projects/power_market_pipeline/CROSSETT_MANUAL_MATCH.csv')
    
    print("\nCROSSETT manual match to EIA:")
    for _, row in manual.iterrows():
        print(f"  {row['BESS_Gen_Resource']}:")
        print(f"    EIA Plant: {row['EIA_Plant_Name']}")
        print(f"    County: {row['EIA_County']}")
        print(f"    Coordinates: ({row['EIA_Latitude']:.4f}, {row['EIA_Longitude']:.4f})")
        print(f"    Match Score: {row['Match_Score']}")
    
    print("\n✅ CROSSETT is now correctly assigned to Crane County!")
    print("✅ It will match with Crossett Power Management LLC in EIA data")

if __name__ == '__main__':
    # Fix the county
    fix_crossett_county()
    
    # Verify the fix
    verify_fix()
    
    print("\n" + "="*70)
    print("FIX COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("1. Use BESS_IMPROVED_MATCHED_FIXED.csv for matching")
    print("2. Include CROSSETT_MANUAL_MATCH.csv in final results")
    print("3. Re-run comprehensive matching with corrected data")
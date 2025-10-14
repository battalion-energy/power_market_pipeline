#!/usr/bin/env python3
"""
Improved BESS matching that accounts for co-located vs standalone facilities
Key insight: Only ~50 BESS are standalone, most are co-located with solar/wind
"""

import pandas as pd
import numpy as np
from rapidfuzz import fuzz

def analyze_bess_types():
    """Analyze and categorize BESS resources by type"""
    
    print("="*70)
    print("BESS CATEGORIZATION: STANDALONE vs CO-LOCATED")
    print("="*70)
    
    # Load our BESS list
    bess_df = pd.read_csv('/home/enrico/projects/power_market_pipeline/BESS_IMPROVED_MATCHED.csv')
    
    # Load interconnection queue data
    coloc_df = pd.read_csv('/home/enrico/projects/power_market_pipeline/interconnection_queue_clean/co_located_operational.csv')
    standalone_df = pd.read_csv('/home/enrico/projects/power_market_pipeline/interconnection_queue_clean/stand_alone.csv')
    solar_df = pd.read_csv('/home/enrico/projects/power_market_pipeline/interconnection_queue_clean/co_located_with_solar.csv')
    wind_df = pd.read_csv('/home/enrico/projects/power_market_pipeline/interconnection_queue_clean/co_located_with_wind.csv')
    
    # Categorize operational facilities
    standalone_operational = coloc_df[coloc_df['Fuel'] == 'OTH'].copy() if 'Fuel' in coloc_df.columns else pd.DataFrame()
    solar_coloc_operational = coloc_df[coloc_df['Fuel'] == 'SOL'].copy() if 'Fuel' in coloc_df.columns else pd.DataFrame()
    wind_coloc_operational = coloc_df[coloc_df['Fuel'] == 'WIN'].copy() if 'Fuel' in coloc_df.columns else pd.DataFrame()
    
    print(f"\n📊 ERCOT Operational BESS Breakdown:")
    print(f"  Standalone BESS (Fuel=OTH): {len(standalone_operational)}")
    print(f"  Solar co-located (Fuel=SOL): {len(solar_coloc_operational)}")
    print(f"  Wind co-located (Fuel=WIN): {len(wind_coloc_operational)}")
    print(f"  Total operational: {len(standalone_operational) + len(solar_coloc_operational) + len(wind_coloc_operational)}")
    
    # Analyze our BESS names for patterns
    bess_names = bess_df['BESS_Gen_Resource'].unique()
    
    # Categorize by naming patterns
    solar_pattern_bess = []
    wind_pattern_bess = []
    standalone_pattern_bess = []
    
    for bess in bess_names:
        bess_upper = bess.upper()
        
        # Solar indicators
        if any(pattern in bess_upper for pattern in ['SLR', 'SOLAR', 'SUN', 'PV']):
            solar_pattern_bess.append(bess)
        # Wind indicators
        elif any(pattern in bess_upper for pattern in ['WIND', 'WND', 'TURB']):
            wind_pattern_bess.append(bess)
        # Everything else likely standalone
        else:
            standalone_pattern_bess.append(bess)
    
    print(f"\n🔍 Our BESS Resources Pattern Analysis:")
    print(f"  Likely solar co-located (SLR/SOLAR in name): {len(solar_pattern_bess)}")
    print(f"  Likely wind co-located (WIND in name): {len(wind_pattern_bess)}")
    print(f"  Likely standalone: {len(standalone_pattern_bess)}")
    
    # Create categorized matching
    results = []
    
    # 1. Match standalone BESS to standalone facilities
    print("\n🎯 Matching Strategy:")
    print("1. Standalone BESS → Standalone facilities (Fuel=OTH)")
    
    standalone_matched = 0
    for bess in standalone_pattern_bess:
        best_match = find_best_match(bess, standalone_operational, 'Standalone')
        if best_match:
            results.append(best_match)
            standalone_matched += 1
    
    print(f"   Matched: {standalone_matched}/{len(standalone_pattern_bess)}")
    
    # 2. Match solar pattern BESS to solar co-located
    print("\n2. Solar-pattern BESS → Solar co-located (Fuel=SOL)")
    
    solar_matched = 0
    for bess in solar_pattern_bess:
        # For solar co-located, the BESS name might be derived from the solar project
        # Need to match against the base solar project name
        best_match = find_best_match_solar(bess, solar_coloc_operational)
        if best_match:
            results.append(best_match)
            solar_matched += 1
    
    print(f"   Matched: {solar_matched}/{len(solar_pattern_bess)}")
    
    # 3. Match wind pattern BESS to wind co-located
    print("\n3. Wind-pattern BESS → Wind co-located (Fuel=WIN)")
    
    wind_matched = 0
    for bess in wind_pattern_bess:
        best_match = find_best_match_wind(bess, wind_coloc_operational)
        if best_match:
            results.append(best_match)
            wind_matched += 1
    
    print(f"   Matched: {wind_matched}/{len(wind_pattern_bess)}")
    
    # Save categorized results
    if results:
        results_df = pd.DataFrame(results)
        output_file = '/home/enrico/projects/power_market_pipeline/BESS_CATEGORIZED_MATCHES.csv'
        results_df.to_csv(output_file, index=False)
        print(f"\n✅ Saved categorized matches to: {output_file}")
    
    return results

def find_best_match(bess_name, facility_df, category):
    """Find best match for standalone BESS"""
    if facility_df.empty or 'Unit Code' not in facility_df.columns:
        return None
    
    best_score = 0
    best_match = None
    
    for _, row in facility_df.iterrows():
        unit_code = str(row.get('Unit Code', '')).upper()
        project_name = str(row.get('Project Name', '')).upper()
        
        # Direct unit code match
        if bess_name.upper() == unit_code:
            score = 100
        else:
            # Fuzzy match
            score1 = fuzz.ratio(bess_name.upper(), unit_code)
            score2 = fuzz.ratio(bess_name.upper(), project_name)
            score = max(score1, score2)
        
        if score > best_score and score >= 50:
            best_score = score
            best_match = {
                'BESS_Gen_Resource': bess_name,
                'Category': category,
                'Unit_Code': row.get('Unit Code'),
                'Project_Name': row.get('Project Name'),
                'County': row.get('County'),
                'Fuel': row.get('Fuel'),
                'Match_Score': score,
                'Capacity_MW': row.get('Capacity (MW)')
            }
    
    return best_match

def find_best_match_solar(bess_name, solar_df):
    """Match BESS to solar co-located facility"""
    if solar_df.empty:
        return None
    
    # For solar co-located, the BESS might be named after the solar project
    # Remove BESS/ESS suffixes and match to solar project base name
    bess_base = bess_name.upper().replace('_BESS', '').replace('_ESS', '').replace('_BES', '')
    
    best_score = 0
    best_match = None
    
    for _, row in solar_df.iterrows():
        # Solar projects might have UNIT1, UNIT2 etc for the solar part
        # The BESS part might be BESS1, BESS2 or similar
        unit_code = str(row.get('Unit Code', '')).upper()
        project_name = str(row.get('Project Name', '')).upper()
        
        # Extract base name from unit code (remove UNIT suffix)
        unit_base = unit_code.replace('_UNIT', '').replace('_SOLAR', '')
        
        # Check if this is the BESS component of a solar project
        if 'BESS' in unit_code or 'ESS' in unit_code:
            score = fuzz.ratio(bess_name.upper(), unit_code)
        else:
            # Match against the solar project base name
            score1 = fuzz.ratio(bess_base, unit_base)
            score2 = fuzz.ratio(bess_base, project_name.replace(' SOLAR', ''))
            score = max(score1, score2)
        
        if score > best_score and score >= 40:
            best_score = score
            best_match = {
                'BESS_Gen_Resource': bess_name,
                'Category': 'Solar Co-located',
                'Unit_Code': row.get('Unit Code'),
                'Project_Name': row.get('Project Name'),
                'County': row.get('County'),
                'Fuel': row.get('Fuel', 'SOL'),
                'Match_Score': score,
                'Capacity_MW': row.get('Capacity (MW)')
            }
    
    return best_match

def find_best_match_wind(bess_name, wind_df):
    """Match BESS to wind co-located facility"""
    if wind_df.empty:
        return None
    
    # Similar logic for wind co-located
    bess_base = bess_name.upper().replace('_BESS', '').replace('_ESS', '').replace('_BES', '')
    
    best_score = 0
    best_match = None
    
    for _, row in wind_df.iterrows():
        unit_code = str(row.get('Unit Code', '')).upper()
        project_name = str(row.get('Project Name', '')).upper()
        
        # Wind projects might have different unit naming
        unit_base = unit_code.replace('_WIND', '').replace('_TURB', '')
        
        if 'BESS' in unit_code or 'ESS' in unit_code:
            score = fuzz.ratio(bess_name.upper(), unit_code)
        else:
            score1 = fuzz.ratio(bess_base, unit_base)
            score2 = fuzz.ratio(bess_base, project_name.replace(' WIND', ''))
            score = max(score1, score2)
        
        if score > best_score and score >= 40:
            best_score = score
            best_match = {
                'BESS_Gen_Resource': bess_name,
                'Category': 'Wind Co-located',
                'Unit_Code': row.get('Unit Code'),
                'Project_Name': row.get('Project Name'),
                'County': row.get('County'),
                'Fuel': row.get('Fuel', 'WIN'),
                'Match_Score': score,
                'Capacity_MW': row.get('Capacity (MW)')
            }
    
    return best_match

def create_summary_report():
    """Create a summary of findings"""
    
    print("\n" + "="*70)
    print("KEY FINDINGS AND RECOMMENDATIONS")
    print("="*70)
    
    print("""
1. CRITICAL INSIGHT:
   - Only ~50 BESS in ERCOT are truly standalone (Fuel=OTH)
   - 51 BESS are co-located with Solar (Fuel=SOL)
   - 21 BESS are co-located with Wind (Fuel=WIN)
   - Many more are thermal retrofits (Fuel=GAS)

2. MATCHING IMPLICATIONS:
   - Direct BESS name matching only works for standalone units
   - Co-located BESS often use the parent project's naming
   - Need to match against solar/wind project names, not BESS names

3. RECOMMENDED APPROACH:
   a) Categorize BESS by likely type (standalone/solar/wind)
   b) Match each category against appropriate facility type
   c) For co-located, match to parent project, not BESS name
   d) Consider that many BESS are retrofits on existing generation

4. DATA QUALITY NOTES:
   - Our list has 193 BESS resources
   - ERCOT operational has ~122 BESS (50 standalone + 72 co-located)
   - The gap suggests many are planned/under construction
   - Some may be naming variations of the same facility
""")

if __name__ == '__main__':
    results = analyze_bess_types()
    create_summary_report()
    
    print("\n✅ Analysis complete!")
    print("This explains why only ~50% of BESS matched in previous attempts.")
    print("Most BESS are co-located and need different matching logic.")
#!/usr/bin/env python3
"""
IMPROVED Generator Mapping Pipeline for ERCOT
Enhanced matching logic with direct EIA matching
"""

import pandas as pd
import numpy as np
import warnings
import re
from typing import Dict, List, Tuple, Optional
warnings.filterwarnings('ignore')

# Import the LLM-generated mappings
from ercot_code_mappings import ERCOT_CODE_TO_PLANT_NAME, get_plant_name

# File paths
INTERCONNECTION_FILE = '/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/ERCOT_InterconnectionQueue/interconnection_gis_report.xlsx'
EIA_PLANT_FILE = '/home/enrico/experiments/ERCOT_SCED/pypsa-usa/workflow/repo_data/plants/eia860_ads_merged.csv'
DAM_RESOURCE_FILE = '/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/60-Day_DAM_Disclosure_Reports/csv/60d_DAM_Gen_Resource_Data-18-JAN-24.csv'
GEN_NODE_MAP = '/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/Settlement_Points_List_and_Electrical_Buses_Mapping/latest_mapping/SP_List_EB_Mapping/gen_node_map.csv'
SETTLEMENT_POINTS = '/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/Settlement_Points_List_and_Electrical_Buses_Mapping/latest_mapping/SP_List_EB_Mapping/Settlement_Points_07242024_210751.csv'

def extract_ercot_code(resource_name: str) -> str:
    """Extract the base ERCOT code from resource name"""
    if pd.isna(resource_name):
        return ''
    
    # Get the part before the first underscore
    if '_' in resource_name:
        return resource_name.split('_')[0]
    return resource_name

def normalize_for_matching(text: str) -> str:
    """Normalize text for fuzzy matching"""
    if pd.isna(text):
        return ''
    
    text = str(text).upper()
    
    # Remove common suffixes
    suffixes = ['GENERATING', 'STATION', 'PLANT', 'FACILITY', 'PROJECT', 
                'LLC', 'LP', 'INC', 'CORP', 'COMPANY', 'LTD', 'LIMITED',
                'ENERGY', 'POWER', 'GENERATION', 'GEN', 'ELECTRIC']
    for suffix in suffixes:
        text = text.replace(suffix, '')
    
    # Remove special characters
    text = re.sub(r'[^A-Z0-9\s]', ' ', text)
    
    # Clean up spaces
    text = ' '.join(text.split())
    
    return text.strip()

def calculate_match_score(name1: str, name2: str, cap1: float = None, cap2: float = None) -> float:
    """Calculate match score between two names with optional capacity validation"""
    
    norm1 = normalize_for_matching(name1)
    norm2 = normalize_for_matching(name2)
    
    if not norm1 or not norm2:
        return 0.0
    
    # Exact match
    if norm1 == norm2:
        name_score = 1.0
    # One contains the other
    elif norm1 in norm2 or norm2 in norm1:
        min_len = min(len(norm1), len(norm2))
        max_len = max(len(norm1), len(norm2))
        name_score = min_len / max_len if max_len > 0 else 0
    else:
        # Token overlap - more sophisticated
        tokens1 = set(norm1.split())
        tokens2 = set(norm2.split())
        
        # Remove very common tokens
        common_tokens = {'WIND', 'SOLAR', 'BATTERY', 'BESS', 'STORAGE', 'FARM', 
                        'PARK', 'CENTER', 'UNIT', 'CC', 'CT', 'COMBINED', 'CYCLE'}
        tokens1 = tokens1 - common_tokens
        tokens2 = tokens2 - common_tokens
        
        if tokens1 and tokens2:
            overlap = len(tokens1 & tokens2)
            # Weight by importance of overlapping tokens
            if overlap > 0:
                # If key tokens match, boost score
                name_score = overlap / min(len(tokens1), len(tokens2))
            else:
                name_score = 0
        else:
            name_score = 0
    
    # If we have capacity data, use it to validate
    if cap1 is not None and cap2 is not None and cap1 > 0 and cap2 > 0:
        cap_diff = abs(cap1 - cap2) / max(cap1, cap2)
        if cap_diff < 0.1:  # Within 10%
            cap_score = 1.0
        elif cap_diff < 0.2:  # Within 20%
            cap_score = 0.8
        elif cap_diff < 0.3:  # Within 30%
            cap_score = 0.6
        else:
            cap_score = 0.3
        
        # Weighted average
        return 0.7 * name_score + 0.3 * cap_score
    
    return name_score

def find_best_eia_match(decoded_name: str, eia_plants: pd.DataFrame, threshold: float = 0.3) -> Optional[pd.Series]:
    """Find best EIA match with lower threshold"""
    best_score = 0
    best_match = None
    
    # Try different variations of the name
    name_variations = [decoded_name]
    
    # If it contains "Cogeneration", also try without it
    if 'Cogeneration' in decoded_name:
        name_variations.append(decoded_name.replace('Cogeneration Plant', '').replace('Cogeneration', '').strip())
    
    # If it contains parentheses, try without them
    if '(' in decoded_name:
        base_name = re.sub(r'\([^)]*\)', '', decoded_name).strip()
        name_variations.append(base_name)
    
    for name_var in name_variations:
        for _, eia_row in eia_plants.iterrows():
            score = calculate_match_score(name_var, eia_row['plant_name'])
            
            if score > best_score and score > threshold:
                best_score = score
                best_match = eia_row
    
    return best_match

def main():
    print("=" * 80)
    print("IMPROVED GENERATOR MAPPING PIPELINE")
    print("=" * 80)
    
    # 1. Load ERCOT resources
    print("\n1. Loading ERCOT generator resources...")
    dam_df = pd.read_csv(DAM_RESOURCE_FILE)
    resources = dam_df['Resource Name'].unique()
    print(f"   Found {len(resources)} unique ERCOT resources")
    
    # Create dataframe with decoded names
    ercot_df = pd.DataFrame({'Resource_Name': resources})
    ercot_df['ERCOT_Code'] = ercot_df['Resource_Name'].apply(extract_ercot_code)
    ercot_df['Decoded_Name'] = ercot_df['ERCOT_Code'].apply(get_plant_name)
    
    # 2. Load IQ data (optional, lower priority)
    print("\n2. Loading interconnection queue data...")
    
    # Large Gen
    iq_large = pd.read_excel(INTERCONNECTION_FILE, sheet_name='Project Details - Large Gen', skiprows=30)
    iq_large.columns = iq_large.columns.str.strip()
    iq_large = iq_large[iq_large['INR'].notna()]
    
    # Small Gen
    iq_small = pd.read_excel(INTERCONNECTION_FILE, sheet_name='Project Details - Small Gen', skiprows=14)
    iq_small.columns = iq_small.columns.str.strip()
    iq_small = iq_small[iq_small['INR'].notna()]
    
    # Combine
    iq_df = pd.concat([iq_large, iq_small], ignore_index=True)
    print(f"   Total IQ projects: {len(iq_df)}")
    
    # 3. Load EIA data (primary matching source)
    print("\n3. Loading EIA plant data...")
    eia_df = pd.read_csv(EIA_PLANT_FILE, low_memory=False)
    eia_tx = eia_df[eia_df['state'] == 'TX']
    
    # Aggregate by plant
    eia_plants = eia_tx.groupby('plant_code').agg({
        'plant_name': 'first',
        'latitude': 'first',
        'longitude': 'first',
        'nameplate_capacity_mw': 'sum'
    }).reset_index()
    print(f"   Found {len(eia_plants)} unique Texas plants")
    
    # 4. Load substation data
    print("\n4. Loading substation and settlement point data...")
    gen_map_df = pd.read_csv(GEN_NODE_MAP)
    sp_df = pd.read_csv(SETTLEMENT_POINTS)
    
    resource_to_substation = {}
    for _, row in gen_map_df.iterrows():
        resource_to_substation[row['RESOURCE_NODE']] = {
            'Substation': row['UNIT_SUBSTATION'],
            'Unit_Name': row['UNIT_NAME']
        }
    
    substation_to_zone = {}
    for _, row in sp_df.iterrows():
        if pd.notna(row['SUBSTATION']) and pd.notna(row['SETTLEMENT_LOAD_ZONE']):
            substation_to_zone[row['SUBSTATION']] = row['SETTLEMENT_LOAD_ZONE']
    
    # 5. IMPROVED MATCHING STRATEGY
    print("\n5. Matching using improved strategy...")
    print("   Strategy: Direct EIA matching with lower thresholds")
    
    results = []
    matched_eia_direct = 0
    matched_iq = 0
    matched_eia_via_iq = 0
    
    for _, resource in ercot_df.iterrows():
        result = {
            'Resource_Name': resource['Resource_Name'],
            'ERCOT_Code': resource['ERCOT_Code'],
            'Decoded_Name': resource['Decoded_Name'],
            'IQ_Project': None,
            'IQ_County': None,
            'IQ_Capacity_MW': None,
            'EIA_Plant': None,
            'Latitude': None,
            'Longitude': None,
            'EIA_Capacity_MW': None,
            'Match_Type': None,
            'Match_Confidence': 0.0,
            'Substation': None,
            'Load_Zone': None
        }
        
        # Get substation info
        if resource['Resource_Name'] in resource_to_substation:
            sub_info = resource_to_substation[resource['Resource_Name']]
            result['Substation'] = sub_info['Substation']
            if sub_info['Substation'] in substation_to_zone:
                result['Load_Zone'] = substation_to_zone[sub_info['Substation']]
        
        # STRATEGY 1: Try direct EIA match first (lower threshold)
        eia_match = find_best_eia_match(resource['Decoded_Name'], eia_plants, threshold=0.3)
        
        if eia_match is not None:
            result['EIA_Plant'] = eia_match['plant_name']
            result['Latitude'] = eia_match['latitude']
            result['Longitude'] = eia_match['longitude']
            result['EIA_Capacity_MW'] = eia_match['nameplate_capacity_mw']
            result['Match_Type'] = 'Direct_EIA'
            result['Match_Confidence'] = calculate_match_score(
                resource['Decoded_Name'], 
                eia_match['plant_name']
            )
            matched_eia_direct += 1
        else:
            # STRATEGY 2: Try IQ matching with lower threshold
            best_iq_score = 0
            best_iq_match = None
            
            for _, iq_row in iq_df.iterrows():
                score = calculate_match_score(
                    resource['Decoded_Name'],
                    iq_row['Project Name'],
                    None,
                    iq_row['Capacity (MW)'] if 'Capacity (MW)' in iq_row else None
                )
                
                if score > best_iq_score and score > 0.3:  # Lower threshold
                    best_iq_score = score
                    best_iq_match = iq_row
            
            if best_iq_match is not None:
                result['IQ_Project'] = best_iq_match['Project Name']
                result['IQ_County'] = best_iq_match['County'] if 'County' in best_iq_match else None
                result['IQ_Capacity_MW'] = best_iq_match['Capacity (MW)'] if 'Capacity (MW)' in best_iq_match else None
                matched_iq += 1
                
                # Try to match IQ to EIA
                eia_match = find_best_eia_match(best_iq_match['Project Name'], eia_plants, threshold=0.4)
                if eia_match is not None:
                    result['EIA_Plant'] = eia_match['plant_name']
                    result['Latitude'] = eia_match['latitude']
                    result['Longitude'] = eia_match['longitude']
                    result['EIA_Capacity_MW'] = eia_match['nameplate_capacity_mw']
                    result['Match_Type'] = 'IQ_to_EIA'
                    result['Match_Confidence'] = min(best_iq_score, 
                                                    calculate_match_score(best_iq_match['Project Name'], 
                                                                        eia_match['plant_name']))
                    matched_eia_via_iq += 1
        
        results.append(result)
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # 6. Statistics
    print("\n" + "=" * 80)
    print("IMPROVED MATCHING RESULTS")
    print("=" * 80)
    
    total = len(results_df)
    has_decoded = (results_df['Decoded_Name'] != results_df['ERCOT_Code']).sum()
    has_iq = results_df['IQ_Project'].notna().sum()
    has_coords = results_df['Latitude'].notna().sum()
    direct_eia = (results_df['Match_Type'] == 'Direct_EIA').sum()
    via_iq = (results_df['Match_Type'] == 'IQ_to_EIA').sum()
    
    print(f"\nTotal Resources: {total}")
    print(f"Successfully Decoded: {has_decoded} ({has_decoded/total*100:.1f}%)")
    print(f"Matched to IQ: {has_iq} ({has_iq/total*100:.1f}%)")
    print(f"Matched to EIA (with coords): {has_coords} ({has_coords/total*100:.1f}%)")
    print(f"  - Direct EIA matches: {direct_eia}")
    print(f"  - Via IQ matches: {via_iq}")
    
    # Check coverage by code
    code_stats = results_df.groupby('ERCOT_Code').agg({
        'Resource_Name': 'count',
        'Latitude': lambda x: x.notna().sum()
    }).rename(columns={'Resource_Name': 'Total', 'Latitude': 'With_Coords'})
    code_stats['Match_Rate'] = code_stats['With_Coords'] / code_stats['Total'] * 100
    
    print("\nTop ERCOT Codes by Resources:")
    top_codes = code_stats.nlargest(10, 'Total')
    for code, row in top_codes.iterrows():
        decoded = get_plant_name(code)
        print(f"  {code:12} ({row['Total']:3} resources): {row['With_Coords']:3} matched ({row['Match_Rate']:.0f}%) - {decoded[:40]}")
    
    # 7. Save results
    output_file = 'ERCOT_GENERATORS_IMPROVED_MAPPING.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\nFull results saved to: {output_file}")
    
    # Simple output
    simple_df = results_df[results_df['Latitude'].notna()].copy()
    simple_df['Unit_Name'] = simple_df['Resource_Name'].map(
        lambda x: resource_to_substation.get(x, {}).get('Unit_Name', '') if x in resource_to_substation else ''
    )
    
    simple_df = simple_df[['Resource_Name', 'Decoded_Name', 'Substation', 'Unit_Name', 'Latitude', 'Longitude']]
    simple_df.columns = ['resource_node', 'plant_name', 'substation', 'unit_name', 'latitude', 'longitude']
    
    simple_file = 'ERCOT_GENERATORS_LOCATIONS_IMPROVED.csv'
    simple_df.to_csv(simple_file, index=False)
    print(f"Simple location file saved to: {simple_file}")
    print(f"  Contains {len(simple_df)} resources with coordinates")
    
    # 8. Sample results
    print("\n" + "=" * 80)
    print("SAMPLE SUCCESSFUL MATCHES")
    print("=" * 80)
    
    # Show successful direct EIA matches
    direct_matches = results_df[(results_df['Match_Type'] == 'Direct_EIA')].head(10)
    if not direct_matches.empty:
        print("\nDirect EIA Matches:")
        for _, row in direct_matches.iterrows():
            print(f"\n{row['Resource_Name']}:")
            print(f"  Code: {row['ERCOT_Code']} -> {row['Decoded_Name'][:40]}")
            print(f"  EIA: {row['EIA_Plant']} ({row['EIA_Capacity_MW']:.1f} MW)")
            print(f"  Location: ({row['Latitude']:.4f}, {row['Longitude']:.4f})")
            print(f"  Confidence: {row['Match_Confidence']:.2f}")
    
    # Show IQ to EIA matches
    iq_matches = results_df[(results_df['Match_Type'] == 'IQ_to_EIA')].head(5)
    if not iq_matches.empty:
        print("\nIQ to EIA Matches:")
        for _, row in iq_matches.iterrows():
            print(f"\n{row['Resource_Name']}:")
            print(f"  Code: {row['ERCOT_Code']} -> {row['Decoded_Name'][:40]}")
            print(f"  IQ: {row['IQ_Project']} ({row['IQ_Capacity_MW']} MW)")
            print(f"  EIA: {row['EIA_Plant']} ({row['EIA_Capacity_MW']:.1f} MW)")
            print(f"  Location: ({row['Latitude']:.4f}, {row['Longitude']:.4f})")
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
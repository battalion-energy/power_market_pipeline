#!/usr/bin/env python3
"""
PROPER Generator Mapping Pipeline for ERCOT
Using correct matching logic with IQ as the bridge
"""

import pandas as pd
import numpy as np
import warnings
import re
from typing import Dict, List, Tuple, Optional
warnings.filterwarnings('ignore')

# File paths
INTERCONNECTION_FILE = '/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/ERCOT_InterconnectionQueue/interconnection_gis_report.xlsx'
EIA_PLANT_FILE = '/home/enrico/experiments/ERCOT_SCED/pypsa-usa/workflow/repo_data/plants/eia860_ads_merged.csv'
DAM_RESOURCE_FILE = '/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/60-Day_DAM_Disclosure_Reports/csv/60d_DAM_Gen_Resource_Data-18-JAN-24.csv'
GEN_NODE_MAP = '/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/Settlement_Points_List_and_Electrical_Buses_Mapping/latest_mapping/SP_List_EB_Mapping/gen_node_map.csv'
SETTLEMENT_POINTS = '/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/Settlement_Points_List_and_Electrical_Buses_Mapping/latest_mapping/SP_List_EB_Mapping/Settlement_Points_07242024_210751.csv'

def extract_plant_identifier(ercot_name: str) -> str:
    """Extract the core plant identifier from ERCOT resource name"""
    if pd.isna(ercot_name):
        return ''
    
    name = str(ercot_name).upper()
    
    # Remove unit numbers at the end
    name = re.sub(r'_UNIT\d+$', '', name)
    name = re.sub(r'_\d+$', '', name)
    
    # Remove technology indicators but keep the base name
    name = re.sub(r'_(CC|GT|ST|CT)\d*', '', name)
    name = re.sub(r'_(BES[S]?|BATTERY|STOR)\d*', '', name)
    
    # Handle special patterns
    if '_W_' in name or name.endswith('_W'):
        # COYOTE_W -> COYOTE
        name = name.replace('_W_', '').replace('_W', '')
    
    # For compound names, keep the main identifier
    if '_' in name:
        parts = name.split('_')
        # Keep the most significant part (usually first)
        name = parts[0]
    
    return name

def normalize_for_matching(text: str) -> str:
    """Normalize text for fuzzy matching"""
    if pd.isna(text):
        return ''
    
    text = str(text).upper()
    
    # Remove common suffixes
    suffixes = ['WIND', 'SOLAR', 'FARM', 'PROJECT', 'ENERGY', 'LLC', 'LP', 'INC', 
                'CORP', 'COMPANY', 'POWER', 'GENERATION', 'STATION', 'PLANT']
    for suffix in suffixes:
        text = text.replace(suffix, '')
    
    # Remove special characters
    text = re.sub(r'[^A-Z0-9]', ' ', text)
    
    # Clean up spaces
    text = ' '.join(text.split())
    
    return text.strip()

def match_by_name_and_capacity(name1: str, name2: str, cap1: float, cap2: float, 
                               tolerance_pct: float = 0.15) -> float:
    """
    Match based on name similarity AND capacity similarity
    Returns confidence score (0-1)
    """
    # Normalize names
    norm1 = normalize_for_matching(name1)
    norm2 = normalize_for_matching(name2)
    
    if not norm1 or not norm2:
        return 0.0
    
    # Name matching score
    name_score = 0.0
    if norm1 == norm2:
        name_score = 1.0
    elif norm1 in norm2 or norm2 in norm1:
        # One contains the other
        min_len = min(len(norm1), len(norm2))
        max_len = max(len(norm1), len(norm2))
        if min_len >= 4:  # Require at least 4 characters
            name_score = min_len / max_len
    else:
        # Token-based matching
        tokens1 = set(norm1.split())
        tokens2 = set(norm2.split())
        if tokens1 and tokens2:
            intersection = tokens1 & tokens2
            union = tokens1 | tokens2
            if len(intersection) > 0 and len(union) > 0:
                name_score = len(intersection) / len(union)
    
    # Capacity matching score (if both capacities available)
    cap_score = 0.0
    if pd.notna(cap1) and pd.notna(cap2) and cap1 > 0 and cap2 > 0:
        # Calculate percentage difference
        diff_pct = abs(cap1 - cap2) / max(cap1, cap2)
        if diff_pct <= tolerance_pct:
            cap_score = 1.0 - (diff_pct / tolerance_pct)
    
    # Combined score (weighted average)
    if cap_score > 0:
        # If we have capacity data, weight it heavily
        return 0.4 * name_score + 0.6 * cap_score
    else:
        # Name only
        return name_score * 0.7  # Reduce confidence without capacity validation

def main():
    print("=" * 80)
    print("PROPER GENERATOR MAPPING PIPELINE")
    print("=" * 80)
    
    # 1. Load ERCOT resources
    print("\n1. Loading ERCOT generator resources...")
    dam_df = pd.read_csv(DAM_RESOURCE_FILE)
    resources = dam_df['Resource Name'].unique()
    print(f"   Found {len(resources)} unique ERCOT resources")
    
    ercot_df = pd.DataFrame({'Resource_Name': resources})
    ercot_df['Plant_Identifier'] = ercot_df['Resource_Name'].apply(extract_plant_identifier)
    
    # 2. Load Interconnection Queue (BOTH sheets)
    print("\n2. Loading interconnection queue data...")
    
    # Large Gen
    iq_large = pd.read_excel(INTERCONNECTION_FILE, sheet_name='Project Details - Large Gen', skiprows=30)
    iq_large.columns = iq_large.columns.str.strip()
    iq_large = iq_large[iq_large['INR'].notna()].copy()
    
    # Small Gen
    iq_small = pd.read_excel(INTERCONNECTION_FILE, sheet_name='Project Details - Small Gen', skiprows=14)
    iq_small.columns = iq_small.columns.str.strip()
    iq_small = iq_small[iq_small['INR'].notna()].copy()
    
    # Combine
    iq_df = pd.concat([iq_large, iq_small], ignore_index=True)
    print(f"   Total IQ projects: {len(iq_df)} (Large: {len(iq_large)}, Small: {len(iq_small)})")
    
    # Normalize IQ project names
    iq_df['Normalized_Name'] = iq_df['Project Name'].apply(normalize_for_matching)
    
    # 3. Load EIA data
    print("\n3. Loading EIA plant data...")
    eia_df = pd.read_csv(EIA_PLANT_FILE, low_memory=False)
    eia_tx = eia_df[eia_df['state'] == 'TX'].copy()
    
    # Aggregate by plant
    eia_plants = eia_tx.groupby('plant_code').agg({
        'plant_name': 'first',
        'latitude': 'first',
        'longitude': 'first',
        'nameplate_capacity_mw': 'sum'
    }).reset_index()
    
    print(f"   Found {len(eia_plants)} unique Texas plants with coords")
    
    # Normalize EIA plant names
    eia_plants['Normalized_Name'] = eia_plants['plant_name'].apply(normalize_for_matching)
    
    # 4. Load substation mapping
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
    
    # 5. PROPER MATCHING PROCESS
    print("\n5. Performing three-way matching (ERCOT -> IQ -> EIA)...")
    
    results = []
    matched_count = 0
    
    for _, resource in ercot_df.iterrows():
        result = {
            'Resource_Name': resource['Resource_Name'],
            'Plant_Identifier': resource['Plant_Identifier'],
            'IQ_Project': None,
            'IQ_County': None,
            'IQ_Capacity_MW': None,
            'EIA_Plant': None,
            'Latitude': None,
            'Longitude': None,
            'EIA_Capacity_MW': None,
            'Match_Confidence': 0.0,
            'Match_Method': None,
            'Substation': None,
            'Load_Zone': None
        }
        
        # Get substation data if available
        if resource['Resource_Name'] in resource_to_substation:
            sub_info = resource_to_substation[resource['Resource_Name']]
            result['Substation'] = sub_info['Substation']
            if sub_info['Substation'] in substation_to_zone:
                result['Load_Zone'] = substation_to_zone[sub_info['Substation']]
        
        # Normalize the plant identifier
        norm_identifier = normalize_for_matching(resource['Plant_Identifier'])
        
        if not norm_identifier or len(norm_identifier) < 3:
            results.append(result)
            continue
        
        # STEP 1: Match ERCOT to IQ
        best_iq_match = None
        best_iq_score = 0
        
        for _, iq_row in iq_df.iterrows():
            iq_norm = iq_row['Normalized_Name']
            
            # Check name similarity
            if norm_identifier in iq_norm or iq_norm in norm_identifier:
                # Calculate match score
                score = len(norm_identifier) / max(len(norm_identifier), len(iq_norm))
                
                if score > best_iq_score and score > 0.5:
                    best_iq_score = score
                    best_iq_match = iq_row
        
        if best_iq_match is not None:
            result['IQ_Project'] = best_iq_match['Project Name']
            result['IQ_County'] = best_iq_match['County'] if 'County' in best_iq_match else None
            result['IQ_Capacity_MW'] = best_iq_match['Capacity (MW)'] if 'Capacity (MW)' in best_iq_match else None
            
            # STEP 2: Match IQ to EIA
            best_eia_match = None
            best_eia_score = 0
            
            for _, eia_row in eia_plants.iterrows():
                # Match based on name AND capacity
                score = match_by_name_and_capacity(
                    best_iq_match['Project Name'],
                    eia_row['plant_name'],
                    best_iq_match['Capacity (MW)'] if 'Capacity (MW)' in best_iq_match else None,
                    eia_row['nameplate_capacity_mw']
                )
                
                if score > best_eia_score and score > 0.6:
                    best_eia_score = score
                    best_eia_match = eia_row
            
            if best_eia_match is not None:
                result['EIA_Plant'] = best_eia_match['plant_name']
                result['Latitude'] = best_eia_match['latitude']
                result['Longitude'] = best_eia_match['longitude']
                result['EIA_Capacity_MW'] = best_eia_match['nameplate_capacity_mw']
                result['Match_Confidence'] = best_iq_score * best_eia_score
                result['Match_Method'] = 'ERCOT->IQ->EIA'
                matched_count += 1
        
        results.append(result)
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # 6. Statistics
    print("\n" + "=" * 80)
    print("MATCHING RESULTS")
    print("=" * 80)
    
    total = len(results_df)
    has_coords = results_df['Latitude'].notna().sum()
    has_iq = results_df['IQ_Project'].notna().sum()
    has_capacity = results_df['IQ_Capacity_MW'].notna().sum()
    
    print(f"\nTotal Resources: {total}")
    print(f"Matched to IQ: {has_iq} ({has_iq/total*100:.1f}%)")
    print(f"Matched to EIA (with coords): {has_coords} ({has_coords/total*100:.1f}%)")
    print(f"With Capacity Data: {has_capacity} ({has_capacity/total*100:.1f}%)")
    
    # Check confidence distribution
    matched = results_df[results_df['Match_Confidence'] > 0]
    if len(matched) > 0:
        print(f"\nConfidence Score Distribution:")
        print(f"  High (>0.8): {len(matched[matched['Match_Confidence'] > 0.8])}")
        print(f"  Medium (0.6-0.8): {len(matched[(matched['Match_Confidence'] >= 0.6) & (matched['Match_Confidence'] <= 0.8)])}")
        print(f"  Low (<0.6): {len(matched[matched['Match_Confidence'] < 0.6])}")
    
    # 7. Save results
    output_file = 'ERCOT_GENERATORS_PROPER_MAPPING.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    # Simple output
    simple_df = results_df[results_df['Latitude'].notna()].copy()
    simple_df['Unit_Name'] = simple_df['Resource_Name'].map(
        lambda x: resource_to_substation.get(x, {}).get('Unit_Name', '') if x in resource_to_substation else ''
    )
    
    simple_df = simple_df[['Resource_Name', 'EIA_Plant', 'Substation', 'Unit_Name', 'Latitude', 'Longitude']]
    simple_df.columns = ['resource_node', 'plant_name', 'substation', 'unit_name', 'latitude', 'longitude']
    
    simple_file = 'ERCOT_GENERATORS_LOCATIONS_PROPER.csv'
    simple_df.to_csv(simple_file, index=False)
    print(f"Simple location file saved to: {simple_file}")
    print(f"  Contains {len(simple_df)} resources with coordinates")
    
    # 8. Sample results
    print("\n" + "=" * 80)
    print("SAMPLE SUCCESSFUL MATCHES")
    print("=" * 80)
    
    # Show high-confidence matches
    high_conf = results_df[results_df['Match_Confidence'] > 0.7].head(10)
    for _, row in high_conf.iterrows():
        print(f"\n{row['Resource_Name']}:")
        print(f"  IQ: {row['IQ_Project']} ({row['IQ_Capacity_MW']} MW)")
        print(f"  EIA: {row['EIA_Plant']} ({row['EIA_Capacity_MW']} MW)")
        print(f"  Location: ({row['Latitude']:.4f}, {row['Longitude']:.4f})")
        print(f"  Confidence: {row['Match_Confidence']:.2f}")
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
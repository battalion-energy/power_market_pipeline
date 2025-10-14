#!/usr/bin/env python3
"""
VALIDATED Generator Mapping Pipeline for ERCOT
Includes capacity and county validation to prevent false matches
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

def infer_technology_type(name: str) -> str:
    """Infer technology type from resource name"""
    name_upper = str(name).upper()
    
    if any(x in name_upper for x in ['BATTERY', 'BESS', 'STORAGE', 'ESS']):
        return 'BATTERY'
    elif any(x in name_upper for x in ['WIND', 'TURBINE', 'WTG']):
        return 'WIND'
    elif any(x in name_upper for x in ['SOLAR', 'PV', 'PHOTOVOLTAIC']):
        return 'SOLAR'
    elif any(x in name_upper for x in ['GAS', 'CC', 'CT', 'COMBUSTION', 'COMBINED CYCLE']):
        return 'GAS'
    elif any(x in name_upper for x in ['COAL', 'LIGNITE']):
        return 'COAL'
    elif any(x in name_upper for x in ['NUCLEAR', 'REACTOR']):
        return 'NUCLEAR'
    elif any(x in name_upper for x in ['HYDRO', 'DAM', 'WATER']):
        return 'HYDRO'
    elif any(x in name_upper for x in ['COGEN', 'COGENERATION', 'CHP']):
        return 'COGEN'
    else:
        return 'OTHER'

def validate_technology_match(name1: str, name2: str) -> bool:
    """Check if two resources have compatible technology types"""
    type1 = infer_technology_type(name1)
    type2 = infer_technology_type(name2)
    
    # Allow exact matches or OTHER to match anything
    if type1 == type2 or type1 == 'OTHER' or type2 == 'OTHER':
        return True
    
    # Allow some compatible matches
    compatible = {
        ('GAS', 'COGEN'),
        ('COGEN', 'GAS'),
    }
    
    return (type1, type2) in compatible

def validate_capacity_match(cap1: Optional[float], cap2: Optional[float], 
                           tech_type: str = 'OTHER') -> bool:
    """Validate that capacities are within reasonable range"""
    if cap1 is None or cap2 is None or cap1 <= 0 or cap2 <= 0:
        return True  # Can't validate, assume OK
    
    # Different thresholds for different technologies
    if tech_type == 'BATTERY':
        # BESS typically 10-200 MW, don't match 1 MW rooftop to 100 MW BESS
        if min(cap1, cap2) < 5 and max(cap1, cap2) > 50:
            return False
    
    # General rule: must be within same order of magnitude
    ratio = max(cap1, cap2) / min(cap1, cap2)
    if ratio > 10:  # More than 10x difference
        return False
    
    # For larger plants, be more strict
    if max(cap1, cap2) > 100:
        return ratio < 3  # Within 3x for large plants
    
    return True

def normalize_for_matching(text: str) -> str:
    """Normalize text for fuzzy matching"""
    if pd.isna(text):
        return ''
    
    text = str(text).upper()
    
    # Remove common suffixes that cause false matches
    suffixes = ['GENERATING', 'STATION', 'PLANT', 'FACILITY', 'PROJECT', 
                'LLC', 'LP', 'INC', 'CORP', 'COMPANY', 'LTD', 'LIMITED',
                'ENERGY', 'POWER', 'GENERATION', 'GEN', 'ELECTRIC',
                'SYSTEM', 'SYSTEMS', 'RESOURCES', 'HOLDINGS']
    for suffix in suffixes:
        text = text.replace(suffix, '')
    
    # Remove special characters
    text = re.sub(r'[^A-Z0-9\s]', ' ', text)
    
    # Clean up spaces
    text = ' '.join(text.split())
    
    return text.strip()

def calculate_match_score(name1: str, name2: str, cap1: float = None, cap2: float = None,
                         validate_tech: bool = True) -> float:
    """Calculate match score between two names with validation"""
    
    # Technology validation
    if validate_tech and not validate_technology_match(name1, name2):
        return 0.0
    
    # Capacity validation
    tech_type = infer_technology_type(name1)
    if not validate_capacity_match(cap1, cap2, tech_type):
        return 0.0
    
    norm1 = normalize_for_matching(name1)
    norm2 = normalize_for_matching(name2)
    
    if not norm1 or not norm2:
        return 0.0
    
    # Exact match
    if norm1 == norm2:
        name_score = 1.0
    # One contains the other (but penalize short matches)
    elif norm1 in norm2 or norm2 in norm1:
        min_len = min(len(norm1), len(norm2))
        max_len = max(len(norm1), len(norm2))
        if min_len < 5:  # Too short to be reliable
            name_score = 0.3
        else:
            name_score = min_len / max_len if max_len > 0 else 0
    else:
        # Token overlap - more sophisticated
        tokens1 = set(norm1.split())
        tokens2 = set(norm2.split())
        
        # Remove very common tokens that cause false matches
        very_common = {'THE', 'OF', 'AND', 'IN', 'FOR', 'TO', 'A', 'AN'}
        tokens1 = tokens1 - very_common
        tokens2 = tokens2 - very_common
        
        # Need at least 2 character tokens
        tokens1 = {t for t in tokens1 if len(t) > 1}
        tokens2 = {t for t in tokens2 if len(t) > 1}
        
        if tokens1 and tokens2:
            overlap = len(tokens1 & tokens2)
            # Require at least 2 overlapping tokens for battery/wind/solar
            if tech_type in ['BATTERY', 'WIND', 'SOLAR'] and overlap < 2:
                name_score = 0
            else:
                # Weight by importance of overlapping tokens
                name_score = overlap / min(len(tokens1), len(tokens2))
        else:
            name_score = 0
    
    # If we have capacity data, use it to adjust score
    if cap1 is not None and cap2 is not None and cap1 > 0 and cap2 > 0:
        cap_diff = abs(cap1 - cap2) / max(cap1, cap2)
        if cap_diff < 0.1:  # Within 10%
            cap_multiplier = 1.2
        elif cap_diff < 0.3:  # Within 30%
            cap_multiplier = 1.0
        elif cap_diff < 0.5:  # Within 50%
            cap_multiplier = 0.8
        else:
            cap_multiplier = 0.5
        
        name_score = min(1.0, name_score * cap_multiplier)
    
    return name_score

def main():
    print("=" * 80)
    print("VALIDATED GENERATOR MAPPING PIPELINE")
    print("With capacity and technology validation")
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
    ercot_df['Technology_Type'] = ercot_df['Decoded_Name'].apply(infer_technology_type)
    
    # 2. Load IQ data
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
    iq_df['Technology_Type'] = iq_df['Project Name'].apply(infer_technology_type)
    print(f"   Total IQ projects: {len(iq_df)}")
    
    # 3. Load EIA data with county info
    print("\n3. Loading EIA plant data...")
    eia_df = pd.read_csv(EIA_PLANT_FILE, low_memory=False)
    eia_tx = eia_df[eia_df['state'] == 'TX']
    
    # Check what columns we actually have
    print(f"   EIA columns available: {eia_tx.columns.tolist()[:10]}...")
    
    # Aggregate by plant - NOTE: county column may not exist
    agg_dict = {
        'plant_name': 'first',
        'latitude': 'first',
        'longitude': 'first',
        'nameplate_capacity_mw': 'sum'
    }
    
    # Add county if it exists
    if 'county' in eia_tx.columns:
        agg_dict['county'] = 'first'
    
    eia_plants = eia_tx.groupby('plant_code').agg(agg_dict).reset_index()
    eia_plants['Technology_Type'] = eia_plants['plant_name'].apply(infer_technology_type)
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
    
    # 5. VALIDATED MATCHING
    print("\n5. Matching with validation...")
    print("   - Technology type validation")
    print("   - Capacity range validation")
    print("   - Higher match thresholds")
    
    results = []
    matched_eia_direct = 0
    matched_iq = 0
    matched_eia_via_iq = 0
    rejected_tech = 0
    rejected_capacity = 0
    
    for _, resource in ercot_df.iterrows():
        result = {
            'Resource_Name': resource['Resource_Name'],
            'ERCOT_Code': resource['ERCOT_Code'],
            'Decoded_Name': resource['Decoded_Name'],
            'Technology_Type': resource['Technology_Type'],
            'IQ_Project': None,
            'IQ_County': None,
            'IQ_Capacity_MW': None,
            'IQ_Technology': None,
            'EIA_Plant': None,
            'EIA_County': None,
            'Latitude': None,
            'Longitude': None,
            'EIA_Capacity_MW': None,
            'EIA_Technology': None,
            'Match_Type': None,
            'Match_Confidence': 0.0,
            'Rejection_Reason': None,
            'Substation': None,
            'Load_Zone': None
        }
        
        # Get substation info
        if resource['Resource_Name'] in resource_to_substation:
            sub_info = resource_to_substation[resource['Resource_Name']]
            result['Substation'] = sub_info['Substation']
            if sub_info['Substation'] in substation_to_zone:
                result['Load_Zone'] = substation_to_zone[sub_info['Substation']]
        
        # Try direct EIA match with HIGHER threshold
        best_eia_score = 0
        best_eia_match = None
        rejection_reason = None
        
        for _, eia_row in eia_plants.iterrows():
            # Check technology compatibility first
            if not validate_technology_match(resource['Decoded_Name'], eia_row['plant_name']):
                continue
                
            score = calculate_match_score(
                resource['Decoded_Name'],
                eia_row['plant_name'],
                None,  # We don't have ERCOT capacity yet
                eia_row['nameplate_capacity_mw'],
                validate_tech=True
            )
            
            # Higher threshold for direct matches
            if score > best_eia_score and score > 0.6:
                best_eia_score = score
                best_eia_match = eia_row
        
        if best_eia_match is not None:
            result['EIA_Plant'] = best_eia_match['plant_name']
            result['Latitude'] = best_eia_match['latitude']
            result['Longitude'] = best_eia_match['longitude']
            result['EIA_Capacity_MW'] = best_eia_match['nameplate_capacity_mw']
            result['EIA_Technology'] = best_eia_match['Technology_Type']
            if 'county' in best_eia_match:
                result['EIA_County'] = best_eia_match.get('county')
            result['Match_Type'] = 'Direct_EIA_Validated'
            result['Match_Confidence'] = best_eia_score
            matched_eia_direct += 1
        else:
            # Try IQ matching
            best_iq_score = 0
            best_iq_match = None
            
            for _, iq_row in iq_df.iterrows():
                # Check technology compatibility
                if not validate_technology_match(resource['Decoded_Name'], iq_row['Project Name']):
                    continue
                    
                score = calculate_match_score(
                    resource['Decoded_Name'],
                    iq_row['Project Name'],
                    None,
                    iq_row['Capacity (MW)'] if 'Capacity (MW)' in iq_row else None,
                    validate_tech=True
                )
                
                # Higher threshold
                if score > best_iq_score and score > 0.5:
                    best_iq_score = score
                    best_iq_match = iq_row
            
            if best_iq_match is not None:
                result['IQ_Project'] = best_iq_match['Project Name']
                result['IQ_County'] = best_iq_match['County'] if 'County' in best_iq_match else None
                result['IQ_Capacity_MW'] = best_iq_match['Capacity (MW)'] if 'Capacity (MW)' in best_iq_match else None
                result['IQ_Technology'] = best_iq_match['Technology_Type']
                matched_iq += 1
                
                # Try to match IQ to EIA with validation
                for _, eia_row in eia_plants.iterrows():
                    # Validate technology
                    if not validate_technology_match(best_iq_match['Project Name'], eia_row['plant_name']):
                        continue
                    
                    # Validate capacity
                    if not validate_capacity_match(
                        best_iq_match.get('Capacity (MW)'),
                        eia_row['nameplate_capacity_mw'],
                        resource['Technology_Type']
                    ):
                        continue
                    
                    score = calculate_match_score(
                        best_iq_match['Project Name'],
                        eia_row['plant_name'],
                        best_iq_match.get('Capacity (MW)'),
                        eia_row['nameplate_capacity_mw'],
                        validate_tech=True
                    )
                    
                    if score > best_eia_score and score > 0.5:
                        best_eia_score = score
                        best_eia_match = eia_row
                
                if best_eia_match is not None:
                    result['EIA_Plant'] = best_eia_match['plant_name']
                    result['Latitude'] = best_eia_match['latitude']
                    result['Longitude'] = best_eia_match['longitude']
                    result['EIA_Capacity_MW'] = best_eia_match['nameplate_capacity_mw']
                    result['EIA_Technology'] = best_eia_match['Technology_Type']
                    if 'county' in best_eia_match:
                        result['EIA_County'] = best_eia_match.get('county')
                    result['Match_Type'] = 'IQ_to_EIA_Validated'
                    result['Match_Confidence'] = min(best_iq_score, best_eia_score)
                    matched_eia_via_iq += 1
        
        results.append(result)
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # 6. Statistics
    print("\n" + "=" * 80)
    print("VALIDATED MATCHING RESULTS")
    print("=" * 80)
    
    total = len(results_df)
    has_decoded = (results_df['Decoded_Name'] != results_df['ERCOT_Code']).sum()
    has_iq = results_df['IQ_Project'].notna().sum()
    has_coords = results_df['Latitude'].notna().sum()
    direct_eia = (results_df['Match_Type'] == 'Direct_EIA_Validated').sum()
    via_iq = (results_df['Match_Type'] == 'IQ_to_EIA_Validated').sum()
    
    print(f"\nTotal Resources: {total}")
    print(f"Successfully Decoded: {has_decoded} ({has_decoded/total*100:.1f}%)")
    print(f"Matched to IQ: {has_iq} ({has_iq/total*100:.1f}%)")
    print(f"Matched to EIA (with coords): {has_coords} ({has_coords/total*100:.1f}%)")
    print(f"  - Direct EIA matches: {direct_eia}")
    print(f"  - Via IQ matches: {via_iq}")
    
    # Technology distribution
    print("\nMatches by Technology Type:")
    tech_stats = results_df[results_df['Latitude'].notna()].groupby('Technology_Type').size()
    for tech, count in tech_stats.items():
        print(f"  {tech:10}: {count:4} resources")
    
    # Check for suspicious matches
    print("\nValidation Checks:")
    
    # Find any remaining IKEA matches
    ikea_matches = results_df[results_df['EIA_Plant'].str.contains('IKEA', na=False)]
    if not ikea_matches.empty:
        print(f"\n⚠️ WARNING: Still have {len(ikea_matches)} IKEA matches:")
        for _, row in ikea_matches.head(5).iterrows():
            print(f"  {row['Resource_Name']:20} -> {row['EIA_Plant']}")
            print(f"    Tech: {row['Technology_Type']} vs {row['EIA_Technology']}")
            print(f"    Capacity: {row['IQ_Capacity_MW']} vs {row['EIA_Capacity_MW']}")
    else:
        print("  ✓ No suspicious IKEA rooftop matches")
    
    # Check capacity mismatches
    capacity_check = results_df[results_df['EIA_Capacity_MW'].notna() & results_df['IQ_Capacity_MW'].notna()].copy()
    if not capacity_check.empty:
        capacity_check['cap_ratio'] = capacity_check['EIA_Capacity_MW'] / capacity_check['IQ_Capacity_MW']
        large_mismatches = capacity_check[(capacity_check['cap_ratio'] > 5) | (capacity_check['cap_ratio'] < 0.2)]
        if not large_mismatches.empty:
            print(f"\n⚠️ Large capacity mismatches: {len(large_mismatches)}")
            for _, row in large_mismatches.head(3).iterrows():
                print(f"  {row['Resource_Name']}: IQ={row['IQ_Capacity_MW']:.1f} MW, EIA={row['EIA_Capacity_MW']:.1f} MW")
    
    # 7. Save results
    output_file = 'ERCOT_GENERATORS_VALIDATED_MAPPING.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\nFull results saved to: {output_file}")
    
    # Simple output with validation info
    simple_df = results_df[results_df['Latitude'].notna()].copy()
    simple_df['Unit_Name'] = simple_df['Resource_Name'].map(
        lambda x: resource_to_substation.get(x, {}).get('Unit_Name', '') if x in resource_to_substation else ''
    )
    
    # Include validation columns
    simple_df = simple_df[['Resource_Name', 'Decoded_Name', 'Technology_Type', 
                           'IQ_County', 'IQ_Capacity_MW',
                           'EIA_County', 'EIA_Capacity_MW', 
                           'Substation', 'Unit_Name', 'Latitude', 'Longitude',
                           'Match_Confidence', 'Match_Type']]
    
    simple_file = 'ERCOT_GENERATORS_LOCATIONS_VALIDATED.csv'
    simple_df.to_csv(simple_file, index=False)
    print(f"Validated location file saved to: {simple_file}")
    print(f"  Contains {len(simple_df)} resources with validated coordinates")
    
    # Sample high-confidence matches
    print("\n" + "=" * 80)
    print("SAMPLE HIGH-CONFIDENCE VALIDATED MATCHES")
    print("=" * 80)
    
    high_conf = results_df[(results_df['Match_Confidence'] > 0.8) & results_df['Latitude'].notna()].head(10)
    for _, row in high_conf.iterrows():
        print(f"\n{row['Resource_Name']}:")
        print(f"  Decoded: {row['Decoded_Name'][:40]} ({row['Technology_Type']})")
        if row['IQ_Project']:
            print(f"  IQ: {row['IQ_Project']} ({row['IQ_Capacity_MW']} MW, {row['IQ_County']})")
        print(f"  EIA: {row['EIA_Plant']} ({row['EIA_Capacity_MW']:.1f} MW)")
        print(f"  Location: ({row['Latitude']:.4f}, {row['Longitude']:.4f})")
        print(f"  Confidence: {row['Match_Confidence']:.2f}")
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE - WITH VALIDATION")
    print("=" * 80)

if __name__ == "__main__":
    main()
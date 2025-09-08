#!/usr/bin/env python3
"""
Complete Generator Mapping Pipeline FINAL
Uses both Large and Small Gen sheets from interconnection queue
Conservative matching to avoid false positives
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
EIA_ERCOT_MAPPING = '/home/enrico/projects/battalion-platform/scripts/data-loaders/eia_ercot_mapping_20250819_123244.csv'
DAM_RESOURCE_FILE = '/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/60-Day_DAM_Disclosure_Reports/csv/60d_DAM_Gen_Resource_Data-18-JAN-24.csv'
GEN_NODE_MAP = '/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/Settlement_Points_List_and_Electrical_Buses_Mapping/latest_mapping/SP_List_EB_Mapping/gen_node_map.csv'
SETTLEMENT_POINTS = '/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/Settlement_Points_List_and_Electrical_Buses_Mapping/latest_mapping/SP_List_EB_Mapping/Settlement_Points_07242024_210751.csv'

# High-confidence ERCOT to EIA plant name mappings
# Only include mappings we're very confident about
HIGH_CONFIDENCE_MAPPINGS = {
    # Major Gas Plants
    'CHE': 'Comanche Peak',
    'FORMOSA': 'Formosa Utility Venture Ltd',
    'BRAUNIG': 'V H Braunig', 
    'TOPAZ': 'Topaz Generating',
    'VICTORIA': 'Victoria',
    'WGU': 'Newgulf Cogen',
    
    # Known Wind Farms (verified)
    'AVIATOR': 'Aviator Wind',
    'ALGODON': 'El Algodon Alto Wind Farm, LLC',
    'ANACACHO': 'Anacacho Wind Farm, LLC',
    'BRISCOE': 'Briscoe Wind Farm',
    'CHALUPA': 'La Chalupa, LLC',
    'CHAMPION': 'Champion Wind Farm LLC',
    'COYOTE_W': 'Coyote Wind LLC',
    'KEECHI': 'Keechi Wind',
    'PRIDDY': 'Priddy Wind Project',
    'RELOJ': 'Reloj del Sol Wind Farm',
    
    # Known Solar (verified)
    'AZURE': 'Azure Sky Solar',
    
    # Known Battery (verified)
    'CROSSETT': 'Crossett Power Management LLC',
    'RAB': 'Rabbit Hill Energy Storage Project',
}

def extract_base_name(resource_name: str) -> str:
    """Extract base plant name from resource name"""
    if pd.isna(resource_name):
        return ''
    
    name = str(resource_name).upper()
    
    # Remove unit designators
    patterns = [
        r'[_-](UNIT|GEN|RN|ALL|G|GT|CC|ST|CT|PV|WD|BES[S]?)\d*$',
        r'[_-]\d+$',
        r'[_-][A-Z]\d*$'
    ]
    
    for pattern in patterns:
        name = re.sub(pattern, '', name)
    
    # Remove other suffixes
    name = name.replace('_AMISTAG', '')
    name = name.replace('_BESS', '')
    name = name.replace('_WIND', '')
    name = name.replace('_SOLAR', '')
    
    return name.strip()

def categorize_resource_type(resource_name: str) -> str:
    """Categorize resource by type based on name patterns"""
    if pd.isna(resource_name):
        return 'Unknown'
    
    name = str(resource_name).upper()
    
    # Battery/BESS
    if 'BESS' in name or 'BATTERY' in name or 'STOR' in name or '_ESS' in name:
        return 'Battery'
    
    # Solar
    if '_PV' in name or 'SOLAR' in name or '_SUN' in name or 'SLR' in name:
        return 'Solar'
    
    # Wind
    if '_WD' in name or 'WIND' in name or 'WND' in name:
        return 'Wind'
    
    # Gas turbines
    if '_CC' in name or '_GT' in name or '_CT' in name or '_ST' in name:
        return 'Gas'
    
    # Nuclear
    if 'NUC' in name or 'STP' in name:
        return 'Nuclear'
    
    # Coal
    if 'COAL' in name or '_LIG' in name:
        return 'Coal'
    
    # Hydro
    if 'HYDRO' in name or '_HY' in name:
        return 'Hydro'
    
    return 'Other'

def conservative_fuzzy_match(name1: str, name2: str) -> float:
    """Conservative fuzzy matching - requires significant similarity"""
    if pd.isna(name1) or pd.isna(name2):
        return 0.0
    
    name1 = str(name1).upper().replace('_', ' ').replace('-', ' ').strip()
    name2 = str(name2).upper().replace('_', ' ').replace('-', ' ').strip()
    
    # Exact match
    if name1 == name2:
        return 1.0
    
    # One fully contains the other (but not just a single character)
    if len(name1) > 3 and len(name2) > 3:
        if name1 in name2 or name2 in name1:
            return 0.85
    
    # Significant token overlap (at least 2 meaningful tokens)
    tokens1 = set(w for w in name1.split() if len(w) > 2)
    tokens2 = set(w for w in name2.split() if len(w) > 2)
    
    if len(tokens1) >= 2 and len(tokens2) >= 2:
        overlap = len(tokens1 & tokens2)
        if overlap >= 2:
            return 0.7
    
    # No significant match
    return 0.0

def main():
    print("=" * 80)
    print("COMPLETE GENERATOR MAPPING PIPELINE - FINAL VERSION")
    print("=" * 80)
    
    # 1. Load ERCOT generator resources
    print("\n1. Loading ERCOT generator resources...")
    dam_df = pd.read_csv(DAM_RESOURCE_FILE)
    resources = dam_df['Resource Name'].unique()
    print(f"   Found {len(resources)} unique ERCOT resources")
    
    # Create base dataframe
    ercot_df = pd.DataFrame({
        'Resource_Name': resources
    })
    
    # Add resource type
    ercot_df['Resource_Type'] = ercot_df['Resource_Name'].apply(categorize_resource_type)
    ercot_df['Base_Name'] = ercot_df['Resource_Name'].apply(extract_base_name)
    
    print(f"\nResource type breakdown:")
    print(ercot_df['Resource_Type'].value_counts())
    
    # 2. Load generator node mapping
    print("\n2. Loading generator node to substation mapping...")
    gen_map_df = pd.read_csv(GEN_NODE_MAP)
    print(f"   Found {len(gen_map_df)} resource node mappings")
    
    # Create a mapping dictionary
    resource_to_substation = {}
    for _, row in gen_map_df.iterrows():
        resource_to_substation[row['RESOURCE_NODE']] = {
            'Substation': row['UNIT_SUBSTATION'],
            'Unit_Name': row['UNIT_NAME']
        }
    
    # 3. Load settlement point mapping
    print("\n3. Loading settlement point and electrical bus mapping...")
    sp_df = pd.read_csv(SETTLEMENT_POINTS)
    print(f"   Found {len(sp_df)} settlement point mappings")
    
    # Create substation to load zone mapping
    substation_to_zone = {}
    for _, row in sp_df.iterrows():
        if pd.notna(row['SUBSTATION']) and pd.notna(row['SETTLEMENT_LOAD_ZONE']):
            substation_to_zone[row['SUBSTATION']] = row['SETTLEMENT_LOAD_ZONE']
    
    # 4. Load interconnection queue data (BOTH sheets)
    print("\n4. Loading interconnection queue data (Large and Small Gen)...")
    
    # Load Large Gen
    iq_large = pd.read_excel(INTERCONNECTION_FILE, sheet_name='Project Details - Large Gen', skiprows=30)
    iq_large.columns = iq_large.columns.str.strip()
    iq_large = iq_large[iq_large['INR'].notna()].copy()
    
    # Load Small Gen
    iq_small = pd.read_excel(INTERCONNECTION_FILE, sheet_name='Project Details - Small Gen', skiprows=14)
    iq_small.columns = iq_small.columns.str.strip()
    iq_small = iq_small[iq_small['INR'].notna()].copy()
    
    # Combine both
    iq_df = pd.concat([iq_large, iq_small], ignore_index=True)
    print(f"   Large Gen: {len(iq_large)} projects")
    print(f"   Small Gen: {len(iq_small)} projects")
    print(f"   Total: {len(iq_df)} projects")
    
    # Create lookup for interconnection projects
    iq_lookup = {}
    for _, proj in iq_df.iterrows():
        if pd.notna(proj['Project Name']):
            # Extract base name from project
            proj_base = proj['Project Name'].upper()
            for suffix in ['WIND', 'SOLAR', 'BESS', 'BATTERY', 'REPOWER', 'HYBRID']:
                proj_base = proj_base.replace(suffix, '').strip()
            
            iq_lookup[proj_base] = {
                'Name': proj['Project Name'],
                'County': proj['County'] if 'County' in proj else None,
                'Capacity': proj['Capacity (MW)'] if 'Capacity (MW)' in proj else None,
                'Fuel': proj['Fuel'] if 'Fuel' in proj else None
            }
    
    # 5. Load EIA plant data
    print("\n5. Loading EIA plant location data...")
    eia_df = pd.read_csv(EIA_PLANT_FILE, low_memory=False)
    
    # Filter to Texas plants
    eia_tx = eia_df[eia_df['state'] == 'TX'].copy()
    print(f"   Found {len(eia_tx)} Texas generators in EIA data")
    
    # Group by plant for location data
    eia_plants = eia_tx.groupby('plant_code').agg({
        'plant_name': 'first',
        'latitude': 'first',
        'longitude': 'first',
        'nameplate_capacity_mw': 'sum',
        'fuel_type': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0] if len(x) > 0 else None,
        'technology': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0] if len(x) > 0 else None
    }).reset_index()
    
    print(f"   {len(eia_plants)} unique plants with coordinates: {eia_plants['latitude'].notna().sum()}")
    
    # 6. Load EIA-ERCOT mapping
    print("\n6. Loading EIA-ERCOT mapping...")
    mapping_df = pd.read_csv(EIA_ERCOT_MAPPING)
    # Remove duplicate/incorrect SOLARA mappings
    mapping_df = mapping_df[mapping_df['ercot_node'] != 'SOLARA_UNIT1']
    print(f"   Found {len(mapping_df)} existing mappings")
    
    # 7. Match resources to locations
    print("\n7. Matching resources to locations (conservative approach)...")
    
    results = []
    matched_count = 0
    
    for _, resource in ercot_df.iterrows():
        result = {
            'Resource_Name': resource['Resource_Name'],
            'Resource_Type': resource['Resource_Type'],
            'Base_Name': resource['Base_Name'],
            'Substation': None,
            'Settlement_Point': None,
            'Load_Zone': None,
            'Plant_Name': None,
            'County': None,
            'Latitude': None,
            'Longitude': None,
            'Capacity_MW': None,
            'Match_Source': None,
            'Match_Confidence': 0.0
        }
        
        # Get substation and load zone from gen_node_map
        if resource['Resource_Name'] in resource_to_substation:
            substation_info = resource_to_substation[resource['Resource_Name']]
            result['Substation'] = substation_info['Substation']
            
            # Get load zone from substation
            if substation_info['Substation'] in substation_to_zone:
                result['Load_Zone'] = substation_to_zone[substation_info['Substation']]
        
        # Find settlement point from SP mapping
        sp_matches = sp_df[sp_df['RESOURCE_NODE'] == resource['Resource_Name']]
        if len(sp_matches) > 0:
            result['Settlement_Point'] = sp_matches.iloc[0]['NODE_NAME']
            if pd.isna(result['Load_Zone']):
                result['Load_Zone'] = sp_matches.iloc[0]['SETTLEMENT_LOAD_ZONE']
        
        # Try high-confidence mappings first
        base_name = resource['Base_Name']
        if base_name in HIGH_CONFIDENCE_MAPPINGS:
            known_plant_name = HIGH_CONFIDENCE_MAPPINGS[base_name]
            
            # Find in EIA data
            eia_matches = eia_plants[eia_plants['plant_name'] == known_plant_name]
            if len(eia_matches) > 0:
                match = eia_matches.iloc[0]
                result['Plant_Name'] = match['plant_name']
                result['Latitude'] = match['latitude']
                result['Longitude'] = match['longitude']
                result['Capacity_MW'] = match['nameplate_capacity_mw']
                result['Match_Source'] = 'High_Confidence_Mapping'
                result['Match_Confidence'] = 0.95
                matched_count += 1
        
        # Try direct EIA-ERCOT mapping
        if result['Latitude'] is None:
            direct_map = mapping_df[mapping_df['ercot_node'] == resource['Resource_Name']]
            if len(direct_map) > 0:
                eia_plant_id = direct_map.iloc[0]['eia_plant_id']
                eia_match = eia_plants[eia_plants['plant_code'] == eia_plant_id]
                
                if len(eia_match) > 0:
                    result['Plant_Name'] = eia_match.iloc[0]['plant_name']
                    result['Latitude'] = eia_match.iloc[0]['latitude']
                    result['Longitude'] = eia_match.iloc[0]['longitude']
                    result['Capacity_MW'] = eia_match.iloc[0]['nameplate_capacity_mw']
                    result['Match_Source'] = 'Direct_EIA_Mapping'
                    result['Match_Confidence'] = 0.9
                    matched_count += 1
        
        # Try matching through interconnection queue
        if result['Latitude'] is None and base_name:
            # Look for exact match in interconnection queue
            if base_name in iq_lookup:
                iq_info = iq_lookup[base_name]
                result['County'] = iq_info['County']
                result['Capacity_MW'] = iq_info['Capacity']
                
                # Try to find matching EIA plant by name
                if iq_info['Name']:
                    # Conservative matching - only if names are very similar
                    for _, eia_plant in eia_plants.iterrows():
                        score = conservative_fuzzy_match(iq_info['Name'], eia_plant['plant_name'])
                        if score > 0.8:  # High threshold
                            result['Plant_Name'] = eia_plant['plant_name']
                            result['Latitude'] = eia_plant['latitude']
                            result['Longitude'] = eia_plant['longitude']
                            if pd.isna(result['Capacity_MW']):
                                result['Capacity_MW'] = eia_plant['nameplate_capacity_mw']
                            result['Match_Source'] = 'IQ_to_EIA_Match'
                            result['Match_Confidence'] = score
                            matched_count += 1
                            break
        
        # Only use conservative fuzzy matching as last resort
        if result['Latitude'] is None and base_name and len(base_name) > 4:
            best_score = 0
            best_match = None
            
            for _, eia_plant in eia_plants.iterrows():
                score = conservative_fuzzy_match(base_name, eia_plant['plant_name'])
                if score > best_score and score > 0.85:  # Very high threshold
                    best_score = score
                    best_match = eia_plant
            
            if best_match is not None:
                result['Plant_Name'] = best_match['plant_name']
                result['Latitude'] = best_match['latitude']
                result['Longitude'] = best_match['longitude']
                result['Capacity_MW'] = best_match['nameplate_capacity_mw']
                result['Match_Source'] = 'Conservative_Fuzzy'
                result['Match_Confidence'] = best_score
                matched_count += 1
        
        results.append(result)
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # 8. Statistics
    print("\n" + "=" * 80)
    print("MAPPING RESULTS")
    print("=" * 80)
    
    total = len(results_df)
    has_coords = results_df['Latitude'].notna().sum()
    has_county = results_df['County'].notna().sum()
    has_capacity = results_df['Capacity_MW'].notna().sum()
    has_substation = results_df['Substation'].notna().sum()
    has_load_zone = results_df['Load_Zone'].notna().sum()
    
    print(f"\nTotal Resources: {total}")
    print(f"With Coordinates: {has_coords} ({has_coords/total*100:.1f}%)")
    print(f"With County (from IQ): {has_county} ({has_county/total*100:.1f}%)")
    print(f"With Substation: {has_substation} ({has_substation/total*100:.1f}%)")
    print(f"With Load Zone: {has_load_zone} ({has_load_zone/total*100:.1f}%)")
    print(f"With Capacity: {has_capacity} ({has_capacity/total*100:.1f}%)")
    
    print("\nMatch Sources:")
    print(results_df['Match_Source'].value_counts())
    
    print("\nBy Resource Type:")
    for rtype in results_df['Resource_Type'].unique():
        type_df = results_df[results_df['Resource_Type'] == rtype]
        coords = type_df['Latitude'].notna().sum()
        print(f"  {rtype}: {len(type_df)} resources, {coords} with coords ({coords/len(type_df)*100:.1f}%)")
    
    # Check for over-matching
    print("\nQuality Check - Plants matched to many resources:")
    if has_coords > 0:
        plant_counts = results_df[results_df['Plant_Name'].notna()]['Plant_Name'].value_counts()
        over_matched = plant_counts[plant_counts > 20]
        if len(over_matched) > 0:
            print("  WARNING: Some plants may be over-matched:")
            for plant, count in over_matched.head(5).items():
                print(f"    {plant}: {count} resources")
        else:
            print("  Good - no plants are matched to excessive resources")
    
    # 9. Save comprehensive results
    output_file = 'ERCOT_ALL_GENERATORS_MAPPING_FINAL.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\nFull results saved to: {output_file}")
    
    # 10. Create simple CSV with expanded columns
    simple_df = results_df[results_df['Latitude'].notna()].copy()
    
    # Add unit name
    simple_df['Unit_Name'] = simple_df['Resource_Name'].map(
        lambda x: resource_to_substation.get(x, {}).get('Unit_Name', '') if x in resource_to_substation else ''
    )
    
    # Select and rename columns
    simple_df = simple_df[['Resource_Name', 'Plant_Name', 'Substation', 'Unit_Name', 'Latitude', 'Longitude']]
    simple_df.columns = ['resource_node', 'plant_name', 'substation', 'unit_name', 'latitude', 'longitude']
    
    simple_file = 'ERCOT_GENERATORS_LOCATIONS_SIMPLE_FINAL.csv'
    simple_df.to_csv(simple_file, index=False)
    print(f"Simple location file saved to: {simple_file}")
    print(f"  Contains {len(simple_df)} resources with coordinates")
    
    # 11. Sample output
    print("\n" + "=" * 80)
    print("SAMPLE RESULTS")
    print("=" * 80)
    
    # Show high-confidence matches
    high_conf = results_df[
        (results_df['Match_Source'] == 'High_Confidence_Mapping') & 
        (results_df['Latitude'].notna())
    ].head(10)
    
    if len(high_conf) > 0:
        print("\nHigh Confidence Matches:")
        for _, row in high_conf.iterrows():
            print(f"  {row['Resource_Name'][:25]:25} -> {row['Plant_Name'][:35]:35} ({row['Latitude']:.4f}, {row['Longitude']:.4f})")
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
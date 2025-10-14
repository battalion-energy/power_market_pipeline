#!/usr/bin/env python3
"""
Complete Generator Mapping Pipeline V2 for ALL ERCOT Generators
Maps all generator types to locations using settlement point and electrical bus mapping
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

# Load zone to county mapping
LOAD_ZONE_TO_COUNTIES = {
    'LZ_HOUSTON': ['Harris', 'Fort Bend', 'Montgomery', 'Galveston', 'Brazoria', 
                   'Liberty', 'Chambers', 'Waller'],
    'LZ_NORTH': ['Dallas', 'Tarrant', 'Collin', 'Denton', 'Rockwall', 
                 'Ellis', 'Johnson', 'Parker', 'Hood', 'Wise'],
    'LZ_SOUTH': ['Bexar', 'Nueces', 'San Patricio', 'Bee', 'Jim Wells',
                 'Live Oak', 'Kleberg', 'Aransas'],
    'LZ_WEST': ['Andrews', 'Martin', 'Howard', 'Midland', 'Ector', 'Crane',
                'Upton', 'Reagan', 'Irion', 'Pecos', 'Reeves', 'Ward'],
    'LZ_CPS': ['Bexar'],
    'LZ_LCRA': ['Travis', 'Williamson', 'Bastrop', 'Llano', 'Burnet', 'Blanco'],
    'LZ_RAYBN': ['Lamar', 'Red River', 'Franklin', 'Titus', 'Morris'],
    'LZ_AEN': ['Cameron', 'Hidalgo', 'Willacy', 'Starr']
}

def clean_resource_name(name: str) -> str:
    """Clean and standardize resource names"""
    if pd.isna(name):
        return ''
    name = str(name).strip().upper()
    # Remove common suffixes
    name = re.sub(r'[_-](UNIT|GEN|RN|ALL|G\d+|GT\d+|CC\d+|ST\d+|CT\d+|PV\d+|WD\d+)\d*$', '', name)
    return name

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
    if 'NUC' in name or 'STP' in name:  # STP = South Texas Project
        return 'Nuclear'
    
    # Coal
    if 'COAL' in name or '_LIG' in name:
        return 'Coal'
    
    # Hydro
    if 'HYDRO' in name or '_HY' in name:
        return 'Hydro'
    
    return 'Other'

def fuzzy_match_score(name1: str, name2: str) -> float:
    """Calculate fuzzy match score between two names"""
    name1 = clean_resource_name(name1)
    name2 = clean_resource_name(name2)
    
    if not name1 or not name2:
        return 0.0
    
    # Exact match
    if name1 == name2:
        return 1.0
    
    # One contains the other
    if name1 in name2 or name2 in name1:
        return 0.8
    
    # Common tokens
    tokens1 = set(name1.replace('_', ' ').split())
    tokens2 = set(name2.replace('_', ' ').split())
    
    if tokens1 and tokens2:
        overlap = len(tokens1 & tokens2)
        total = len(tokens1 | tokens2)
        if total > 0:
            return overlap / total
    
    return 0.0

def main():
    print("=" * 80)
    print("COMPLETE GENERATOR MAPPING PIPELINE V2")
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
    
    # 4. Load interconnection queue data
    print("\n4. Loading interconnection queue data...")
    iq_df = pd.read_excel(INTERCONNECTION_FILE, sheet_name='Project Details - Large Gen', skiprows=30)
    iq_df.columns = iq_df.columns.str.strip()
    
    # Filter to operational/construction projects
    iq_df = iq_df[iq_df['INR'].notna()].copy()
    print(f"   Loaded {len(iq_df)} interconnection projects")
    
    # Clean IQ data
    iq_df['Clean_Name'] = iq_df['Project Name'].apply(clean_resource_name)
    iq_df['County_Clean'] = iq_df['County'].str.strip().str.upper() if 'County' in iq_df.columns else None
    
    # 5. Load EIA plant data
    print("\n5. Loading EIA plant location data...")
    eia_df = pd.read_csv(EIA_PLANT_FILE)
    
    # Filter to Texas plants
    eia_tx = eia_df[eia_df['state'] == 'TX'].copy()
    print(f"   Found {len(eia_tx)} Texas generators in EIA data")
    print(f"   With coordinates: {eia_tx['latitude'].notna().sum()}")
    
    # Group by plant for location data
    eia_plants = eia_tx.groupby('plant_code').agg({
        'plant_name': 'first',
        'latitude': 'first',
        'longitude': 'first',
        'nameplate_capacity_mw': 'sum',
        'fuel_type': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0],
        'technology': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
    }).reset_index()
    
    eia_plants['Clean_Name'] = eia_plants['plant_name'].apply(clean_resource_name)
    
    # 6. Load EIA-ERCOT mapping
    print("\n6. Loading EIA-ERCOT mapping...")
    mapping_df = pd.read_csv(EIA_ERCOT_MAPPING)
    print(f"   Found {len(mapping_df)} existing mappings")
    
    # 7. Match resources to locations
    print("\n7. Matching resources to locations using all data sources...")
    
    results = []
    
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
                
                # Get potential counties from load zone
                if result['Load_Zone'] in LOAD_ZONE_TO_COUNTIES:
                    potential_counties = LOAD_ZONE_TO_COUNTIES[result['Load_Zone']]
                    # We'll use this for matching later
        
        # Find settlement point from SP mapping
        sp_matches = sp_df[sp_df['RESOURCE_NODE'] == resource['Resource_Name']]
        if len(sp_matches) > 0:
            result['Settlement_Point'] = sp_matches.iloc[0]['NODE_NAME']
            if pd.isna(result['Load_Zone']):
                result['Load_Zone'] = sp_matches.iloc[0]['SETTLEMENT_LOAD_ZONE']
        
        # Try direct mapping first
        direct_map = mapping_df[mapping_df['ercot_node'] == resource['Resource_Name']]
        if len(direct_map) > 0:
            # Get EIA plant data
            eia_plant_id = direct_map.iloc[0]['eia_plant_id']
            eia_match = eia_plants[eia_plants['plant_code'] == eia_plant_id]
            
            if len(eia_match) > 0:
                result['Plant_Name'] = eia_match.iloc[0]['plant_name']
                result['Latitude'] = eia_match.iloc[0]['latitude']
                result['Longitude'] = eia_match.iloc[0]['longitude']
                result['Capacity_MW'] = eia_match.iloc[0]['nameplate_capacity_mw']
                result['Match_Source'] = 'Direct_Mapping'
                result['Match_Confidence'] = direct_map.iloc[0]['confidence'] if 'confidence' in direct_map.columns else 0.9
        
        # Try matching by substation name if no direct match
        if result['Latitude'] is None and result['Substation']:
            # Try to match substation name with plant names
            best_score = 0
            best_match = None
            
            for _, eia_plant in eia_plants.iterrows():
                score = fuzzy_match_score(result['Substation'], eia_plant['plant_name'])
                if score > best_score and score > 0.5:
                    best_score = score
                    best_match = eia_plant
            
            if best_match is not None:
                result['Plant_Name'] = best_match['plant_name']
                result['Latitude'] = best_match['latitude']
                result['Longitude'] = best_match['longitude']
                result['Capacity_MW'] = best_match['nameplate_capacity_mw']
                result['Match_Source'] = 'Substation_Match'
                result['Match_Confidence'] = best_score * 0.85
        
        # Try fuzzy matching with base name if still no match
        if result['Latitude'] is None and resource['Base_Name']:
            best_score = 0
            best_match = None
            
            for _, eia_plant in eia_plants.iterrows():
                score = fuzzy_match_score(resource['Base_Name'], eia_plant['plant_name'])
                if score > best_score and score > 0.6:
                    best_score = score
                    best_match = eia_plant
            
            if best_match is not None:
                result['Plant_Name'] = best_match['plant_name']
                result['Latitude'] = best_match['latitude']
                result['Longitude'] = best_match['longitude']
                result['Capacity_MW'] = best_match['nameplate_capacity_mw']
                result['Match_Source'] = 'Fuzzy_Match'
                result['Match_Confidence'] = best_score * 0.8
        
        # Try interconnection queue matching
        if result['County'] is None and resource['Base_Name']:
            iq_matches = iq_df[iq_df['Clean_Name'].str.contains(resource['Base_Name'], na=False)]
            if len(iq_matches) > 0:
                match = iq_matches.iloc[0]
                result['County'] = match['County'] if 'County' in match else None
                if pd.isna(result['Plant_Name']):
                    result['Plant_Name'] = match['Project Name']
                if pd.isna(result['Capacity_MW']):
                    result['Capacity_MW'] = match['Capacity (MW)'] if 'Capacity (MW)' in match else None
                if result['Match_Source'] is None:
                    result['Match_Source'] = 'Interconnection_Queue'
                    result['Match_Confidence'] = 0.6
        
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
    has_settlement = results_df['Settlement_Point'].notna().sum()
    
    print(f"\nTotal Resources: {total}")
    print(f"With Substation: {has_substation} ({has_substation/total*100:.1f}%)")
    print(f"With Load Zone: {has_load_zone} ({has_load_zone/total*100:.1f}%)")
    print(f"With Settlement Point: {has_settlement} ({has_settlement/total*100:.1f}%)")
    print(f"With Coordinates: {has_coords} ({has_coords/total*100:.1f}%)")
    print(f"With County: {has_county} ({has_county/total*100:.1f}%)")
    print(f"With Capacity: {has_capacity} ({has_capacity/total*100:.1f}%)")
    
    print("\nMatch Sources:")
    print(results_df['Match_Source'].value_counts())
    
    print("\nBy Resource Type:")
    for rtype in results_df['Resource_Type'].unique():
        type_df = results_df[results_df['Resource_Type'] == rtype]
        coords = type_df['Latitude'].notna().sum()
        substations = type_df['Substation'].notna().sum()
        print(f"  {rtype}: {len(type_df)} resources, {substations} with substation, {coords} with coords")
    
    print("\nLoad Zone Distribution:")
    print(results_df['Load_Zone'].value_counts().head(10))
    
    # 9. Save comprehensive results
    output_file = 'ERCOT_ALL_GENERATORS_MAPPING_V2.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\nFull results saved to: {output_file}")
    
    # 10. Create simple CSV with expanded columns
    simple_df = results_df[results_df['Latitude'].notna()].copy()
    
    # Add unit name from gen_map_df
    simple_df['Unit_Name'] = simple_df['Resource_Name'].map(
        lambda x: resource_to_substation.get(x, {}).get('Unit_Name', '') if x in resource_to_substation else ''
    )
    
    # Select and rename columns
    simple_df = simple_df[['Resource_Name', 'Plant_Name', 'Substation', 'Unit_Name', 'Latitude', 'Longitude']]
    simple_df.columns = ['resource_node', 'plant_name', 'substation', 'unit_name', 'latitude', 'longitude']
    
    simple_file = 'ERCOT_GENERATORS_LOCATIONS_SIMPLE_V2.csv'
    simple_df.to_csv(simple_file, index=False)
    print(f"Simple location file saved to: {simple_file}")
    print(f"  Contains {len(simple_df)} resources with coordinates")
    
    # 11. Create substation mapping file
    substation_df = results_df[results_df['Substation'].notna()][
        ['Resource_Name', 'Substation', 'Settlement_Point', 'Load_Zone', 'Resource_Type']
    ].copy()
    
    substation_file = 'ERCOT_GENERATORS_SUBSTATION_MAPPING.csv'
    substation_df.to_csv(substation_file, index=False)
    print(f"Substation mapping saved to: {substation_file}")
    print(f"  Contains {len(substation_df)} resources with substation data")
    
    # 12. Sample output
    print("\n" + "=" * 80)
    print("SAMPLE RESULTS WITH FULL MAPPING")
    print("=" * 80)
    
    # Show samples with complete mapping
    complete_samples = results_df[
        (results_df['Substation'].notna()) & 
        (results_df['Latitude'].notna()) &
        (results_df['Load_Zone'].notna())
    ].head(10)
    
    if len(complete_samples) > 0:
        print("\nResources with Complete Mapping:")
        for _, row in complete_samples.iterrows():
            print(f"\n  Resource: {row['Resource_Name']}")
            print(f"    Type: {row['Resource_Type']}")
            print(f"    Substation: {row['Substation']}")
            print(f"    Load Zone: {row['Load_Zone']}")
            print(f"    Plant: {row['Plant_Name']}")
            print(f"    Location: ({row['Latitude']:.4f}, {row['Longitude']:.4f})")
    
    # Show samples by type
    print("\n" + "-" * 40)
    for rtype in ['Gas', 'Wind', 'Solar', 'Battery']:
        type_sample = results_df[
            (results_df['Resource_Type'] == rtype) & 
            (results_df['Substation'].notna())
        ].head(2)
        if len(type_sample) > 0:
            print(f"\n{rtype} Resources with Substation:")
            for _, row in type_sample.iterrows():
                coords_str = f"({row['Latitude']:.4f}, {row['Longitude']:.4f})" if pd.notna(row['Latitude']) else "No coords"
                print(f"  {row['Resource_Name'][:25]:25} -> {row['Substation'][:15]:15} in {row['Load_Zone']:12} {coords_str}")
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
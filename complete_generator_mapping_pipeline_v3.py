#!/usr/bin/env python3
"""
Complete Generator Mapping Pipeline V3 for ALL ERCOT Generators
Improved matching logic with known plant mappings
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

# Known ERCOT to EIA plant name mappings
KNOWN_PLANT_MAPPINGS = {
    # Gas Plants
    'AMOCOOIL': 'Amoco Oil',
    'CHE': 'Comanche Peak',
    'FORMOSA': 'Formosa Utility Venture Ltd',
    'BRAUNIG': 'V H Braunig',
    'TOPAZ': 'Topaz Generating',
    'CAL': 'Houston Chemical Complex Battleground',
    'LEG': 'Houston Chemical Complex Battleground',
    'DANSBY': 'Dansby',
    'DAN': 'Dansby',
    'PRO': 'Fayette Power Project',
    'VICTORIA': 'Victoria',
    'RAYBURN': 'Sam Rayburn',
    'WGU': 'Newgulf Cogen',
    'WES': 'WestRock (TX)',
    'FORMOSA': 'Formosa Utility Venture Ltd',
    'THW': 'Goldthwaite Wind Energy Facility',
    'INKSDA': 'Inks',
    'TEN': 'Tenet Hospital',
    'AUSTPL': 'Austin',
    'ALVIN': 'Alvin',
    'ANGLETON': 'Angleton',
    'BRAZORIA': 'Brazoria',
    'SWEENY': 'Sweeny',
    
    # Wind Farms
    'WND': 'Whitney',
    'WHITNEY': 'Whitney',
    'AJAXWIND': 'Aviator Wind',
    'AVIATOR': 'Aviator Wind',
    'ALGODON': 'El Algodon Alto Wind Farm, LLC',
    'ANACACHO': 'Anacacho Wind Farm, LLC',
    'ANCHOR': 'Canyon',
    'CANYONWD': 'Canyon',
    'RDCANYON': 'Canyon',
    'BRISCOE': 'Briscoe Wind Farm',
    'CHALUPA': 'La Chalupa, LLC',
    'CHAMPION': 'Champion Wind Farm LLC',
    'COYOTE_W': 'Coyote Wind LLC',
    'CRANELL': 'Cranell Wind Farm LLC',
    'DERMOTT': 'Dermott Wind',
    'ELB': 'Elbow Creek Wind Project LLC',
    'FLUVANNA': 'Fluvanna',
    'GOPHER': 'Gopher Creek Wind Farm',
    'HOL': 'Nichols',
    'KEECHI': 'Keechi Wind',
    'LOCKETT': 'Lockett Windfarm',
    'MARYNEAL': 'Maryneal Windpower',
    'PENA': 'Penascal Wind Power LLC',
    'PEY': 'Peyton Creek Wind Farm LLC',
    'PRIDDY': 'Priddy Wind Project',
    'PYR': 'Pyron Wind Farm LLC Hybrid',
    'RANCHERO': 'Ranchero Wind Farm LLC',
    'RELOJ': 'Reloj del Sol Wind Farm',
    'STELLA': 'Stella Wind Farm',
    'TAHOKA': 'Tahoka Wind',
    'TORR': 'Torrecillas Wind Energy, LLC',
    'TRENT': 'Trent Wind Farm LP',
    'BAFFIN': 'Baffin Wind',
    'ASTRA': 'Astra Wind Farm',
    
    # Solar Farms
    'ARAGORN': 'Aragorn Solar Project',
    'AZURE': 'Azure Sky Solar',
    'CONIGLIO': 'Coniglio Solar',
    'IMPACT': 'Impact Solar 1',
    'JAY': 'Fighting Jays Solar Project',
    'JUNO': 'Juno Solar Project',
    'LAPETUS': 'Lapetus',
    'LILY': 'Lily Solar Hybrid',
    'MISAE': 'Misae Solar',
    'NEBULA': 'Nebula Solar',
    'PHOEBE': 'Phoebe Solar',
    'PROSPERO': 'Prospero Solar',
    'SAMSON': 'Samson Solar Energy',
    'VANCOURT': 'Vancourt Solar Interconnections',
    
    # Battery Storage
    'CROSSETT': 'Crossett Power Management LLC',
    'FAULKNER': 'Faulkner',
    'FLAT_TOP': 'TX7 Flat Top',
    'GAMBIT': 'Gambit Energy Storage - Angleton Storage',
    'LONESTAR': 'Lonestar',
    'RAB': 'Rabbit Hill Energy Storage Project',
    'RAMBLER': 'Rambler',
    'TOYAH': 'Toyah Power Station',
    'WORSHAM': 'TX8 Worsham',
    
    # Other
    'CORAZON': 'Corazon Energy LLC',
    'COTTON': 'Cottonwood Energy Project',
    'LOPENO': 'Lopeno',
    'MESTENO': 'Mesteno',
    'ROW': 'Rio Grande Valley Sugar Growers',
    'SBE': 'PowerFin Kingsbery',
    'TAYGETE': 'Taygete Energy Project LLC',
}

# Load zone to county mapping
LOAD_ZONE_TO_COUNTIES = {
    'LZ_HOUSTON': ['Harris', 'Fort Bend', 'Montgomery', 'Galveston', 'Brazoria'],
    'LZ_NORTH': ['Dallas', 'Tarrant', 'Collin', 'Denton', 'Rockwall'],
    'LZ_SOUTH': ['Bexar', 'Nueces', 'San Patricio', 'Bee', 'Jim Wells'],
    'LZ_WEST': ['Andrews', 'Martin', 'Howard', 'Midland', 'Ector', 'Crane'],
    'LZ_CPS': ['Bexar'],
    'LZ_LCRA': ['Travis', 'Williamson', 'Bastrop', 'Llano', 'Burnet'],
    'LZ_RAYBN': ['Lamar', 'Red River', 'Franklin', 'Titus', 'Morris'],
    'LZ_AEN': ['Cameron', 'Hidalgo', 'Willacy', 'Starr']
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
    
    # Handle special cases
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

def fuzzy_match_score(name1: str, name2: str) -> float:
    """Calculate fuzzy match score between two names"""
    if pd.isna(name1) or pd.isna(name2):
        return 0.0
    
    name1 = str(name1).upper().replace('_', ' ').replace('-', ' ')
    name2 = str(name2).upper().replace('_', ' ').replace('-', ' ')
    
    # Exact match
    if name1 == name2:
        return 1.0
    
    # One contains the other
    if name1 in name2 or name2 in name1:
        return 0.8
    
    # Token-based matching
    tokens1 = set(name1.split())
    tokens2 = set(name2.split())
    
    if tokens1 and tokens2:
        overlap = len(tokens1 & tokens2)
        min_len = min(len(tokens1), len(tokens2))
        if min_len > 0:
            return overlap / min_len
    
    return 0.0

def main():
    print("=" * 80)
    print("COMPLETE GENERATOR MAPPING PIPELINE V3")
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
    
    # 4. Load EIA plant data
    print("\n4. Loading EIA plant location data...")
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
    
    # Create name lookup dictionary for EIA plants
    eia_name_lookup = {}
    for _, plant in eia_plants.iterrows():
        if pd.notna(plant['plant_name']):
            # Add original name
            eia_name_lookup[plant['plant_name'].upper()] = plant
            
            # Add simplified names
            simple_name = plant['plant_name'].upper().replace(' ', '').replace('-', '')
            eia_name_lookup[simple_name] = plant
            
            # Add partial names
            parts = plant['plant_name'].upper().split()
            if len(parts) > 0:
                eia_name_lookup[parts[0]] = plant
    
    # 5. Load interconnection queue data
    print("\n5. Loading interconnection queue data...")
    iq_df = pd.read_excel(INTERCONNECTION_FILE, sheet_name='Project Details - Large Gen', skiprows=30)
    iq_df.columns = iq_df.columns.str.strip()
    
    # Filter to operational/construction projects
    iq_df = iq_df[iq_df['INR'].notna()].copy()
    print(f"   Loaded {len(iq_df)} interconnection projects")
    
    # 6. Load EIA-ERCOT mapping
    print("\n6. Loading EIA-ERCOT mapping...")
    mapping_df = pd.read_csv(EIA_ERCOT_MAPPING)
    print(f"   Found {len(mapping_df)} existing mappings")
    
    # 7. Match resources to locations
    print("\n7. Matching resources to locations using all data sources...")
    
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
        
        # Try known mappings first
        base_name = resource['Base_Name']
        if base_name in KNOWN_PLANT_MAPPINGS:
            known_plant_name = KNOWN_PLANT_MAPPINGS[base_name]
            
            # Find in EIA data
            eia_matches = eia_plants[eia_plants['plant_name'].str.contains(known_plant_name, case=False, na=False)]
            if len(eia_matches) > 0:
                match = eia_matches.iloc[0]
                result['Plant_Name'] = match['plant_name']
                result['Latitude'] = match['latitude']
                result['Longitude'] = match['longitude']
                result['Capacity_MW'] = match['nameplate_capacity_mw']
                result['Match_Source'] = 'Known_Mapping'
                result['Match_Confidence'] = 0.95
                matched_count += 1
        
        # Try direct mapping if no known mapping
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
                    result['Match_Source'] = 'Direct_Mapping'
                    result['Match_Confidence'] = direct_map.iloc[0]['confidence'] if 'confidence' in direct_map.columns else 0.9
                    matched_count += 1
        
        # Try fuzzy matching
        if result['Latitude'] is None and base_name:
            best_score = 0
            best_match = None
            
            # Check various name variations
            names_to_check = [base_name]
            
            # Add substation name if available
            if result['Substation']:
                names_to_check.append(result['Substation'])
            
            for check_name in names_to_check:
                for plant_name, plant_data in eia_name_lookup.items():
                    score = fuzzy_match_score(check_name, plant_name)
                    if score > best_score and score > 0.5:
                        best_score = score
                        best_match = plant_data
            
            if best_match is not None:
                result['Plant_Name'] = best_match['plant_name']
                result['Latitude'] = best_match['latitude']
                result['Longitude'] = best_match['longitude']
                result['Capacity_MW'] = best_match['nameplate_capacity_mw']
                result['Match_Source'] = 'Fuzzy_Match'
                result['Match_Confidence'] = best_score * 0.8
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
    
    # 9. Save comprehensive results
    output_file = 'ERCOT_ALL_GENERATORS_MAPPING_V3.csv'
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
    
    simple_file = 'ERCOT_GENERATORS_LOCATIONS_SIMPLE_V3.csv'
    simple_df.to_csv(simple_file, index=False)
    print(f"Simple location file saved to: {simple_file}")
    print(f"  Contains {len(simple_df)} resources with coordinates")
    
    # 11. Sample output
    print("\n" + "=" * 80)
    print("SAMPLE RESULTS BY TYPE")
    print("=" * 80)
    
    for rtype in ['Gas', 'Wind', 'Solar', 'Battery']:
        type_sample = results_df[
            (results_df['Resource_Type'] == rtype) & 
            (results_df['Latitude'].notna())
        ].head(5)
        if len(type_sample) > 0:
            print(f"\n{rtype} Resources with Locations:")
            for _, row in type_sample.iterrows():
                print(f"  {row['Resource_Name'][:25]:25} -> {row['Plant_Name'][:30] if row['Plant_Name'] else 'Unknown':30} "
                      f"({row['Latitude']:.4f}, {row['Longitude']:.4f}) [{row['Match_Source']}]")
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
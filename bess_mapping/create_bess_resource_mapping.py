#!/usr/bin/env python3
"""
Create definitive BESS resource mapping table for ERCOT
Maps Generation resources to Load resources with settlement points and substations

Verified patterns:
1. Same name: BESS_NAME -> BESS_NAME (same)
2. BES pattern: XXX_BES1 -> XXX_LD1
3. BESS pattern: XXX_BESS1 -> XXX_LD1  
4. UNIT pattern: XXX_UNIT1 -> XXX_LD1
5. Multiple units: XXX_BESS1/2/3/4 -> XXX_LD1/2/3/4
"""

import pandas as pd
import re
from pathlib import Path

def get_load_resource_name(gen_name: str) -> str:
    """
    Convert BESS Generation resource name to Load resource name
    using verified patterns
    
    Args:
        gen_name: Generation resource name (e.g., BATCAVE_BES1)
        
    Returns:
        Load resource name (e.g., BATCAVE_LD1)
    """
    # Pattern 1: If it doesn't have _BES, _BESS, or _UNIT, it's the same
    if not any(pattern in gen_name for pattern in ['_BES', '_BESS', '_UNIT']):
        return gen_name
    
    # Pattern 2-5: Replace patterns with _LD
    load_name = gen_name
    
    # Handle _BES{N} -> _LD{N}
    load_name = re.sub(r'_BES(\d+)', r'_LD\1', load_name)
    
    # Handle _BESS{N} -> _LD{N}
    load_name = re.sub(r'_BESS(\d+)', r'_LD\1', load_name)
    
    # Handle _UNIT{N} -> _LD{N}
    load_name = re.sub(r'_UNIT(\d+)', r'_LD\1', load_name)
    
    # Handle UNIT{N} without underscore -> LD{N}
    load_name = re.sub(r'UNIT(\d+)$', r'LD\1', load_name)
    
    return load_name


def create_bess_mapping_table():
    """
    Create comprehensive BESS mapping table with all resource relationships
    """
    print('=== Creating Definitive BESS Resource Mapping Table ===\n')
    
    # Step 1: Get all BESS from DAM Gen Resources
    dam_file = '/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/rollup_files/DAM_Gen_Resources/2024.parquet'
    dam_df = pd.read_parquet(dam_file, columns=['ResourceName', 'ResourceType', 'SettlementPointName'])
    
    # Filter for BESS (PWRSTR = Power Storage)
    bess_gen = dam_df[dam_df['ResourceType'] == 'PWRSTR'][['ResourceName', 'SettlementPointName']].drop_duplicates()
    print(f'Found {len(bess_gen)} unique BESS Generation resources')
    
    # Step 2: Load settlement point and substation mappings
    base_path = Path('/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/Settlement_Points_List_and_Electrical_Buses_Mapping/latest_mapping/SP_List_EB_Mapping')
    
    resource_to_unit = pd.read_csv(base_path / 'Resource_Node_to_Unit_07242024_210751.csv')
    settlement_points = pd.read_csv(base_path / 'Settlement_Points_07242024_210751.csv')
    
    # Create mapping dictionaries
    resource_to_substation = dict(zip(resource_to_unit['RESOURCE_NODE'], resource_to_unit['UNIT_SUBSTATION']))
    
    # Also get substation from settlement points
    sp_to_substation = {}
    sp_to_zone = {}
    for _, row in settlement_points.iterrows():
        if pd.notna(row['RESOURCE_NODE']):
            if pd.notna(row['SUBSTATION']):
                sp_to_substation[row['RESOURCE_NODE']] = row['SUBSTATION']
            if pd.notna(row['SETTLEMENT_LOAD_ZONE']):
                sp_to_zone[row['RESOURCE_NODE']] = row['SETTLEMENT_LOAD_ZONE']
    
    print(f'Loaded {len(resource_to_substation)} resource->substation mappings')
    print(f'Loaded {len(sp_to_zone)} settlement point->zone mappings')
    
    # Step 3: Create mapping table
    bess_mapping = []
    
    for _, row in bess_gen.iterrows():
        gen_name = row['ResourceName']
        settlement_point = row['SettlementPointName']
        
        # Get Load resource name using verified patterns
        load_name = get_load_resource_name(gen_name)
        
        # Get substation
        substation = 'Unknown'
        if settlement_point in resource_to_substation:
            substation = resource_to_substation[settlement_point]
        elif settlement_point in sp_to_substation:
            substation = sp_to_substation[settlement_point]
        else:
            # Try extracting from resource name
            if '_' in gen_name:
                potential_substation = gen_name.split('_')[0]
                if potential_substation in resource_to_unit['UNIT_SUBSTATION'].values:
                    substation = potential_substation
        
        # Get load zone
        load_zone = 'Unknown'
        if settlement_point in sp_to_zone:
            load_zone = sp_to_zone[settlement_point]
        elif settlement_point and settlement_point.startswith('LZ_'):
            load_zone = settlement_point
        elif settlement_point and settlement_point.startswith('HB_'):
            load_zone = 'HB_BUSAVG'
        
        bess_mapping.append({
            'BESS_Gen_Resource': gen_name,
            'BESS_Load_Resource': load_name,
            'Settlement_Point': settlement_point if settlement_point else 'Unknown',
            'Substation': substation,
            'Load_Zone': load_zone
        })
    
    # Convert to DataFrame and sort
    bess_df = pd.DataFrame(bess_mapping)
    bess_df = bess_df.sort_values('BESS_Gen_Resource')
    
    # Save to CSV
    output_file = Path('/home/enrico/projects/power_market_pipeline/BESS_RESOURCE_MAPPING.csv')
    bess_df.to_csv(output_file, index=False)
    
    print(f'\n✅ Created definitive mapping table with {len(bess_df)} BESS resources')
    print(f'💾 Saved to: {output_file}')
    
    # Show statistics
    print('\n=== Mapping Statistics ===')
    print(f'Total BESS Resources: {len(bess_df)}')
    print(f'With Substation: {(bess_df["Substation"] != "Unknown").sum()} ({100*(bess_df["Substation"] != "Unknown").sum()/len(bess_df):.1f}%)')
    print(f'With Load Zone: {(bess_df["Load_Zone"] != "Unknown").sum()} ({100*(bess_df["Load_Zone"] != "Unknown").sum()/len(bess_df):.1f}%)')
    
    print('\n=== Load Zone Distribution ===')
    print(bess_df['Load_Zone'].value_counts().to_string())
    
    print('\n=== Sample Mappings ===')
    sample = bess_df.head(20)
    print(sample[['BESS_Gen_Resource', 'BESS_Load_Resource', 'Substation', 'Load_Zone']].to_string(index=False))
    
    # Show examples of each pattern
    print('\n=== Pattern Examples ===')
    patterns = [
        ('BATCAVE_BES1', 'BES -> LD pattern'),
        ('ANCHOR_BESS1', 'BESS -> LD pattern'),
        ('ALVIN_UNIT1', 'UNIT -> LD pattern'),
        ('BAY_CITY_BESS', 'Same name pattern'),
        ('DKNS_ESS_BESS1', 'Multiple pattern (_BESS1)')
    ]
    
    for gen_name, pattern_desc in patterns:
        if gen_name in bess_df['BESS_Gen_Resource'].values:
            row = bess_df[bess_df['BESS_Gen_Resource'] == gen_name].iloc[0]
            print(f'{pattern_desc:20} {gen_name:20} -> {row["BESS_Load_Resource"]:20}')
    
    return bess_df


if __name__ == "__main__":
    # Create the mapping table
    mapping_df = create_bess_mapping_table()
    
    print('\n' + '='*60)
    print('✅ BESS Resource Mapping Complete!')
    print('='*60)
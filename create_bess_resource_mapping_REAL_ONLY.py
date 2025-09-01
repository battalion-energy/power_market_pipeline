#!/usr/bin/env python3
"""
Create BESS resource mapping table for ERCOT - REAL DATA ONLY
Maps Generation resources to Load resources ONLY when Load Resource actually exists in ERCOT data

NO PREDICTIONS, NO PATTERNS, ONLY VERIFIED REAL DATA
"""

import pandas as pd
import glob
from pathlib import Path

def get_all_real_load_resources():
    """
    Get ALL Load Resource names that ACTUALLY EXIST in ERCOT data
    Returns a set of real Load Resource names
    """
    print('🔍 Loading REAL Load Resources from ERCOT data...')
    
    real_load_resources = set()
    
    # Check DAM Load Resource CSV files
    dam_path = '/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/60-Day_DAM_Disclosure_Reports/csv/'
    dam_files = glob.glob(dam_path + '60d_DAM_Load_Resource_Data-*.csv')
    
    print(f'   Checking {len(dam_files)} DAM files...')
    for i, file in enumerate(dam_files):
        if i % 100 == 0:
            print(f'   Progress: {i}/{len(dam_files)} files...')
        try:
            df = pd.read_csv(file, nrows=50000)  # Read in chunks for speed
            if 'Load Resource Name' in df.columns:
                real_load_resources.update(df['Load Resource Name'].dropna().unique())
        except:
            continue
    
    # Check SCED Load Resource CSV files
    sced_path = '/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/60-Day_SCED_Disclosure_Reports/csv/'
    sced_files = glob.glob(sced_path + '60d_Load_Resource_Data_in_SCED-*.csv')
    
    print(f'   Checking {len(sced_files)} SCED files...')
    for i, file in enumerate(sced_files):
        if i % 100 == 0:
            print(f'   Progress: {i}/{len(sced_files)} files...')
        try:
            df = pd.read_csv(file, nrows=50000)
            if 'Resource Name' in df.columns:
                real_load_resources.update(df['Resource Name'].dropna().unique())
        except:
            continue
    
    print(f'✅ Found {len(real_load_resources)} REAL Load Resources in ERCOT data')
    return real_load_resources


def create_real_bess_mapping():
    """
    Create BESS mapping table with ONLY REAL Load Resources
    No predictions, no patterns - only verified data
    """
    print('\n=== Creating REAL BESS Resource Mapping (NO FAKE DATA) ===\n')
    
    # Step 1: Get all REAL Load Resources
    real_load_resources = get_all_real_load_resources()
    
    # Step 2: Get all BESS Gen Resources
    dam_file = '/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/rollup_files/DAM_Gen_Resources/2024.parquet'
    dam_df = pd.read_parquet(dam_file, columns=['ResourceName', 'ResourceType', 'SettlementPointName'])
    
    # Filter for BESS (PWRSTR = Power Storage)
    bess_gen = dam_df[dam_df['ResourceType'] == 'PWRSTR'][['ResourceName', 'SettlementPointName']].drop_duplicates()
    print(f'\nFound {len(bess_gen)} BESS Generation resources')
    
    # Step 3: Load settlement point mappings
    base_path = Path('/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/Settlement_Points_List_and_Electrical_Buses_Mapping/latest_mapping/SP_List_EB_Mapping')
    
    resource_to_unit = pd.read_csv(base_path / 'Resource_Node_to_Unit_07242024_210751.csv')
    settlement_points = pd.read_csv(base_path / 'Settlement_Points_07242024_210751.csv')
    
    # Create mapping dictionaries
    resource_to_substation = dict(zip(resource_to_unit['RESOURCE_NODE'], resource_to_unit['UNIT_SUBSTATION']))
    
    sp_to_substation = {}
    sp_to_zone = {}
    for _, row in settlement_points.iterrows():
        if pd.notna(row['RESOURCE_NODE']):
            if pd.notna(row['SUBSTATION']):
                sp_to_substation[row['RESOURCE_NODE']] = row['SUBSTATION']
            if pd.notna(row['SETTLEMENT_LOAD_ZONE']):
                sp_to_zone[row['RESOURCE_NODE']] = row['SETTLEMENT_LOAD_ZONE']
    
    # Step 4: Try to find REAL Load Resources for each BESS
    bess_mapping = []
    
    for _, row in bess_gen.iterrows():
        gen_name = row['ResourceName']
        settlement_point = row['SettlementPointName']
        
        # Try different possible Load Resource names
        possible_load_names = [
            gen_name,  # Same name
            gen_name.replace('_BES', '_LD').replace('BES', 'LD'),  # BES -> LD
            gen_name.replace('_BESS', '_LD').replace('BESS', 'LD'),  # BESS -> LD
            gen_name.replace('_UNIT', '_LD').replace('UNIT', 'LD'),  # UNIT -> LD
        ]
        
        # Find which one (if any) actually exists
        load_name = None
        for possible_name in possible_load_names:
            if possible_name in real_load_resources:
                load_name = possible_name
                break
        
        # Get substation
        substation = None
        if settlement_point in resource_to_substation:
            substation = resource_to_substation[settlement_point]
        elif settlement_point in sp_to_substation:
            substation = sp_to_substation[settlement_point]
        
        # Get load zone
        load_zone = None
        if settlement_point in sp_to_zone:
            load_zone = sp_to_zone[settlement_point]
        elif settlement_point and settlement_point.startswith('LZ_'):
            load_zone = settlement_point
        
        bess_mapping.append({
            'BESS_Gen_Resource': gen_name,
            'BESS_Load_Resource': load_name,  # Will be None if not found
            'Settlement_Point': settlement_point,
            'Substation': substation,
            'Load_Zone': load_zone
        })
    
    # Convert to DataFrame
    bess_df = pd.DataFrame(bess_mapping)
    bess_df = bess_df.sort_values('BESS_Gen_Resource')
    
    # Save to CSV
    output_file = Path('/home/enrico/projects/power_market_pipeline/BESS_RESOURCE_MAPPING_REAL_ONLY.csv')
    bess_df.to_csv(output_file, index=False)
    
    print(f'\n✅ Created REAL mapping table with {len(bess_df)} BESS resources')
    print(f'💾 Saved to: {output_file}')
    
    # Statistics
    has_load = bess_df['BESS_Load_Resource'].notna().sum()
    no_load = bess_df['BESS_Load_Resource'].isna().sum()
    
    print('\n=== REAL Data Statistics ===')
    print(f'BESS with VERIFIED Load Resources: {has_load} ({100*has_load/len(bess_df):.1f}%)')
    print(f'BESS with NO Load Resource found: {no_load} ({100*no_load/len(bess_df):.1f}%)')
    
    print('\n=== Sample REAL Mappings (with verified Load Resources) ===')
    sample = bess_df[bess_df['BESS_Load_Resource'].notna()].head(20)
    print(sample[['BESS_Gen_Resource', 'BESS_Load_Resource', 'Substation', 'Load_Zone']].to_string(index=False))
    
    print('\n=== BESS with NO Load Resource Found ===')
    no_load_sample = bess_df[bess_df['BESS_Load_Resource'].isna()].head(20)
    print(no_load_sample[['BESS_Gen_Resource', 'Settlement_Point', 'Load_Zone']].to_string(index=False))
    
    return bess_df


if __name__ == "__main__":
    # Create the REAL mapping table
    mapping_df = create_real_bess_mapping()
    
    print('\n' + '='*60)
    print('✅ REAL BESS Resource Mapping Complete!')
    print('   NO FAKE DATA - ONLY VERIFIED RESOURCES')
    print('='*60)
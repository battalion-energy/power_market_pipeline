#!/usr/bin/env python3
"""
Create a unified BESS mapping file combining all data sources:
- BESS Generation/Load Resources
- Settlement Points
- Substations
- ERCOT Interconnection Queue
- EIA Generator Data
"""

import pandas as pd
import numpy as np

def create_unified_mapping():
    """Combine all BESS mappings into a single comprehensive dataset"""
    
    print("="*70)
    print("CREATING UNIFIED BESS MAPPING")
    print("="*70)
    
    # 1. Start with base BESS resource mapping
    print("\n1️⃣ Loading base BESS resource mapping...")
    base_df = pd.read_csv('/home/enrico/projects/power_market_pipeline/BESS_RESOURCE_MAPPING_REAL_ONLY.csv')
    print(f"   Loaded {len(base_df)} BESS resources with settlement point mappings")
    
    # 2. Add Interconnection Queue matches
    print("\n2️⃣ Loading Interconnection Queue matches...")
    iq_df = pd.read_csv('/home/enrico/projects/power_market_pipeline/BESS_IMPROVED_MATCHED.csv')
    
    # Select relevant IQ columns
    iq_cols = ['BESS_Gen_Resource', 'match_score', 'Source', 'Pass', 'match_reason']
    
    # Find columns with IQ project info
    for col in iq_df.columns:
        if 'Project Name' in col or 'Unit Code' in col or 'County' in col:
            if col not in iq_cols and 'BESS' not in col:
                iq_cols.append(col)
    
    # Rename to be clear these are IQ fields
    iq_rename = {}
    for col in iq_cols:
        if col != 'BESS_Gen_Resource':
            if 'match' not in col and 'Source' not in col and 'Pass' not in col:
                iq_rename[col] = f'IQ_{col}'
            else:
                iq_rename[col] = f'IQ_{col}'
    
    iq_subset = iq_df[iq_cols].rename(columns=iq_rename)
    
    # Merge with base
    unified_df = base_df.merge(iq_subset, on='BESS_Gen_Resource', how='left')
    print(f"   Added IQ data: {len(iq_df[iq_df['match_score'] > 0])} matched resources")
    
    # 3. Add EIA Generator matches
    print("\n3️⃣ Loading EIA Generator matches...")
    eia_df = pd.read_csv('/home/enrico/projects/power_market_pipeline/BESS_EIA_MATCHED.csv')
    
    # Select relevant EIA columns
    eia_cols = [
        'BESS_Gen_Resource', 'EIA_Plant_Name', 'EIA_Generator_ID', 
        'EIA_County', 'EIA_Technology', 'EIA_Capacity_MW',
        'match_score', 'match_reason', 'Pass', 'Source'
    ]
    
    # Rename match columns to avoid conflicts
    eia_rename = {
        'match_score': 'EIA_match_score',
        'match_reason': 'EIA_match_reason',
        'Pass': 'EIA_Pass',
        'Source': 'EIA_Source'
    }
    
    eia_subset = eia_df[eia_cols].rename(columns=eia_rename)
    
    # Merge with unified
    unified_df = unified_df.merge(eia_subset, on='BESS_Gen_Resource', how='left')
    print(f"   Added EIA data: {len(eia_df[eia_df['match_score'] > 0])} matched resources")
    
    # 4. Add substation to county mapping if available
    print("\n4️⃣ Checking for substation county data...")
    try:
        sub_county_df = pd.read_csv('/home/enrico/projects/power_market_pipeline/substation_to_county_mapping.csv')
        
        # Merge to get county for substations
        unified_df = unified_df.merge(
            sub_county_df[['Substation', 'County']].rename(columns={'County': 'Substation_County'}),
            on='Substation',
            how='left'
        )
        print(f"   Added county data for {unified_df['Substation_County'].notna().sum()} substations")
    except:
        print("   Substation county mapping not found, skipping")
        unified_df['Substation_County'] = None
    
    # 5. Calculate match completeness
    print("\n5️⃣ Calculating match completeness...")
    
    unified_df['Has_Load_Resource'] = unified_df['BESS_Load_Resource'].notna()
    unified_df['Has_Settlement_Point'] = unified_df['Settlement_Point'].notna()
    unified_df['Has_Substation'] = unified_df['Substation'].notna()
    unified_df['Has_IQ_Match'] = (unified_df['IQ_match_score'] > 0).fillna(False)
    unified_df['Has_EIA_Match'] = (unified_df['EIA_match_score'] > 0).fillna(False)
    
    # Calculate completeness score (0-100%)
    unified_df['Data_Completeness'] = (
        unified_df['Has_Load_Resource'].astype(int) * 20 +
        unified_df['Has_Settlement_Point'].astype(int) * 20 +
        unified_df['Has_Substation'].astype(int) * 20 +
        unified_df['Has_IQ_Match'].astype(int) * 20 +
        unified_df['Has_EIA_Match'].astype(int) * 20
    )
    
    # Sort by completeness and name
    unified_df = unified_df.sort_values(['Data_Completeness', 'BESS_Gen_Resource'], ascending=[False, True])
    
    # Save the unified file
    output_file = '/home/enrico/projects/power_market_pipeline/BESS_UNIFIED_MAPPING.csv'
    unified_df.to_csv(output_file, index=False)
    
    print("\n" + "="*70)
    print("UNIFIED MAPPING SUMMARY")
    print("="*70)
    
    print(f"\nTotal BESS Resources: {len(unified_df)}")
    
    print("\n📊 Data Coverage:")
    print(f"  Load Resources:      {unified_df['Has_Load_Resource'].sum():3} ({100*unified_df['Has_Load_Resource'].mean():.1f}%)")
    print(f"  Settlement Points:   {unified_df['Has_Settlement_Point'].sum():3} ({100*unified_df['Has_Settlement_Point'].mean():.1f}%)")
    print(f"  Substations:         {unified_df['Has_Substation'].sum():3} ({100*unified_df['Has_Substation'].mean():.1f}%)")
    print(f"  IQ Matches:          {unified_df['Has_IQ_Match'].sum():3} ({100*unified_df['Has_IQ_Match'].mean():.1f}%)")
    print(f"  EIA Matches:         {unified_df['Has_EIA_Match'].sum():3} ({100*unified_df['Has_EIA_Match'].mean():.1f}%)")
    
    print("\n📈 Completeness Distribution:")
    completeness_bins = [0, 20, 40, 60, 80, 100]
    for i in range(len(completeness_bins)-1):
        count = len(unified_df[(unified_df['Data_Completeness'] >= completeness_bins[i]) & 
                               (unified_df['Data_Completeness'] < completeness_bins[i+1])])
        if completeness_bins[i+1] == 100:
            count = len(unified_df[unified_df['Data_Completeness'] == 100])
        print(f"  {completeness_bins[i]:3}-{completeness_bins[i+1]:3}%: {count:3} resources")
    
    # Show sample of best mapped resources
    print("\n✨ Sample Fully Mapped Resources (100% complete):")
    fully_mapped = unified_df[unified_df['Data_Completeness'] == 100]
    if len(fully_mapped) > 0:
        cols_to_show = ['BESS_Gen_Resource', 'BESS_Load_Resource', 'Settlement_Point', 
                       'Substation', 'Load_Zone', 'EIA_Plant_Name']
        available_cols = [col for col in cols_to_show if col in fully_mapped.columns]
        print(fully_mapped[available_cols].head(5).to_string(index=False))
    else:
        print("  No resources with 100% complete mapping")
    
    print(f"\n✅ Unified mapping saved to: {output_file}")
    
    # Create a summary report
    print("\n📝 Creating summary report...")
    
    summary = {
        'Metric': [],
        'Count': [],
        'Percentage': []
    }
    
    metrics = [
        ('Total BESS Resources', len(unified_df), 100.0),
        ('Has Load Resource', unified_df['Has_Load_Resource'].sum(), 100*unified_df['Has_Load_Resource'].mean()),
        ('Has Settlement Point', unified_df['Has_Settlement_Point'].sum(), 100*unified_df['Has_Settlement_Point'].mean()),
        ('Has Substation', unified_df['Has_Substation'].sum(), 100*unified_df['Has_Substation'].mean()),
        ('Has IQ Match', unified_df['Has_IQ_Match'].sum(), 100*unified_df['Has_IQ_Match'].mean()),
        ('Has EIA Match', unified_df['Has_EIA_Match'].sum(), 100*unified_df['Has_EIA_Match'].mean()),
        ('Fully Mapped (100%)', len(unified_df[unified_df['Data_Completeness'] == 100]), 
         100*len(unified_df[unified_df['Data_Completeness'] == 100])/len(unified_df)),
        ('Mostly Mapped (80%+)', len(unified_df[unified_df['Data_Completeness'] >= 80]), 
         100*len(unified_df[unified_df['Data_Completeness'] >= 80])/len(unified_df)),
    ]
    
    for metric, count, pct in metrics:
        summary['Metric'].append(metric)
        summary['Count'].append(count)
        summary['Percentage'].append(f"{pct:.1f}%")
    
    summary_df = pd.DataFrame(summary)
    summary_file = '/home/enrico/projects/power_market_pipeline/BESS_UNIFIED_SUMMARY.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"   Summary saved to: {summary_file}")
    
    return unified_df

if __name__ == '__main__':
    unified_df = create_unified_mapping()
    
    print("\n" + "="*70)
    print("UNIFIED MAPPING COMPLETE")
    print("="*70)
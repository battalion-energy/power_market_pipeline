#!/usr/bin/env python3
"""
Update unified BESS mapping with clarifications about operational status
and the misleading "Standalone" source name
"""

import pandas as pd
import numpy as np

def update_with_clarifications():
    """Add clarifying columns and notes to unified mapping"""
    
    print("="*70)
    print("UPDATING UNIFIED MAPPING WITH CLARIFICATIONS")
    print("="*70)
    
    # Load current unified mapping
    unified = pd.read_csv('/home/enrico/projects/power_market_pipeline/BESS_UNIFIED_MAPPING_V2.csv')
    
    # 1. Create clarified IQ Source column
    print("\n1️⃣ Creating clarified IQ Source descriptions...")
    
    iq_source_clarification = {
        'Operational': 'Operational (from Co-located Operational tab)',
        'Standalone': 'Planned Standalone (from Stand-Alone tab - 698/701 are PLANNED)',
        'Solar Co-located': 'Planned Solar Co-located',
        'Wind Co-located': 'Planned Wind Co-located',
        'Solar_Colocated': 'Planned Solar Co-located',
        'Planned (Planned)': 'Planned (from Planned projects)',
        'Planned (Planned_Solar)': 'Planned Solar Co-located',
        'LLM Match': 'LLM Matched (likely operational)',
        'LLM Match (Planned)': 'LLM Matched to Planned project'
    }
    
    unified['IQ_Source_Clarified'] = unified['IQ_Source'].map(iq_source_clarification).fillna(unified['IQ_Source'])
    
    # 2. Add Load Resource explanation
    print("2️⃣ Adding Load Resource status explanation...")
    
    def explain_load_resource_status(row):
        """Explain why a BESS may lack Load Resource"""
        if pd.notna(row['BESS_Load_Resource']):
            return 'Has Load Resource - Operational standalone or market-charging'
        else:
            # No Load Resource - explain why
            if row['Operational_Status'] == 'Operational':
                # Operational but no Load Resource
                if 'solar' in str(row.get('BESS_Type', '')).lower() or 'wind' in str(row.get('BESS_Type', '')).lower():
                    return 'No Load Resource - Likely charges from co-located renewable'
                else:
                    return 'No Load Resource - Operational co-located BESS charging from renewable'
            elif row['Operational_Status'] == 'Planned/Construction':
                return 'No Load Resource - Planned project (not yet operational)'
            else:
                return 'No Load Resource - Status unknown'
    
    unified['Load_Resource_Explanation'] = unified.apply(explain_load_resource_status, axis=1)
    
    # 3. Add true operational status based on all evidence
    print("3️⃣ Determining true operational status...")
    
    def determine_true_status(row):
        """Determine true operational status from all evidence"""
        # If has Load Resource, very likely operational
        if pd.notna(row['BESS_Load_Resource']):
            return 'Operational (has Load Resource)'
        
        # Check IQ Source
        if row['IQ_Source'] == 'Operational':
            return 'Operational (co-located, no Load Resource needed)'
        elif row['IQ_Source'] == 'Standalone':
            # This is from Stand-Alone sheet which is 698/701 PLANNED
            return 'Planned (from Stand-Alone sheet)'
        elif 'Planned' in str(row['IQ_Source']):
            return 'Planned/Under Construction'
        elif row['IQ_Source'] == 'LLM Match':
            # Could be either
            if row.get('IQ_Pass') == 'Pass 1' or row.get('IQ_Pass') == 'Pass 2':
                return 'Likely Operational (LLM matched to operational)'
            else:
                return 'Likely Planned (LLM matched to planned)'
        else:
            return 'Unknown'
    
    unified['True_Operational_Status'] = unified.apply(determine_true_status, axis=1)
    
    # 4. Add notes column with key insights
    print("4️⃣ Adding clarification notes...")
    
    def add_notes(row):
        """Add helpful notes about each BESS"""
        notes = []
        
        # Note about standalone confusion
        if row['IQ_Source'] == 'Standalone':
            notes.append('IQ "Standalone" = PLANNED standalone project (not operational)')
        
        # Note about missing Load Resource
        if pd.isna(row['BESS_Load_Resource']):
            if row['Operational_Status'] == 'Operational':
                notes.append('Operational w/o Load Resource = likely co-located charging')
            elif 'Planned' in str(row['True_Operational_Status']):
                notes.append('No Load Resource because not yet operational')
        
        # Note about co-location
        if 'Solar' in str(row.get('BESS_Type', '')):
            notes.append('Solar co-located BESS')
        elif 'Wind' in str(row.get('BESS_Type', '')):
            notes.append('Wind co-located BESS')
        
        return '; '.join(notes) if notes else ''
    
    unified['Clarification_Notes'] = unified.apply(add_notes, axis=1)
    
    # 5. Reorder columns for clarity
    print("5️⃣ Reordering columns for better clarity...")
    
    # Define column order (key columns first)
    priority_cols = [
        'BESS_Gen_Resource',
        'BESS_Load_Resource',
        'True_Operational_Status',
        'Load_Resource_Explanation',
        'Settlement_Point',
        'Substation',
        'Load_Zone',
        'BESS_Type',
        'IQ_Source_Clarified',
        'IQ_Capacity_MW',
        'EIA_Plant_Name',
        'EIA_Capacity_MW',
        'Clarification_Notes'
    ]
    
    # Get remaining columns
    other_cols = [col for col in unified.columns if col not in priority_cols]
    
    # Reorder
    unified = unified[priority_cols + other_cols]
    
    # Sort by operational status and completeness
    unified = unified.sort_values(
        ['True_Operational_Status', 'Data_Completeness_%', 'BESS_Gen_Resource'],
        ascending=[True, False, True]
    )
    
    # Save updated file
    output_file = '/home/enrico/projects/power_market_pipeline/BESS_UNIFIED_MAPPING_V3_CLARIFIED.csv'
    unified.to_csv(output_file, index=False)
    
    # Summary
    print("\n" + "="*70)
    print("UPDATE SUMMARY")
    print("="*70)
    
    print("\n✅ Added Clarifications:")
    print("  1. IQ_Source_Clarified - explains misleading 'Standalone' name")
    print("  2. Load_Resource_Explanation - why Load Resource may be missing")
    print("  3. True_Operational_Status - based on all evidence")
    print("  4. Clarification_Notes - key insights for each BESS")
    
    print("\n📊 True Operational Status Distribution:")
    status_counts = unified['True_Operational_Status'].value_counts()
    for status, count in status_counts.items():
        print(f"  {status}: {count}")
    
    print(f"\n✅ Updated file saved: {output_file}")
    
    # Show sample of clarified data
    print("\n📋 Sample of clarified data:")
    sample_cols = ['BESS_Gen_Resource', 'True_Operational_Status', 'Load_Resource_Explanation', 'Clarification_Notes']
    print(unified[sample_cols].head(10).to_string(index=False))
    
    return unified

if __name__ == '__main__':
    unified = update_with_clarifications()
    
    print("\n" + "="*70)
    print("KEY CLARIFICATIONS ADDED")
    print("="*70)
    
    print("""
1. IQ_Source "Standalone" CLARIFIED:
   - Actually means from "Stand-Alone" Excel sheet
   - Which contains 698/701 PLANNED projects
   - NOT operational standalone batteries!

2. Load Resource Missing EXPLAINED:
   - Planned projects: Not yet operational
   - Operational co-located: Charge from renewable, not market
   - Only 9 operational BESS lack Load Resources (co-located)

3. True Status DETERMINED:
   - Based on all evidence (Load Resource, IQ Source, etc.)
   - Clearly distinguishes operational vs planned
   - Explains the misleading naming conventions
""")
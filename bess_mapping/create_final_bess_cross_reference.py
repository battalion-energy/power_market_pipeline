#!/usr/bin/env python3
"""
Create final comprehensive BESS cross-reference file
Combines BESS mapping, interconnection queue matches, and additional metadata
"""

import pandas as pd
from pathlib import Path
import numpy as np

def create_comprehensive_mapping():
    """Create comprehensive BESS mapping with all available data"""
    
    print('Creating Comprehensive BESS Cross-Reference')
    print('='*60)
    
    # Load base BESS mapping
    bess_mapping = pd.read_csv('/home/enrico/projects/power_market_pipeline/BESS_RESOURCE_MAPPING_REAL_ONLY.csv')
    print(f'Loaded {len(bess_mapping)} BESS resources')
    
    # Load interconnection matches
    iq_matches = pd.read_csv('/home/enrico/projects/power_market_pipeline/BESS_INTERCONNECTION_MATCHED.csv')
    print(f'Loaded {len(iq_matches)} interconnection matches')
    
    # Merge the data
    comprehensive = bess_mapping.merge(
        iq_matches,
        on=['BESS_Gen_Resource', 'BESS_Load_Resource', 'Settlement_Point', 'Substation', 'Load_Zone'],
        how='left',
        suffixes=('', '_iq')
    )
    
    # Clean up and organize columns
    column_order = [
        # Core BESS identification
        'BESS_Gen_Resource',
        'BESS_Load_Resource',
        'Settlement_Point',
        'Substation',
        'Load_Zone',
        
        # Interconnection queue match info
        'match_score',
        'match_reason',
        'Source',
        
        # Operational data (if matched)
        'Unit_Name',
        'Unit_Code',
        
        # Project details from interconnection queue
        'IQ_INR',
        'IQ_Project_Name',
        'IQ_Status',
        'IQ_Entity',
        'IQ_POI',
        'IQ_County',
        'IQ_Zone',
        'IQ_Capacity_MW',
        'IQ_In_Service',
        'IQ_COD',
        'IQ_IA_Signed',
        'IQ_Fuel',
        'IQ_Technology',
        
        # Estimated county (if no IQ match)
        'Estimated_County'
    ]
    
    # Keep only columns that exist
    available_columns = [col for col in column_order if col in comprehensive.columns]
    comprehensive = comprehensive[available_columns]
    
    # Add match quality category
    def categorize_match(score):
        if pd.isna(score) or score == 0:
            return 'No Match'
        elif score >= 90:
            return 'Excellent'
        elif score >= 70:
            return 'Good'
        elif score >= 50:
            return 'Fair'
        else:
            return 'Poor'
    
    comprehensive['Match_Quality'] = comprehensive['match_score'].apply(categorize_match)
    
    # Add operational status
    comprehensive['Is_Operational'] = comprehensive['Source'] == 'Operational'
    
    # Fill in county where possible
    comprehensive['County'] = comprehensive['IQ_County'].fillna(comprehensive['Estimated_County'])
    
    # Sort by match quality and resource name
    comprehensive = comprehensive.sort_values(['Match_Quality', 'BESS_Gen_Resource'])
    
    # Save comprehensive file
    output_file = '/home/enrico/projects/power_market_pipeline/BESS_COMPREHENSIVE_CROSS_REFERENCE.csv'
    comprehensive.to_csv(output_file, index=False)
    print(f'\n✅ Saved comprehensive cross-reference to: {output_file}')
    
    # Create summary statistics
    print('\n=== Cross-Reference Summary ===')
    print(f'Total BESS Resources: {len(comprehensive)}')
    print(f'With Load Resources (verified): {comprehensive["BESS_Load_Resource"].notna().sum()}')
    print(f'With Interconnection Match: {(comprehensive["match_score"] > 0).sum()}')
    print(f'Operational (confirmed): {comprehensive["Is_Operational"].sum()}')
    
    print('\n=== Match Quality Distribution ===')
    quality_counts = comprehensive['Match_Quality'].value_counts()
    for quality, count in quality_counts.items():
        pct = 100 * count / len(comprehensive)
        print(f'{quality:12} {count:3} ({pct:5.1f}%)')
    
    print('\n=== Load Zone Distribution ===')
    zone_counts = comprehensive['Load_Zone'].value_counts()
    for zone, count in zone_counts.items():
        pct = 100 * count / len(comprehensive)
        print(f'{zone:12} {count:3} ({pct:5.1f}%)')
    
    # Create a simplified version for easy use
    simplified_cols = [
        'BESS_Gen_Resource',
        'BESS_Load_Resource',
        'Settlement_Point',
        'Substation',
        'County',
        'Load_Zone',
        'Match_Quality',
        'Is_Operational',
        'IQ_Capacity_MW',
        'IQ_In_Service',
        'IQ_Project_Name',
        'IQ_Entity'
    ]
    
    available_simplified = [col for col in simplified_cols if col in comprehensive.columns]
    simplified = comprehensive[available_simplified].copy()
    
    simplified_file = '/home/enrico/projects/power_market_pipeline/BESS_CROSS_REFERENCE_SIMPLIFIED.csv'
    simplified.to_csv(simplified_file, index=False)
    print(f'\n✅ Saved simplified version to: {simplified_file}')
    
    # Show sample of excellent matches
    excellent = comprehensive[comprehensive['Match_Quality'] == 'Excellent']
    if len(excellent) > 0:
        print('\n=== Sample Excellent Matches ===')
        sample_cols = ['BESS_Gen_Resource', 'Unit_Code', 'IQ_Project_Name', 'IQ_Capacity_MW', 'IQ_In_Service']
        available_sample = [col for col in sample_cols if col in excellent.columns]
        print(excellent[available_sample].head(10).to_string(index=False))
    
    # Create a report of unmatched resources for investigation
    unmatched = comprehensive[comprehensive['Match_Quality'] == 'No Match']
    if len(unmatched) > 0:
        report_file = '/home/enrico/projects/power_market_pipeline/BESS_UNMATCHED_REPORT.csv'
        unmatched_report = unmatched[['BESS_Gen_Resource', 'Substation', 'Load_Zone', 'County']].copy()
        unmatched_report.to_csv(report_file, index=False)
        print(f'\n✅ Saved unmatched resources report to: {report_file}')
        print(f'   {len(unmatched)} resources need manual investigation')
    
    return comprehensive

def create_markdown_documentation():
    """Create markdown documentation for the cross-reference"""
    
    doc_content = """# BESS Cross-Reference Documentation

## Overview
This document describes the comprehensive BESS (Battery Energy Storage System) cross-reference created by matching ERCOT operational data with the interconnection queue.

## Data Sources

### 1. BESS Resource Mapping (Real Data Only)
- **File**: `BESS_RESOURCE_MAPPING_REAL_ONLY.csv`
- **Source**: ERCOT DAM/SCED operational data
- **Contents**: 195 BESS Generation resources with verified Load Resources

### 2. ERCOT Interconnection Queue
- **File**: `interconnection_queue.xlsx`
- **Sheets Processed**:
  - Co-located Operational (127 projects)
  - Stand-Alone (701 projects)
  - Co-located with Solar (635 projects)
  - Co-located with Wind (28 projects)
  - Co-located with Thermal (3 projects)

## Matching Methodology

### Primary Matching (Operational Data)
1. **Exact Unit Code Match**: Direct match between BESS_Gen_Resource and Unit_Code
2. **Similarity Match**: Fuzzy matching with >80% similarity threshold
3. **County Verification**: Additional points for county match

### Secondary Matching (Queue Data)
1. **Project Name Similarity**: >60% similarity threshold
2. **POI Location Match**: Matches with substation (>70% similarity)
3. **County Match**: Exact county match adds confidence

## Match Quality Categories

- **Excellent (≥90)**: High confidence, usually exact Unit Code match
- **Good (70-90)**: Strong similarity with supporting evidence
- **Fair (50-70)**: Moderate confidence, may need verification
- **Poor (<50)**: Low confidence, manual review recommended
- **No Match**: No suitable match found in interconnection data

## Output Files

### Main Files
1. **BESS_COMPREHENSIVE_CROSS_REFERENCE.csv**: Complete dataset with all fields
2. **BESS_CROSS_REFERENCE_SIMPLIFIED.csv**: Essential fields for daily use
3. **BESS_UNMATCHED_REPORT.csv**: Resources requiring manual investigation

### Supporting Files
- `interconnection_queue_clean/`: Cleaned interconnection queue data
- `BESS_INTERCONNECTION_MATCHED.csv`: Raw matching results

## Key Fields

### Identification
- `BESS_Gen_Resource`: Generation resource name from ERCOT
- `BESS_Load_Resource`: Load resource name (if verified to exist)
- `Settlement_Point`: Price settlement point
- `Substation`: Physical connection point

### Match Information
- `match_score`: Numerical confidence score (0-110)
- `Match_Quality`: Categorical quality (Excellent/Good/Fair/Poor/No Match)
- `match_reason`: Explanation of match logic
- `Source`: Data source (Operational/Standalone/Solar Co-located/etc.)

### Project Details (from Interconnection Queue)
- `IQ_Project_Name`: Official project name
- `IQ_Capacity_MW`: Rated capacity in megawatts
- `IQ_In_Service`: Year entered service
- `IQ_Entity`: Owner/operator entity
- `IQ_County`: Texas county location

## Usage Notes

1. **For Revenue Calculations**: Use BESS_Gen_Resource and BESS_Load_Resource with Settlement_Point
2. **For Capacity Analysis**: Use IQ_Capacity_MW where available
3. **For Geographic Analysis**: Use County and Load_Zone fields
4. **For Operational Status**: Check Is_Operational flag

## Data Quality Notes

- 62.6% of BESS have verified Load Resources
- 22.1% have excellent interconnection queue matches
- 40% have no interconnection match (may be newer or use different names)

## Update Process

1. Re-run `create_bess_resource_mapping_REAL_ONLY.py` monthly
2. Download latest interconnection queue from ERCOT
3. Run `match_bess_with_interconnection.py`
4. Run `create_final_bess_cross_reference.py`
5. Review unmatched report for manual investigation

## Contact
For questions or corrections, contact the data team.

Generated: """ + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    
    doc_file = Path('/home/enrico/projects/power_market_pipeline/BESS_CROSS_REFERENCE_README.md')
    doc_file.write_text(doc_content)
    print(f'\n✅ Created documentation: {doc_file}')

if __name__ == '__main__':
    # Create comprehensive mapping
    comprehensive_df = create_comprehensive_mapping()
    
    # Create documentation
    create_markdown_documentation()
    
    print('\n' + '='*60)
    print('✅ BESS Cross-Reference Creation Complete!')
    print('='*60)
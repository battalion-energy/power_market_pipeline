# BESS Cross-Reference Documentation

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

Generated: 2025-09-01 11:47:35
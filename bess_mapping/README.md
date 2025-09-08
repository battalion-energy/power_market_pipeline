# BESS Mapping Pipeline

This folder contains the complete Battery Energy Storage System (BESS) mapping pipeline for ERCOT data.

## Purpose
Maps ERCOT BESS resources to their physical locations and characteristics by cross-referencing:
- ERCOT Resource Nodes → Settlement Points
- Interconnection Queue data (county/capacity information)
- EIA generator database (coordinates/operational details)

## Main Files

### Pipeline Script
- `complete_bess_mapping_pipeline.py` - Main pipeline that performs all mapping and validation

### Input Data Files
- `BESS_ERCOT_MAPPING_TABLE.csv` - ERCOT resource to settlement point mapping
- `BESS_INTERCONNECTION_MATCHED.csv` - Pre-matched interconnection queue data
- `BESS_EIA_COMPREHENSIVE_VERIFIED.csv` - Verified EIA generator database
- `interconnection_queue_data/` - Raw interconnection queue CSV files

### Output Files
- `BESS_COMPLETE_MAPPING_PIPELINE.csv` - Pipeline output with all matches
- `BESS_COMPREHENSIVE_WITH_COORDINATES_V2.csv` - Final comprehensive mapping with coordinates

## Usage

### From project root:
```bash
make bess-mapping
```

### Or directly:
```bash
cd bess_mapping
python3 complete_bess_mapping_pipeline.py
```

## Key Features
- Fuzzy name matching between different data sources
- County-based validation
- Distance validation from county centers
- Capacity matching and validation
- Special handling for known issues (e.g., CROSSETT location correction)

## Data Quality Notes
- Successfully maps ~195 BESS resources
- County data available for ~9 resources (4.6%)
- Coordinate data depends on EIA matches
- Ongoing improvements to matching algorithms

## Related Files (Historical)
The folder contains various historical mapping attempts and intermediate files from the development process:
- Various BESS_*_MATCHED.csv files - Different matching approaches
- BESS_UNIFIED_MAPPING*.csv - Previous unified mapping attempts
- BESS_*_VALIDATION.csv - Validation runs on different datasets

The current production pipeline uses the files listed in "Input Data Files" above.
# ERCOT Power Market Pipeline Documentation

## Overview
This directory contains comprehensive specifications and documentation for the ERCOT Power Market Pipeline, including methodologies, data structures, and analysis results.

## Quick Start

### 🔋 BESS Revenue Analysis
- **[BESS Revenue Methodology](BESS_REVENUE_METHODOLOGY.md)** - Complete methodology for calculating BESS revenues
- **[BESS Results Summary](BESS_RESULTS_SUMMARY.md)** - Data locations and access methods
- **[BESS Revenue Analysis Results](BESS_Revenue_Analysis_Results.md)** - Detailed analysis findings

### 📊 Data Locations
```
Primary Results: /home/enrico/data/ERCOT_data/bess_analysis/
Database Export: /home/enrico/data/ERCOT_data/bess_analysis/database_export/
Temporary Files: /tmp/
```

## Documentation Index

### Core Methodologies
1. **[BESS Revenue Methodology](BESS_REVENUE_METHODOLOGY.md)** ⭐
   - Forensic accounting approach for historical BESS revenues
   - DA, RT, and AS revenue calculations
   - Settlement point mapping and time alignment

2. **[ERCOT Generator Mapping](ERCOT_GENERATOR_MAPPING_DOCUMENTATION.md)** 🆕 ✨
   - Comprehensive mapping for ALL ERCOT generators (Gas, Wind, Solar, BESS)
   - **68.2% match rate** (871/1,278 resources with coordinates)
   - LLM-assisted code decoding using Claude AI
   - Settlement point and substation mapping

3. **[TBX Calculator](TBX_CALCULATOR.md)**
   - TB2/TB4 battery arbitrage value calculations
   - Perfect foresight optimization methodology

4. **[BESS Revenue Accounting Approach](BESS_Revenue_Accounting_Approach.md)**
   - Historical analysis vs optimization distinction
   - Data sources and processing flow

### Data Structures & Schemas
1. **[ERCOT Parquet Schemas](ERCOT_Parquet_Schemas.md)**
   - Complete schema documentation for all parquet files
   - Data types and column descriptions

2. **[60-Day Disclosure Data](60-Day_Disclosure_Data.md)**
   - ERCOT disclosure file formats
   - DAM and SCED resource data structures

3. **[ERCOT Price File Structures](ERCOT_Price_File_Structures.md)**
   - DA, RT, and AS price file formats
   - Settlement point types and pricing nodes

4. **[Resource Settlement Point Mapping](RESOURCE_SETTLEMENT_POINT_MAPPING.md)**
   - BESS-specific settlement point relationships
   - Load zone mappings

5. **[BESS Unified Mapping](BESS_UNIFIED_MAPPING_DOCUMENTATION.md)**
   - Comprehensive BESS resource mapping
   - Interconnection queue and EIA integration

### Processing Documentation
1. **[ERCOT Annual Parquet Processor](ERCOT_Annual_Parquet_Processor.md)**
   - High-performance Rust processor documentation
   - Parallel processing architecture

2. **[Parallelization Architecture](parallelization-architecture.md)**
   - Multi-threading strategy
   - Memory management and optimization

3. **[Date Handling Documentation](Date_Handling_Documentation.md)**
   - Date32 format and timezone handling
   - DST flag evolution

### Analysis Results
1. **[BESS Results Summary](BESS_RESULTS_SUMMARY.md)** ⭐
   - Current results location and structure
   - Key findings and statistics

2. **[BESS Revenue Analysis Results](BESS_Revenue_Analysis_Results.md)**
   - Detailed revenue breakdowns
   - Top performing resources

3. **[BESS Revenue Algorithm Audit](BESS_Revenue_Algorithm_Audit.md)**
   - Validation and verification procedures

### Technical Fixes & Updates
1. **[Float64 Schema Fix Summary](Float64_Schema_Fix_Summary.md)**
   - Price column type enforcement

2. **[COP File Format Evolution](COP_File_Format_Evolution.md)**
   - Historical format changes

3. **[RT Processing Order Change](RT_Processing_Order_Change.md)**
   - Real-time data processing improvements

## Key Commands

### Running Analysis
```bash
# Full BESS revenue analysis
make bess-leaderboard

# Quick test (10 resources)
python simple_bess_test.py

# TBX arbitrage calculation
make tbx

# Compare Python vs Rust
python compare_bess_results.py

# Generator mapping pipeline
python3 complete_generator_mapping_pipeline_final.py
```

### Accessing Results
```python
import pandas as pd

# Load results
df = pd.read_parquet('/home/enrico/data/ERCOT_data/bess_analysis/database_export/bess_daily_revenues.parquet')

# Top performers
top_10 = df.groupby('resource_name')['total_revenue'].sum().nlargest(10)
```

## Data Pipeline Flow
```
ERCOT Raw Data (CSV/ZIP)
    ↓
CSV Extraction & Validation
    ↓
Parquet Conversion (Schema Normalized)
    ↓
Revenue Calculation (Python/Rust)
    ↓
Database Export (Parquet/JSON)
    ↓
NextJS Frontend / API
```

## Key Results Summary

### 2024 BESS Revenue Analysis (Sample)
- **Total Revenue**: $3,682,055 (10 resources)
- **Top Performer**: BATCAVE_BES1 ($2.76M)
- **Revenue Split**: 60-95% AS, 5-40% DA
- **Processing Time**: 2.8 seconds

### Generator Mapping Results (December 2024) ✨
- **Total ERCOT Resources**: 1,278
- **Resources with Coordinates**: 871 (68.2%) ✓
- **LLM-Generated Mappings**: 393 ERCOT codes decoded
- **Direct EIA Matches**: 832
- **Via IQ Matches**: 39
- **With Substation Data**: 77 (6.0%)
- **Interconnection Queue Projects**: 1,905 (Large: 1,849, Small: 56)

### Data Coverage
- **Years**: 2019-2024
- **BESS Resources**: 582 identified
- **All Generators**: 1,278 mapped
- **Data Frequency**: Hourly (DA), 5-min (RT)
- **File Size**: ~50GB compressed

## Support & Contact

For questions or issues:
1. Check relevant documentation in this directory
2. Review code in `/home/enrico/projects/power_market_pipeline/`
3. Submit issues to project repository

## Version History
- v1.0 (Aug 2024): Initial BESS revenue methodology
- v1.1 (Aug 2024): Added TBX calculator and reports
- v1.2 (Aug 2024): Rust implementation and parallelization
- v1.3 (Dec 2024): Added comprehensive generator mapping for all resource types
- v1.4 (Dec 2024): Enhanced to 68.2% match rate using LLM-assisted decoding

---
*Last Updated: December 5, 2024*
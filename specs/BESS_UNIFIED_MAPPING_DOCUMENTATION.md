# BESS Unified Mapping Documentation

## Overview
This document describes the comprehensive BESS (Battery Energy Storage System) resource mapping for ERCOT, including the complete data chain from BESS resources to settlement points, substations, interconnection queue projects, and EIA generator data.

## Final Output Files

### Primary File: `BESS_COMPREHENSIVE_WITH_COORDINATES.csv`
- **209 BESS resources** with **95 data columns** (expanded from V3's 32 columns)
- Complete mapping chain: BESS → Settlement Point → Substation → Interconnection Queue → EIA Generator
- Includes clarifications for operational status and Load Resource presence
- **NEW:** Geographic coordinates with 81.3% coverage
- **NEW:** Complete IQ and EIA data fields

### Previous Version: `BESS_UNIFIED_MAPPING_V3_CLARIFIED.csv`
- 197 BESS resources with 32+ data columns
- Base mapping without coordinates or full IQ/EIA details

## Key Findings

### 1. Operational vs Planned BESS

**Distribution (Updated with 209 total BESS):**
- **~140 Operational BESS** (67%)
  - 124 have Load Resources (standalone or market-charging)
  - 16 operational without Load Resources (co-located, charge from renewable)
- **~69 Planned BESS** (33%)
  - Will receive Load Resources when operational
  - COD dates range from 2025-2029

### 2. Standalone vs Co-located Classification

**Critical Insight:** Only ~50 BESS in ERCOT are truly standalone
- **Operational BESS (122 total):**
  - 50 standalone (Fuel=OTH in Co-located Operational tab)
  - 51 solar co-located (Fuel=SOL)
  - 21 wind co-located (Fuel=WIN)
  - 4 thermal co-located (Fuel=GAS)

**The "Standalone" Confusion - RESOLVED:**
- ERCOT's "Stand-Alone" Excel sheet contains **PLANNED** standalone projects (698/701 are status="Planned")
- These are **NOT** operational standalone batteries
- Operational standalone BESS are in "Co-located Operational" sheet with Fuel=OTH

### 3. Load Resource Analysis

**Why Some BESS Lack Load Resources:**

| Reason | Count | Percentage | Explanation |
|--------|-------|------------|-------------|
| Has Load Resource | 124 | 62.9% | Operational standalone or market-charging BESS |
| Planned Project | 58 | 29.4% | Not yet operational, will get Load Resource when built |
| Co-located Charging | 9 | 4.6% | Operational but charge from renewable, not market |
| Unknown | 6 | 3.0% | Status unclear |

**Key Insight:** Co-located BESS that charge directly from solar/wind:
- Don't need DAM/RTM Load Resource registration
- Avoid transmission charges and grid losses
- Use curtailed renewable energy

### 4. Data Coverage Statistics

| Data Element | Coverage | Notes |
|--------------|----------|-------|
| Settlement Points | 100% | Required for all BESS |
| Interconnection Queue Match | 97.0% | Excellent coverage |
| Substations | 95.4% | Nearly complete |
| IQ Capacity Data | 90.9% | From Capacity (MW)* column |
| Load Resources | 62.9% | Only for operational standalone/market-charging |
| EIA Generator Match | 55.8% | Limited by co-location reporting |

### 5. EIA Matching Challenges

**Why EIA Matching is Only 56%:**

1. **Co-location Issue (Primary):**
   - 72 of 122 operational BESS are co-located
   - EIA reports them under parent solar/wind plant
   - Not separately listed as battery facilities

2. **Match Rates by Type:**
   - Standalone BESS: 59% match to EIA
   - Solar Co-located: Only 28% match
   - Co-located BESS are "invisible" in EIA battery searches

3. **Other Factors:**
   - Reporting lag (EIA monthly vs ERCOT daily)
   - Small facilities (<10 MW) may be below thresholds
   - Naming mismatches (ERCOT: ANCHOR_BESS1 vs EIA: "Anchor Battery Storage")

## Data Sources

### 1. ERCOT Sources
- **60-Day DAM/SCED Disclosure Files:** Verified Load Resources from actual market data
- **Interconnection Queue Excel:** `interconnection_queue.xlsx` with multiple sheets
- **Settlement Point Mappings:** Latest mapping files from ERCOT

### 2. Interconnection Queue Structure

| Excel Sheet | Content | Status |
|-------------|---------|---------|
| Co-located Operational | ALL operational BESS (standalone + co-located) | Currently Operating |
| Stand-Alone | Planned standalone BESS projects | 698/701 are Planned |
| Co-located with Solar | Planned solar+battery projects | Future Projects |
| Co-located with Wind | Planned wind+battery projects | Future Projects |

### 3. EIA Generator Data
- **Source:** `EIA_generators_latest.xlsx`
- **Sheets Used:** Operating and Planned tabs
- **Filter:** Texas battery storage facilities only

## Key Columns in Comprehensive Mapping

### Identification Columns
- `BESS_Gen_Resource`: Generation resource name
- `BESS_Load_Resource`: Load resource name (if exists)
- `True_Operational_Status`: Clarified operational status

### Location/Market Columns
- `Settlement_Point`: ERCOT settlement point
- `Substation`: Physical substation
- `Load_Zone`: ERCOT load zone

### Interconnection Queue Columns
- `IQ_Source_Clarified`: Explains misleading source names
- `IQ_Project_Name`: Project name from interconnection queue
- `IQ_Capacity_MW`: Capacity from interconnection queue
- `IQ_match_score`: Confidence of IQ match (0-100)

### EIA Columns
- `EIA_Plant_Name`: EIA plant name
- `EIA_Generator_ID`: EIA generator ID
- `EIA_Capacity_MW`: Capacity from EIA
- `EIA_match_score`: Confidence of EIA match (0-100)

### Clarification Columns (New in V3)
- `Load_Resource_Explanation`: Why Load Resource may be missing
- `IQ_Source_Clarified`: Explains "Standalone" confusion
- `Clarification_Notes`: Key insights for each BESS
- `Data_Completeness_%`: Percentage of fields populated

### Geographic Columns (New in Comprehensive)
- `Latitude`: Final latitude coordinate
- `Longitude`: Final longitude coordinate
- `Coordinate_Source`: Source of coordinates (EIA Data, Google Places, County Center)
- `EIA_Latitude`: Original latitude from EIA
- `EIA_Longitude`: Original longitude from EIA

### Expanded IQ Columns (22 new fields)
- `IQ_POI Location`: Point of Interconnection details
- `IQ_Interconnecting Entity`: TSP/DSP information
- `IQ_IA Signed`: Interconnection Agreement date
- `IQ_Approved for Energization`: Energization approval date
- `IQ_Approved for Synchronization`: Sync approval date
- `IQ_Comment`: Additional project notes
- Plus 16 more detailed IQ fields

### Expanded EIA Columns (37 new fields)
- `EIA_Entity ID` and `EIA_Entity Name`: Owner information
- `EIA_Plant ID`: Unique plant identifier
- `EIA_Google Map` and `EIA_Bing Map`: Map links
- `EIA_Nameplate Energy Capacity (MWh)`: Battery energy capacity
- `EIA_DC Net Capacity (MW)`: DC-side capacity
- `EIA_Operating Month/Year`: Commercial operation date
- `EIA_Planned Retirement Month/Year`: Expected retirement
- Plus 30 more detailed EIA fields

## Data Quality Notes

### High Quality Data
- 100% have Settlement Points (required)
- 97% matched to Interconnection Queue
- 95% have Substation mapping
- 91% have Capacity data

### Data Gaps
- 44% lack EIA match (mostly co-located, reported under parent)
- 37% lack Load Resources (mostly planned projects or co-located)

### Verification
- All matches based on REAL data only
- No fabricated or predicted values
- Load Resources verified against 4933 DAM files and 1855 SCED files
- IQ matches verified against official ERCOT interconnection queue
- EIA matches verified against official EIA generator reports

## Usage Guidelines

1. **For Operational Analysis:** Filter by `True_Operational_Status` = "Operational*"
2. **For Market Analysis:** Use only BESS with `BESS_Load_Resource` populated
3. **For Capacity Planning:** Include both operational and planned with COD dates
4. **For Co-location Studies:** Check `BESS_Type` and `Clarification_Notes`

## Update History

- **V1:** Initial unified mapping
- **V2:** Added IQ capacity data and operational status
- **V3:** Added clarifications for "Standalone" confusion and Load Resource explanations
- **Comprehensive:** Added geographic coordinates, complete IQ data (22 new fields), complete EIA data (37 new fields), expanded to 209 BESS resources

## Contact
For questions about this mapping, refer to the scripts in `/home/enrico/projects/power_market_pipeline/`:
- `create_bess_mapping_with_coordinates.py` (Latest - generates comprehensive file)
- `create_unified_bess_mapping_v2.py`
- `update_unified_with_clarifications.py`
- `verify_match_integrity.py`

## Geographic Coordinate Methods

The comprehensive mapping adds coordinates using three methods:

1. **EIA Data (Primary):** Uses EIA-reported lat/long when available (110 resources)
2. **Google Places API:** Geocodes substation names with county context (20 resources)
3. **County Centers (Fallback):** Uses Texas county center coordinates (40 resources)

**Coverage:** 81.3% of BESS resources now have geographic coordinates
# CAISO Historical Data Investigation (2019-2021)

**Date**: 2025-10-25
**Objective**: Find sources for CAISO nodal LMP data before 2022

## Summary of Findings

### ❌ OASIS API Limitation
**Finding**: CAISO OASIS API (`oasis.caiso.com/oasisapi`) does **NOT** provide data before 2022-01-01.

**Evidence**:
- Direct API queries for 2019-2021 dates return error code 1000: "No data returned for the specified selection"
- XML responses contain only metadata/headers, no actual price data
- Retention policy: OASIS API maintains approximately **39 months** of historical data
- CSV format data starts from 2022-01-01

**Conclusion**: Historical nodal LMP data for 2019-2021 is **not available** via standard OASIS API queries.

## Alternative Data Sources Investigated

### 1. CAISO Bulk Download Portal
**URL**: `https://oasis-bulk.caiso.com/`

**Status**: ⚠️ Limited Investigation
- Site is a React single-page application (requires JavaScript)
- Cannot easily determine available data ranges without interactive browsing
- Potential source but requires manual exploration

**Action Required**: Manual visit to site to browse available bulk data archives

### 2. Third-Party Data Providers

#### Michael Champ's CAISO Data Project
**URL**: `https://projects.michaelchamp.com/CAISOData.html`

**Data Available**:
- LMP and component prices
- Hub locations: NP15, SP15, ZP26
- Format: TXT file download or JSON API
- **Limitation**: Hub data only, NOT nodal

**Assessment**: ❌ Not suitable - we need nodal data, not just hubs

#### Commercial Data Products
**Sources**: Barchart.com, LCG Consulting, EnergyOnline

**Assessment**:
- Commercial/premium services (paid)
- May have historical data but uncertain about nodal coverage
- Not free/public access

### 3. GitHub Open Source Projects

**Found**:
- `romilandc/CAISO-OASIS-LMP-downloader` - Script to download LMP data via OASIS API
- `emunsing/CAISO-Scrapers` - Tools to extract data from CAISO website
- `gridstatus` library - Python interface for CAISO data

**Limitation**: All rely on OASIS API, which has the 39-month retention limit

## Data Retention Policies

### CAISO OASIS System
| Data Type | Retention Period | Start Date Available |
|-----------|------------------|---------------------|
| DA LMP (CSV) | ~39 months | 2022-01-01 |
| RT 5-Min LMP (CSV) | ~39 months | 2022-01-01 |
| Ancillary Services (CSV) | ~39 months | 2022-01-01 |
| Older data (XML) | Archived | Not accessible via API |

### Historical Data Archive Location
**Unknown**: CAISO may maintain historical archives, but:
- Not publicly documented
- Not accessible via current OASIS API
- May require contacting CAISO directly

## Recommendations

### Option 1: Accept 2022+ Coverage (RECOMMENDED)
**Pros**:
- 3+ years of high-quality data (2022-2025)
- Complete nodal coverage (~15K nodes)
- Automated daily updates via cron
- Already implemented and working

**Cons**:
- Missing 2019-2021 historical period

**Use Cases Supported**:
- Recent market analysis (last 3 years)
- BESS optimization modeling
- Price forecasting with recent trends
- Sufficient for most practical applications

### Option 2: Contact CAISO for Historical Archives
**Process**:
1. Register at `developer.caiso.com` (requires organizational email)
2. Submit data request through official channels
3. Inquire about historical data archives for 2019-2021

**Potential Outcomes**:
- May provide access to archived data
- May have different format (bulk CSV/XML dumps)
- May require justification/approval
- Timeline unknown

**Pros**:
- Potential access to complete historical record
- Official source

**Cons**:
- Uncertain availability
- Potentially lengthy approval process
- May require business justification
- Unknown format/complexity

### Option 3: Alternative Analysis Approach
**Strategy**: Use hub data for 2019-2021, nodal data for 2022+

**Sources**:
- Michael Champ project: Hub LMPs (NP15, SP15, ZP26) for 2019-2021
- Our downloads: Nodal LMPs for 2022-2025

**Pros**:
- Can still analyze long-term trends at hub level
- Complements recent nodal data

**Cons**:
- Inconsistent granularity across time periods
- Limited spatial resolution for historical period

### Option 4: Use Commercial Data Sources
**Providers** (potential):
- Energy market data vendors
- Platts/S&P Global
- ICE/EIA databases

**Pros**:
- May have archived historical data
- Professional quality assurance

**Cons**:
- Cost (subscription/licensing fees)
- Access restrictions
- Uncertain nodal coverage

## Conclusion

**RECOMMENDED PATH**: **Option 1** - Accept 2022-2025 coverage

**Rationale**:
1. **3+ years of data** is sufficient for most BESS optimization and market analysis
2. **Complete nodal granularity** for the available period
3. **Production-ready pipeline** already implemented
4. **Automated daily updates** ensure data stays current
5. **Known data quality** (CSV format, validated structure)

**Additional Action** (Optional):
- Register at CAISO Developer Portal for future access to announcements
- Monitor for any historical data releases or bulk archive availability
- Consider Option 3 (hub data) if specific 2019-2021 analysis needed

## Technical Details

### Data Tested
- **2019-01-01**: XML format, error 1000 (no data)
- **2020-01-01**: XML format, error 1000 (no data)
- **2021-01-01**: XML format, error 1000 (no data)
- **2022-01-01**: CSV format, data available ✅
- **2023-01-01**: CSV format, data available ✅

### API Endpoints Tested
- `PRC_LMP` (DA Nodal LMPs) - 2019-2021: No data
- `PRC_INTVL_LMP` (RT 5-Min) - Not tested for 2019-2021
- `PRC_AS` (Ancillary Services) - Not tested for 2019-2021

---

**Status**: Investigation Complete
**Decision Required**: Choose option for handling 2019-2021 gap
**Current Downloads**: 2022-2025 data in progress (~28 hours remaining)

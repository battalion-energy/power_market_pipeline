# BESS Revenue Analysis Methodology

## Executive Summary

This document describes the methodology for calculating historical Battery Energy Storage System (BESS) revenues in ERCOT from 2019-2024. The analysis reconstructs actual revenues earned by BESS resources through energy arbitrage and ancillary services provision.

## Key Principle: Historical Analysis, Not Optimization

**IMPORTANT**: This is forensic revenue accounting of actual BESS operations, NOT an optimization model. We are calculating what batteries actually earned based on their recorded operations in ERCOT's 60-day disclosure data.

## Data Sources

### Primary Data Files
- **Day-Ahead Market (DAM) Generation Resources**: `60d_DAM_Gen_Resource_Data-*.csv`
  - Contains hourly awards for energy and ancillary services
  - Fields: ResourceName, SettlementPointName, DeliveryDate, HourEnding, AwardedQuantity, RegUpAwarded, RegDownAwarded, etc.
  
- **Day-Ahead Market (DAM) Load Resources**: `60d_DAM_Load_Resource_Data-*.csv`
  - Contains charging schedules for BESS operating as load
  - Used to track when batteries charge from the grid

- **Real-Time SCED Generation Resources**: `60d_SCED_Gen_Resource_Data-*.csv`
  - 5-minute interval dispatch data
  - Actual output levels and ancillary service deployments

- **Price Data**:
  - `DA_prices_YYYY.parquet`: Hourly day-ahead settlement point prices
  - `RT_prices_15min_YYYY.parquet`: 15-minute real-time prices
  - `AS_prices_YYYY.parquet`: Ancillary service clearing prices

### BESS Identification
BESS resources are identified by:
- ResourceType = 'PWRSTR' (Power Storage)
- Resource naming patterns (e.g., "_BESS", "_BES1", "_UNIT1")
- Cross-referenced with ERCOT's registered storage resources

## Revenue Streams

### 1. Day-Ahead Energy Revenue
```
DA Energy Revenue = Σ(AwardedQuantity × DA_Price)
```
- **Discharge Revenue**: Positive AwardedQuantity × Settlement Point Price
- **Charging Cost**: Negative AwardedQuantity × Settlement Point Price
- Net arbitrage = Discharge Revenue - Charging Cost

### 2. Real-Time Energy Revenue
```
RT Energy Revenue = Σ(BasePointDeviation × RT_Price)
```
- Deviations from day-ahead schedule settled at real-time prices
- Positive deviation = additional discharge revenue
- Negative deviation = reduced discharge or additional charging

### 3. Ancillary Services Revenue

#### a. Regulation Up (RegUp)
```
RegUp Revenue = Σ(RegUpAwarded × REGUP_Price)
```
- Payment for capability to increase output on AGC signal
- Capacity payment ($/MW-hr)

#### b. Regulation Down (RegDn)
```
RegDn Revenue = Σ(RegDownAwarded × REGDN_Price)
```
- Payment for capability to decrease output on AGC signal
- Capacity payment ($/MW-hr)

#### c. Responsive Reserve Service (RRS)
```
RRS Revenue = Σ(RRSAwarded × RRS_Price)
```
- Payment for 10-minute spinning reserve capability
- Capacity payment ($/MW-hr)

#### d. Non-Spinning Reserve (NonSpin)
```
NonSpin Revenue = Σ(NonSpinAwarded × NSPIN_Price)
```
- Payment for 30-minute reserve capability
- Can be offline but available

#### e. ERCOT Contingency Reserve Service (ECRS)
```
ECRS Revenue = Σ(ECRSAwarded × ECRS_Price)
```
- Fast frequency response service
- Premium payment for sub-10-minute response

### 4. Total Revenue Calculation
```
Total Revenue = DA Energy Revenue + RT Energy Revenue + AS Revenue
Where:
AS Revenue = RegUp + RegDn + RRS + NonSpin + ECRS
```

## Implementation Details

### Settlement Point Mapping
- Each BESS has an associated settlement point (node)
- Prices are matched based on SettlementPointName
- Fallback to hub prices (HB_BUSAVG) when specific node prices unavailable

### Time Alignment
- DA awards: Hourly (HourEnding 01:00 - 24:00)
- RT dispatch: 5-minute intervals
- AS awards: Hourly capacity payments
- All timestamps converted to UTC for consistency

### Data Quality Checks
1. Verify resource consistency across files
2. Check for missing price data
3. Validate award quantities (MW limits)
4. Ensure temporal continuity
5. Cross-check totals with ERCOT settlement statements

## Output Files

### 1. Daily Revenue Details
**Location**: `/home/enrico/data/ERCOT_data/bess_analysis/daily/`
- `bess_daily_revenues.parquet`
- Granular daily revenue breakdown by resource and revenue stream

### 2. Database Export Format
**Location**: `/home/enrico/data/ERCOT_data/bess_analysis/database_export/`
- `bess_daily_revenues.parquet`: Daily revenue records
- `bess_annual_leaderboard.parquet`: Annual rankings
- `metadata.json`: Processing metadata

### 3. Analysis Results
**Location**: `/tmp/` (temporary) and `/home/enrico/data/ERCOT_data/bess_analysis/`
- `python_bess_results.parquet`: Simplified test results
- Various CSV exports for specific date ranges

## Key Findings (2024 Sample)

Based on analysis of 10 BESS resources:

| Metric | Value |
|--------|-------|
| Total Revenue | $3,682,055 |
| Average Revenue/Resource | $368,205 |
| Top Performer | BATCAVE_BES1 ($2,761,779) |
| AS Revenue Share | 60-95% of total |
| DA Energy Share | 5-40% of total |

### Revenue Breakdown by Service
- **Ancillary Services**: Primary revenue driver (60-95%)
- **Day-Ahead Energy**: Supplementary revenue (5-40%)
- **Real-Time Energy**: Minor adjustments (±5%)

### Top 5 BESS Resources (2024)
1. **BATCAVE_BES1**: $2,761,779
   - DA: $880,435 (32%)
   - AS: $1,881,344 (68%)

2. **ANGLETON_UNIT1**: $272,024
   - DA: $69,006 (25%)
   - AS: $203,018 (75%)

3. **ALVIN_UNIT1**: $252,792
   - DA: $57,672 (23%)
   - AS: $195,120 (77%)

4. **AZURE_BESS1**: $177,589
   - DA: $6,810 (4%)
   - AS: $170,778 (96%)

5. **ANCHOR_BESS1**: $91,654
   - DA: $0 (0%)
   - AS: $91,654 (100%)

## Processing Performance

### Python Implementation
- Processing Time: ~2.8 seconds (10 resources)
- Memory Usage: ~10GB (full dataset)
- Suitable for: Detailed analysis, validation

### Rust Implementation (In Development)
- Target: <1 second for full dataset
- Parallel processing with Rayon
- Columnar operations with Polars

## Future Enhancements

1. **Cycling Analysis**: Track charge/discharge cycles and degradation
2. **State of Charge**: Reconstruct SOC from telemetered data
3. **Performance Metrics**: Capacity factor, round-trip efficiency
4. **Market Impact**: Price impact of BESS operations
5. **Forecast Integration**: Forward revenue projections

## Validation

Results are validated against:
- ERCOT settlement statements
- Public market reports
- Resource-specific performance data
- Industry benchmarks

## Contact

For questions about this methodology or access to the full dataset, please contact the project maintainers.

---
*Last Updated: August 2024*
*Version: 1.0*
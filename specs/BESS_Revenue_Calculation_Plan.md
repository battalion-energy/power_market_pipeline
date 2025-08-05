# BESS Revenue Calculation Plan

## Overview

This document outlines the comprehensive plan for calculating Battery Energy Storage System (BESS) revenues using ERCOT's 60-day disclosure data and real-time pricing data. The calculation covers three main revenue streams:

1. **Day-Ahead Energy Market Revenue**
2. **Real-Time Energy Market Revenue** 
3. **Ancillary Services Revenue** (RegUp, RegDown, RRS variants, ECRS, NonSpin)

## Quick Reference: Data Sources

### Operations Data (60-Day Lag)
| Data Type | Directory | File Pattern | Contains |
|-----------|-----------|--------------|----------|
| DAM Awards | `/60-Day_DAM_Disclosure_Reports/csv/` | `60d_DAM_Gen_Resource_Data-DD-MMM-YY.csv` | Energy & AS awards |
| RT Dispatch | `/60-Day_SCED_Disclosure_Reports/csv/` | `60d_SCED_Gen_Resource_Data-DD-MMM-YY.csv` | 5-min dispatch |
| SOC Status | `/60-Day_COP_Adjustment_Period_Snapshot/csv/` | `60d_COP_Adjustment_Period_Snapshot-DD-MMM-YY.csv` | State of charge |

### Price Data (Real-Time)
| Price Type | Directory | File Pattern | Interval |
|------------|-----------|--------------|----------|
| DAM SPP | `/DAM_Settlement_Point_Prices/csv/` | `cdr.*.YYYYMMDD.*.DAMSPNP4190.csv` | Hourly |
| RT SPP | `/Settlement_Point_Prices_at_Resource_Nodes,_Hubs_and_Load_Zones/csv/` | `cdr.*.YYYYMMDD.*.SPPHLZNP6905_*.csv` | 15-min |
| AS MCPC | `/DAM_Clearing_Prices_for_Capacity/csv/` | `cdr.*.YYYYMMDD.*.DAMCPCNP4188.csv` | Hourly |

**Note**: All directories are under `/Users/enrico/data/ERCOT_data/`

## Data Requirements

### 1. Resource Identification
- **60-Day Disclosure Files**: Identify BESS resources
  - **Primary Method**: Use `Resource Type = "PWRSTR"` in `60d_DAM_Gen_Resource_Data` files
  - This is the definitive way to identify BESS resources in ERCOT
  - Found ~195-199 BESS resources using this method (vs ~144 using name patterns)
  - Note: SOC columns may not always be present for all BESS resources

### 2. Settlement Point Mapping
- **Settlement Point Information**: Already embedded in DAM data
  - The `60d_DAM_Gen_Resource_Data` files contain `Settlement Point Name` column
  - No separate mapping file needed - direct relationship available
  - Example mappings: ADL_BESS1 → ADL_RN, ANCHOR_BESS1 → ANCHOR_ALL
  - Also includes `Energy Settlement Point Price` for direct revenue calculation

### 3. Price Data

**IMPORTANT**: Use Settlement Point Prices (SPP) for all revenue calculations, NOT Locational Marginal Prices (LMP). SPPs include scarcity adders which affect actual payments to resources.

#### Day-Ahead Energy Prices
- **Source**: `/Users/enrico/data/ERCOT_data/DAM_Settlement_Point_Prices/csv/`
- **File Pattern**: `cdr.00012331.0000000000000000.YYYYMMDD.HHMMSS.DAMSPNP4190.csv`
- **Key Columns**: `DeliveryDate`, `HourEnding`, `SettlementPoint`, `SettlementPointPrice`
- **Usage**: DAM energy revenue calculation

#### Real-Time Energy Prices (15-minute intervals)
- **Source**: `/Users/enrico/data/ERCOT_data/Settlement_Point_Prices_at_Resource_Nodes,_Hubs_and_Load_Zones/csv/`
- **File Pattern**: `cdr.00012301.0000000000000000.YYYYMMDD.HHMMSS.SPPHLZNP6905_YYYYMMDD_HHMM.csv`
- **Key Columns**: `DeliveryDate`, `DeliveryHour`, `DeliveryInterval`, `SettlementPointName`, `SettlementPointPrice`
- **Usage**: RT energy revenue calculation (4 intervals per hour)

#### Ancillary Services Clearing Prices
- **Source**: `/Users/enrico/data/ERCOT_data/DAM_Clearing_Prices_for_Capacity/csv/`
- **File Pattern**: `cdr.00012329.0000000000000000.YYYYMMDD.HHMMSS.DAMCPCNP4188.csv`
- **Key Columns**: `DeliveryDate`, `HourEnding`, `AncillaryType`, `MCPC`
- **Usage**: AS capacity payment calculation

#### Settlement Point Mapping (Reference)
- **Source**: `/Users/enrico/data/ERCOT_data/Settlement_Points_List_and_Electrical_Buses_Mapping/`
- **File Pattern**: `Settlement_Points_MMDDYYYY_HHMMSS.csv`
- **Usage**: Map resources to correct settlement points (if needed)

### 4. BESS Operations Data (60-Day Disclosure)

#### DAM Awards and AS Awards
- **Source**: `/Users/enrico/data/ERCOT_data/60-Day_DAM_Disclosure_Reports/csv/`
- **File Pattern**: `60d_DAM_Gen_Resource_Data-DD-MMM-YY.csv`
- **Example**: `60d_DAM_Gen_Resource_Data-07-JAN-25.csv`
- **Key Columns**: 
  - `Resource Name`, `Resource Type` (PWRSTR for BESS)
  - `Awarded Quantity` (DAM energy award in MW)
  - `RegUp Awarded`, `RegDown Awarded`, `RRSFFR Awarded`, `ECRSSD Awarded`, `NonSpin Awarded`
  - `Settlement Point Name` (for price matching)
  - `HSL`, `LSL` (operating limits)

#### Real-Time Dispatch (5-minute SCED)
- **Source**: `/Users/enrico/data/ERCOT_data/60-Day_SCED_Disclosure_Reports/csv/`
- **File Pattern**: `60d_SCED_Gen_Resource_Data-DD-MMM-YY.csv`
- **Example**: `60d_SCED_Gen_Resource_Data-07-JAN-25.csv`
- **Key Columns**:
  - `SCED Time Stamp` (5-minute intervals)
  - `Base Point` (dispatch instruction in MW)
  - `Telemetered Net Output` (actual output)
  - `RegUp Resource Responsibility`, `RegDown Resource Responsibility`
  - `RegUp Deployed MW`, `RegDown Deployed MW`

#### State of Charge Management
- **Source**: `/Users/enrico/data/ERCOT_data/60-Day_COP_Adjustment_Period_Snapshot/csv/`
- **File Pattern**: `60d_COP_Adjustment_Period_Snapshot-DD-MMM-YY.csv`
- **Example**: `60d_COP_Adjustment_Period_Snapshot-07-JAN-25.csv`
- **Key Columns**:
  - `Telemetered Resource Status` (ON/OFF)
  - `Telemetered State Of Charge`
  - `HSL`, `LSL` (may vary by SOC)

## Revenue Calculation Methodology

### 1. Day-Ahead Energy Revenue

**Data Sources**:
- `60d_DAM_Gen_Resource_Data` - Awards and resource data
- `DAM_Settlement_Point_Prices` - Settlement prices

**Calculation Steps**:
1. Extract DAM energy awards from `Awarded Quantity` column in disclosure data
2. Load corresponding DAM SPP prices from `/DAM_Settlement_Point_Prices/csv/`
3. Match prices to resources using `Settlement Point Name` column
4. Calculate: `DAM_Energy_Revenue = Σ(Awarded_Quantity_MW × Settlement_Point_Price × 1 hour)`

**Key Columns**:
- Resource Name
- Delivery Date, Hour Ending
- Awarded Quantity (MW)
- Settlement Point Name
- Energy Settlement Point Price ($/MWh)

### 2. Real-Time Energy Revenue

**ERCOT Real-Time Settlement Structure**:
- **SCED Runs**: Every 5 minutes (Security Constrained Economic Dispatch)
- **Settlement Intervals**: 15-minute periods (average of three 5-minute SCED intervals)
- **Settlement Files**: 96 files per day (4 per hour × 24 hours)

**Data Sources**:
- `60d_SCED_Gen_Resource_Data` - Base points and telemetered output (5-minute granularity)
- `Settlement_Point_Prices_at_Resource_Nodes_YYYYMMDD_HHMM_HHMM.csv` - RT settlement prices (15-minute)

**Calculation Steps**:
1. For each 15-minute settlement interval:
   - Average the three 5-minute SCED base points
   - Calculate RT position: `RT_Position = Average_Base_Point - DAM_Award`
2. Match to 15-minute RT settlement point prices
3. Calculate revenue/cost:
   - If RT_Position > 0 (generation): `Revenue = RT_Position × RT_SPP × 0.25 hour`
   - If RT_Position < 0 (charging): `Cost = |RT_Position| × RT_SPP × 0.25 hour`
4. Net RT Revenue = Generation Revenue - Charging Cost

**Key Columns**:
- SCED Time Stamp (5-minute intervals)
- Base Point (MW)
- Telemetered Net Output (MW)
- HSL (High Sustainable Limit)
- LSL (Low Sustainable Limit)
- Settlement Point Price ($/MWh)

### 3. Ancillary Services Revenue

**Data Sources**:
- `60d_DAM_Gen_Resource_Data` - AS awards (hourly)
- `60d_SCED_Gen_Resource_Data` - AS responsibilities and deployments (5-minute)
- `DAM_Clearing_Prices_for_Capacity` - AS clearing prices (hourly)

**Revenue Components**:
1. **Capacity Payments**: Payment for being available to provide AS
2. **Performance Payments**: Additional payments for actual deployment (not in 60-day disclosure)

**Services and Calculations**:

#### Regulation Up (RegUp)
- Award: `RegUp Awarded` column (MW)
- Responsibility: `RegUp Resource Responsibility` in SCED data
- Price: `RegUp MCPC` ($/MW per hour)
- Capacity Revenue: `RegUp_MW × MCPC × 1 hour`
- Deployment tracked via `RegUp Deployed MW` (5-minute)

#### Regulation Down (RegDown)
- Award: `RegDown Awarded` column
- Price: `RegDown MCPC` or market clearing price
- Revenue: `RegDown_MW × MCPC × Hours`

#### Responsive Reserve Service (RRS)
- Types: RRSPFR, RRSFFR, RRSUFR
- Awards: Respective columns in DAM data
- Price: `RRS MCPC`
- Revenue: `RRS_MW × MCPC × Hours`

#### ERCOT Contingency Reserve Service (ECRS)
- Award: `ECRSSD Awarded` column
- Price: `ECRS MCPC`
- Revenue: `ECRS_MW × MCPC × Hours`

#### Non-Spinning Reserve (NonSpin)
- Award: `NonSpin Awarded` column
- Price: `NonSpin MCPC`
- Revenue: `NonSpin_MW × MCPC × Hours`

## Implementation Steps

### Phase 1: Data Collection
1. Identify all BESS resources in the system
   - Read `/60-Day_DAM_Disclosure_Reports/csv/60d_DAM_Gen_Resource_Data-*.csv`
   - Filter for `Resource Type = 'PWRSTR'`
2. Extract settlement point mapping from DAM files
   - Use `Settlement Point Name` column (already in DAM Gen Resource Data)
3. Collect historical data
   - Disclosure data from `/60-Day_*_Disclosure_Reports/csv/`
   - Price data from respective SPP directories

### Phase 2: Data Processing
1. Parse and clean disclosure files
2. Join operational data with pricing data
3. Handle missing data and outliers

### Phase 3: Revenue Calculation
1. Calculate hourly DAM energy revenues
2. Calculate 15-minute RT energy revenues
3. Calculate hourly AS capacity revenues
4. Aggregate by resource and time period

### Phase 4: Validation
1. Cross-check total MW awards against resource capabilities
2. Verify SOC constraints are respected
3. Validate price ranges against historical norms

## Important Note on Data Timing

### 60-Day Disclosure Lag
- **Disclosure Data**: Published 60 days after the operating day
  - Example: Data for January 1, 2025 published on March 2, 2025
  - File named: `60d_DAM_Gen_Resource_Data-01-JAN-25.csv` (contains Jan 1 data)
  - Location: `/Users/enrico/data/ERCOT_data/60-Day_DAM_Disclosure_Reports/csv/`
  
- **Price Data**: Published in real-time or shortly after operating day
  - DAM SPP: Published after day-ahead market clears (~12:30 PM day before)
  - RT SPP: Published every 15 minutes during operating day
  - AS Prices: Published with DAM results
  
- **Matching**: Use the delivery date in both files to match:
  - Disclosure file dated "01-JAN-25" matches price files with DeliveryDate "01/01/2025"
  - No 60-day offset needed for price lookups

## Data Quality Considerations

### 1. Time Alignment
- DAM data: Hourly granularity
- RT data: 5-minute SCED intervals, 15-minute settlement
- Ensure proper time zone handling (CPT)

### 2. Date Format for Files
- **60-Day Disclosure Files**: Use format `DD-MMM-YY` (e.g., "07-JAN-25")
- Python implementation: `f"{date.day:02d}-{date.strftime('%b').upper()}-{date.strftime('%y')}"`
- Critical for file lookups as format must match exactly

### 2. Missing Data
- Some BESS may not participate in all markets
- Handle zeros vs. nulls appropriately
- Check for data gaps in disclosure files

### 3. Resource Status
- Verify resource is "ON" during awarded periods
- Check telemetered status for availability

## Expected Output

### Revenue Aggregation Hierarchy:

#### 1. **15-Minute Period Metrics** (Real-Time Only):
- RT Energy Revenue/Cost ($)
- RT Position (MW)
- RT Settlement Price ($/MWh)
- AS Deployment Status

#### 2. **Hourly Metrics**:
- DAM Energy Revenue ($)
- RT Energy Revenue Net (sum of four 15-min periods)
- AS Capacity Revenue by Service:
  - RegUp, RegDown, RRS (PFR/FFR/UFR), ECRS, NonSpin
- Total Hourly Revenue ($)
- Average Position (MW)
- Weighted Average Prices Captured

#### 3. **Daily Rollup**:
- Total DAM Energy Revenue
- Total RT Energy Revenue (Net)
- Total AS Revenue by Service
- Peak Hour Performance
- Capacity Factor (%)
- Revenue per MW Capacity

#### 4. **Monthly Rollup**:
- Total Revenue by Stream
- Average Daily Revenue
- Market Share of Total AS Awards
- Availability Factor
- Top Revenue Days Analysis

#### 5. **Annual Summary**:
- Total Annual Revenue
- Revenue Growth Trends
- Seasonal Patterns
- Market Strategy Evolution
- Comparative Performance vs Fleet

## File Processing Order

1. **Setup Phase**:
   - Scan all `/60-Day_DAM_Disclosure_Reports/csv/60d_DAM_Gen_Resource_Data-*.csv`
   - Identify BESS resources where `Resource Type = 'PWRSTR'`
   - Extract settlement points from `Settlement Point Name` column

2. **Daily Processing Loop** (for each operating day):
   
   **Step 2.1: Load Disclosure Data**
   - DAM file: `/60-Day_DAM_Disclosure_Reports/csv/60d_DAM_Gen_Resource_Data-DD-MMM-YY.csv`
   - SCED file: `/60-Day_SCED_Disclosure_Reports/csv/60d_SCED_Gen_Resource_Data-DD-MMM-YY.csv`
   - COP file: `/60-Day_COP_Adjustment_Period_Snapshot/csv/60d_COP_Adjustment_Period_Snapshot-DD-MMM-YY.csv`
   
   **Step 2.2: Load Price Data** (matching delivery date)
   - DAM SPP: `/DAM_Settlement_Point_Prices/csv/cdr.*.YYYYMMDD.*.DAMSPNP4190.csv`
   - RT SPP: `/Settlement_Point_Prices_at_Resource_Nodes,_Hubs_and_Load_Zones/csv/cdr.*.YYYYMMDD.*.SPPHLZNP6905_*.csv`
   - AS Prices: `/DAM_Clearing_Prices_for_Capacity/csv/cdr.*.YYYYMMDD.*.DAMCPCNP4188.csv`
   
   **Step 2.3: Calculate Revenues**
   - Match disclosure awards/dispatch to SPP prices by settlement point
   - Calculate DAM energy, RT energy, and AS revenues
   - Aggregate and store results

3. **Summary Phase**:
   - Aggregate across 90 days
   - Generate summary statistics
   - Create visualizations

## Validation Checks

1. **Revenue Reasonableness**:
   - Daily revenue should be positive (with rare exceptions)
   - Revenue per MW should be within market norms
   - AS revenue should not exceed energy revenue significantly

2. **Operational Consistency**:
   - SOC should remain within Min/Max bounds
   - Charging + Discharging should respect efficiency
   - Awards should not exceed HSL/LSL limits

3. **Data Completeness**:
   - All hours should have data
   - All awarded services should have prices
   - Resource names should be consistent

## Known Limitations

1. **60-Day Data Lag**: Cannot analyze most recent 60 days
2. **Performance Payments**: AS performance payments not included
3. **Make-Whole Payments**: Not captured in these files
4. **Uplift Charges**: Not included in basic settlement data
5. **Efficiency Losses**: Must be estimated (~85-95% round-trip)

## Implementation Insights

### Key Findings from Analysis

1. **Market Participation Patterns**:
   - ~90% of BESS focus exclusively on AS markets (no DAM energy awards)
   - Only top performers like ANEM_ESS_BESS1 actively arbitrage energy
   - ECRS is popular with high MW commitments but needs price data for revenue
   - Market shift noted: Recent batteries moving to energy arbitrage as AS markets saturate

### Historical BESS Deployment Timeline

1. **Early Deployments (2019-2021)**:
   - First utility-scale BESS appeared around 2019
   - Initially focused on frequency regulation (RegUp/RegDown)
   - Small capacity systems (10-50 MW)

2. **Growth Phase (2022-2023)**:
   - Rapid deployment of 100+ MW systems
   - Introduction of 4-hour duration standard
   - Shift to co-optimized energy + AS strategies

3. **Current Market (2024-2025)**:
   - 195-199 active BESS resources
   - Increasing focus on energy arbitrage
   - AS market saturation driving strategy changes

2. **Revenue Magnitudes**:
   - Top performer (ANEM_ESS_BESS1): ~$22,859/day total revenue
   - Energy arbitrage: Up to ~$21,284/day for active participants
   - RegUp capacity: $200-1,500/day depending on MW awards
   - Most BESS earn primarily from AS capacity payments

3. **Data Processing Considerations**:
   - Some columns (like MCPC prices) may be missing - need fallback to market clearing prices
   - Resource names remain consistent across files
   - QSE (Qualified Scheduling Entity) information available for ownership tracking

4. **Price Volatility**:
   - DAM prices ranged from ~$19-41/MWh in January 2025
   - High price days present arbitrage opportunities for active BESS
   - AS prices more stable than energy prices

## Comprehensive Historical Analysis Plan

### Data Availability:
- **60-Day Disclosure**: Available from ~2019 onwards
- **Price Data**: Historical SPP data available in ERCOT directories
  - DAM SPP: `/DAM_Settlement_Point_Prices/csv/`
  - RT SPP: `/Settlement_Point_Prices_at_Resource_Nodes,_Hubs_and_Load_Zones/csv/`
  - AS Prices: `/DAM_Clearing_Prices_for_Capacity/csv/`
- **BESS Identification**: Use Resource Type = "PWRSTR" across all historical files

### Processing Strategy:
1. **Batch Processing**: Process data in monthly chunks to manage memory
2. **Parallel Processing**: Use Rust processor for high-performance calculation
3. **Incremental Updates**: Store results in database for efficient queries
4. **Data Validation**: Cross-check totals with ERCOT market reports

### Output Tables:
1. **bess_revenue_15min**: Real-time revenue at settlement interval
2. **bess_revenue_hourly**: Hourly aggregated revenues
3. **bess_revenue_daily**: Daily summaries with all revenue streams
4. **bess_revenue_monthly**: Monthly rollups with statistics
5. **bess_revenue_annual**: Yearly summaries with trends

### Performance Considerations:
- Use columnar storage (Parquet) for historical data
- Implement caching for frequently accessed price data
- Optimize joins between operational and price data
- Consider materialized views for common aggregations
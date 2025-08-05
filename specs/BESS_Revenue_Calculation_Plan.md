# BESS Revenue Calculation Plan

## Overview

This document outlines the comprehensive plan for calculating Battery Energy Storage System (BESS) revenues using ERCOT's 60-day disclosure data and real-time pricing data. The calculation covers three main revenue streams:

1. **Day-Ahead Energy Market Revenue**
2. **Real-Time Energy Market Revenue** 
3. **Ancillary Services Revenue** (RegUp, RegDown, RRS variants, ECRS, NonSpin)

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
- **DAM Settlement Point Prices**: Day-ahead energy prices
- **RTM Settlement Point Prices**: Real-time energy prices (15-minute intervals)
- **DAM Clearing Prices for Capacity**: Ancillary services clearing prices

### 4. BESS Operations Data
- **60d_DAM_Gen_Resource_Data**: DAM awards and AS awards
- **60d_SCED_Gen_Resource_Data**: Real-time dispatch and AS deployment
- **60d_COP_Adjustment_Period_Snapshot**: SOC management data

## Revenue Calculation Methodology

### 1. Day-Ahead Energy Revenue

**Data Sources**:
- `60d_DAM_Gen_Resource_Data` - Awards and resource data
- `DAM_Settlement_Point_Prices` - Settlement prices

**Calculation Steps**:
1. Extract DAM energy awards from `Awarded Quantity` column
2. Match to settlement point prices using `Settlement Point Name`
3. Calculate: `DAM_Energy_Revenue = Σ(Awarded_Quantity_MW × Settlement_Point_Price × 1 hour)`

**Key Columns**:
- Resource Name
- Delivery Date, Hour Ending
- Awarded Quantity (MW)
- Settlement Point Name
- Energy Settlement Point Price ($/MWh)

### 2. Real-Time Energy Revenue

**Data Sources**:
- `60d_SCED_Gen_Resource_Data` - Base points and telemetered output
- `Settlement_Point_Prices_at_Resource_Nodes` - RT settlement prices

**Calculation Steps**:
1. Calculate RT deviations: `RT_Deviation = Base_Point - DAM_Award`
2. Match to 15-minute RT prices
3. Calculate: `RT_Energy_Revenue = Σ(RT_Deviation_MW × RT_Settlement_Price × 0.25 hour)`

**Key Columns**:
- SCED Time Stamp
- Base Point (MW)
- Telemetered Net Output (MW)
- Settlement Point Price ($/MWh)

### 3. Ancillary Services Revenue

**Data Sources**:
- `60d_DAM_Gen_Resource_Data` - AS awards
- `60d_SCED_Gen_Resource_Data` - AS responsibilities
- `DAM_Clearing_Prices_for_Capacity` - AS clearing prices

**Services and Calculations**:

#### Regulation Up (RegUp)
- Award: `RegUp Awarded` column
- Price: `RegUp MCPC` or market clearing price
- Revenue: `RegUp_MW × MCPC × Hours`

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
2. Create resource-to-settlement-point mapping
3. Collect 90 days of historical data (considering 60-day lag)

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

### Summary Metrics per BESS:
1. **Daily Revenue Breakdown**:
   - DAM Energy Revenue ($)
   - RT Energy Revenue ($)
   - AS Revenue by Service ($)
   - Total Daily Revenue ($)

2. **Operational Metrics**:
   - Average DAM Award (MW)
   - RT Deviation Statistics
   - AS Award Percentages
   - Capacity Factor

3. **Price Capture Metrics**:
   - Average DAM Price Captured ($/MWh)
   - Average RT Price Captured ($/MWh)
   - Price Volatility Captured

## File Processing Order

1. **Setup Phase**:
   - Load Settlement Points mapping
   - Identify BESS resources

2. **Daily Processing Loop**:
   - Load DAM disclosure files for date
   - Load corresponding DAM prices
   - Calculate DAM revenues
   - Load SCED files for date
   - Load RT prices (96 files per day)
   - Calculate RT revenues
   - Load AS clearing prices
   - Calculate AS revenues
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
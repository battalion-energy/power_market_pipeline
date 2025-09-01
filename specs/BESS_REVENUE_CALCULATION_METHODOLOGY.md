# BESS Revenue Calculation Methodology for ERCOT

## Overview
This document defines the exact methodology for calculating BESS revenues in ERCOT, based on the Modo Energy approach. BESS operate as both generation and load resources, and we must combine revenue/cost data from both to determine net revenue.

## 1. Data Sources and File Locations

### 1.1 Resource Identification
- **BESS Resources**: Filter for `ResourceType == 'PWRSTR'` (Power Storage)
- **Location**: `/home/enrico/data/ERCOT_data/rollup_files/DAM_Gen_Resources/{year}.parquet`

### 1.2 Settlement Point Mapping
- **File**: `/home/enrico/data/ERCOT_data/Settlement_Points_List_and_Electrical_Buses_Mapping/latest_mapping/SP_List_EB_Mapping/gen_node_map.csv`
- **Columns**: `RESOURCE_NODE` (settlement point), `UNIT_SUBSTATION`, `UNIT_NAME`

## 2. Revenue Calculation Components

### 2.1 Day-Ahead Energy Market Revenue

#### 2.1.1 Discharge Revenue (Generation)
```
DA_Discharge_Revenue = Σ(AwardedQuantity_MW × DA_SettlementPointPrice)
```

**Data Sources:**
- **Awards**: `/home/enrico/data/ERCOT_data/rollup_files/DAM_Gen_Resources/{year}.parquet`
  - Column: `AwardedQuantity` (MW)
  - Filter: `ResourceName == '{BESS_NAME}' AND ResourceType == 'PWRSTR'`
- **Prices**: `/home/enrico/data/ERCOT_data/rollup_files/flattened/DA_prices_{year}.parquet`
  - Column: Use settlement point column (e.g., `BATCAVE_RN`) or `HB_BUSAVG` as fallback
  - Match by: `DeliveryDate` → `datetime`

#### 2.1.2 Charging Cost (Load)
**IMPORTANT**: Load resources cannot bid for day-ahead energy awards in ERCOT directly.

**Approach per Modo Energy:**
- Negative day-ahead energy revenues are attributed to **energy bids at the settlement point**
- Only consider bids from same QSE/DME as the resource
- Use Energy Bid Awards data at settlement point level

**Data Sources:**
- **Energy Bids**: `/home/enrico/data/ERCOT_data/60_Day_DAM_Disclosure/{year}/DAM_EnergyBidAwards/*.csv`
  - Column: `EnergyBidAwardMW` (negative values = charging)
  - Filter: `SettlementPoint == '{RESOURCE_SETTLEMENT_POINT}'`
  - **Note**: This is settlement-point level, not resource-specific
- **Prices**: Same as discharge

```
DA_Charge_Cost = Σ(|NegativeEnergyBidAward_MW| × DA_SettlementPointPrice)
```

### 2.2 Real-Time Energy Market Revenue

#### 2.2.1 RT Net Position Calculation
Per Modo Energy: "Real-time energy revenues are calculated using the net physical position between both load and generation resources"

```
RT_Imbalance = (RT_Physical_Position - DA_Position) × RT_Price
```

#### 2.2.2 Discharge Component (Generation)
**Data Sources:**
- **SCED Dispatch**: `/home/enrico/data/ERCOT_data/rollup_files/SCED_Gen_Resources/{year}.parquet`
  - Column: `BasePoint` (MW) - this is the dispatch instruction
  - Filter: `ResourceName == '{BESS_NAME}'`
- **Metered Energy**: Use settlement-metered net energy if available
- **Prices**: `/home/enrico/data/ERCOT_data/rollup_files/flattened/RT_prices_{year}.parquet`
  - Column: Settlement point LMP (includes SCED LMP + price adders)
  - Time: 5-minute or 15-minute intervals

```
RT_Discharge_Revenue = Σ(BasePoint_MW × 0.25 × RT_SettlementPointPrice)
```
*Note: × 0.25 converts MW to MWh for 15-minute interval*

#### 2.2.3 Charging Component (Load)
**Data Sources:**
- **SCED Load**: `/home/enrico/data/ERCOT_data/rollup_files/SCED_Load_Resources/{year}.parquet`
  - Column: `BasePoint` (MW) - telemetered net output
  - Filter: Load resource associated with BESS
- **Prices**: Same RT prices as discharge

```
RT_Charge_Cost = Σ(LoadBasePoint_MW × 0.25 × RT_SettlementPointPrice)
```

### 2.3 Ancillary Services Revenue

Per Modo Energy: "Based on responsibility (MW) reported to SCED multiplied by hourly clearing price"

**Data Sources:**
- **AS Responsibility**: `/home/enrico/data/ERCOT_data/rollup_files/DAM_Gen_Resources/{year}.parquet`
  - Columns: `RegUpAwarded`, `RegDownAwarded`, `RRSAwarded`, `NonSpinAwarded`, `ECRSAwarded`
  - Note: Use SCED responsibility if available, otherwise DAM awards
- **AS Prices**: `/home/enrico/data/ERCOT_data/rollup_files/flattened/AS_prices_{year}.parquet`
  - Columns: `REGUP`, `REGDN`, `RRS`, `NSPIN`, `ECRS`

```
AS_Revenue = Σ(AS_Responsibility_MW × AS_HourlyPrice)
```

## 3. Complete Revenue Calculation

```
Total_BESS_Revenue = 
    DA_Discharge_Revenue 
    - DA_Charge_Cost 
    + RT_Discharge_Revenue 
    - RT_Charge_Cost 
    + AS_Revenue
```

## 4. Implementation Status

| Component | Data Available | Implementation Status | Issues |
|-----------|---------------|----------------------|---------|
| **DA Discharge** | ✅ Yes - DAM_Gen_Resources | ✅ Implemented | Working correctly |
| **DA Charging** | ⚠️ Partial - Energy Bid Awards | ❌ Not properly implemented | Need to map settlement point bids to resources |
| **RT Discharge** | ✅ Yes - SCED_Gen_Resources | ❌ Not implemented | Need to add |
| **RT Charging** | ✅ Yes - SCED_Load_Resources | ❌ Not implemented | Need to add |
| **AS Revenue** | ✅ Yes - DAM_Gen_Resources | ✅ Implemented | Working correctly |

## 5. Current Issues and Gaps

### 5.1 DA Charging Data Challenge
- Energy Bid Awards is at **settlement point level**, not resource-specific
- Multiple resources may share same settlement point
- Need to identify which bids belong to our BESS vs other resources at same point
- **Current workaround**: Inferring charge based on discharge × efficiency

### 5.2 Resource Naming Mismatch
- Gen Resource: `BATCAVE_BES1`
- Load Resource: Might be `BATCAVE_BESL1` or similar
- Need mapping between Gen and Load resource names

### 5.3 QSE/DME Filtering
- Energy Bid Awards should be filtered by QSE/DME
- We don't currently have QSE/DME mapping for resources

## 6. Next Steps

1. **Find Load Resource Names**: Map each Gen BESS to its corresponding Load resource
2. **Implement RT Calculations**: Add SCED-based RT revenue/cost calculations
3. **Fix DA Charging**: Properly attribute Energy Bid Awards to specific BESS
4. **Add QSE/DME Filtering**: Ensure we only count bids from same entity as resource

## 7. Validation Checks

- **Energy Balance**: Total charge MWh should be ~1.18x discharge MWh (85% efficiency)
- **Price Reasonableness**: Revenue/Cost ratio should be 1.2-2x for profitable arbitrage
- **AS Dependency**: Most BESS revenue typically from AS (60-90%)
- **Capacity Limits**: No single hour dispatch should exceed nameplate capacity

## References
- ERCOT Protocol 4.6.2.2 (Day-Ahead Energy Resource Revenues)
- ERCOT Protocol 6.6.3.1 (Real-Time Energy Imbalance Payment)
- ERCOT Protocol 4.6.4.1 (Day-Ahead Ancillary Service Revenues)
- Modo Energy BESS Index Methodology
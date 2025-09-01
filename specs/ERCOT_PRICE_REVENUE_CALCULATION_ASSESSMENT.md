# ERCOT BESS Price and Revenue Calculation Assessment

## Executive Summary

After reviewing the existing documentation and methodology, I've identified **critical issues** with the current BESS revenue calculations, particularly for Real-Time (RT) energy revenue. The current approach appears to be **missing key components** and **misunderstanding** how ERCOT settles BESS operations.

## Critical Issues Identified

### 1. ❌ **RT Energy Revenue Calculation is Fundamentally Wrong**

**Current Implementation Problems:**
- The existing documentation suggests using `BasePoint × RT_Price` directly
- This is **incorrect** - RT settlement is based on **imbalance** from DA position
- Missing the critical formula: `RT_Revenue = (RT_Position - DA_Position) × RT_Price`

**What Actually Happens in ERCOT:**
```
RT Settlement = (Metered Energy - DA Award) × RT LMP
```

This means if a BESS:
- Has DA Award: 50 MW
- Actually delivers (BasePoint/Metered): 60 MW
- RT Price: $100/MWh
- **RT Revenue = (60 - 50) × $100 = $1,000** (NOT 60 × $100 = $6,000)

### 2. ⚠️ **DA Charging Cost Attribution is Problematic**

**Current Issue:**
- Energy Bid Awards are at **settlement point level**, not resource-specific
- Multiple resources can share the same settlement point
- Cannot directly attribute which bids belong to specific BESS

**The Reality:**
- BESS generation resources **cannot directly bid** for DA energy as load
- Charging happens through **Energy Bids** at the settlement point
- Need QSE/DME filtering to identify relevant bids

### 3. ❌ **Missing Load Resource Integration**

**Problem:**
- Most analyses only look at Generation Resources
- BESS have **two separate resources**:
  - Generation Resource (e.g., `BATCAVE_BES1`)
  - Load Resource (e.g., `BATCAVE_LD1`)
- **Both must be analyzed together** for complete picture

### 4. ⚠️ **Offer Curves Not Being Used**

**Current Gap:**
- SCED uses offer curves to determine dispatch
- These curves show the BESS's willingness to charge/discharge at different prices
- Located in `60d_SCED_Gen_Resource_Data` as `Energy Offer Curve`
- **Not currently incorporated** in revenue calculations

## Correct BESS Revenue Calculation Methodology

### Day-Ahead Market Revenue

#### DA Energy Revenue (Discharge)
```python
# From DAM_Gen_Resources
DA_Discharge_Revenue = Σ(AwardedQuantity_MW × DA_SPP)
```
✅ This part appears correct in current implementation

#### DA Energy Cost (Charging)
```python
# From DAM_EnergyBidAwards at settlement point level
# Filter by QSE/DME of the BESS
DA_Charge_Cost = Σ(|EnergyBidAward_MW| × DA_SPP)
# where EnergyBidAward < 0 indicates purchase/charging
```
⚠️ Needs proper QSE/DME filtering

### Real-Time Market Revenue

#### RT Energy Settlement (CRITICAL FIX NEEDED)
```python
# For Generation Resource
RT_Gen_Imbalance = (Metered_Gen_Output - DA_Gen_Award)
RT_Gen_Revenue = RT_Gen_Imbalance × RT_SPP

# For Load Resource  
RT_Load_Imbalance = (Metered_Load_Consumption - DA_Load_Schedule)
RT_Load_Cost = RT_Load_Imbalance × RT_SPP

# Net RT Revenue
RT_Net_Revenue = RT_Gen_Revenue - RT_Load_Cost
```

**Key Points:**
- RT is **imbalance settlement** only
- Must track **both** Gen and Load resources
- BasePoint is dispatch instruction, not necessarily metered output
- Use metered values when available, BasePoint as proxy otherwise

### Ancillary Services Revenue
```python
AS_Revenue = Σ(AS_Responsibility_MW × AS_MCPC)
```
✅ This appears correct in current implementation

## Data Flow for Correct Implementation

### 1. Day-Ahead Market
```
DAM_Gen_Resources → AwardedQuantity → × DA_SPP → DA Revenue
DAM_EnergyBidAwards → Filter by QSE → × DA_SPP → DA Cost
```

### 2. Real-Time Market
```
SCED_Gen_Resources → BasePoint/Metered → - DA_Award → × RT_SPP → RT Revenue
SCED_Load_Resources → BasePoint/Metered → - DA_Schedule → × RT_SPP → RT Cost
```

### 3. Settlement Point Mapping
```
Resource_Node → Settlement_Point → Price_Node
Use: Settlement_Points_List_and_Electrical_Buses_Mapping
```

## Validation Checks

### 1. Energy Balance Check
```python
Total_Charge_MWh = DA_Charge + RT_Load_Consumption
Total_Discharge_MWh = DA_Discharge + RT_Gen_Output
Efficiency_Check = Total_Discharge_MWh / Total_Charge_MWh
# Should be ~0.85-0.90 for typical BESS
```

### 2. Revenue Reasonableness
```python
Arbitrage_Ratio = Total_Revenue / Total_Cost
# Should be 1.2-2.0 for profitable operation
# If > 3.0, likely calculation error
```

### 3. Capacity Constraints
```python
Max_Dispatch = max(BasePoint across all intervals)
# Should not exceed nameplate capacity
```

## Recommended Implementation Steps

### Phase 1: Fix RT Calculation
1. Implement imbalance-based RT settlement
2. Include both Gen and Load resources
3. Use metered data when available

### Phase 2: Improve DA Charging
1. Implement QSE/DME filtering for Energy Bid Awards
2. Map settlement point bids to specific resources
3. Validate against known charging patterns

### Phase 3: Add Offer Curve Analysis
1. Extract offer curves from SCED data
2. Validate dispatch against offer prices
3. Use for revenue verification

### Phase 4: Complete Integration
1. Combine Gen and Load resource data
2. Implement full energy balance tracking
3. Add comprehensive validation checks

## Key Files and Data Sources

### Essential Files for Correct Calculation
```
# Day-Ahead
/rollup_files/DAM_Gen_Resources/{year}.parquet         # DA Awards
/60_Day_DAM_Disclosure/DAM_EnergyBidAwards/*.csv      # DA Charging

# Real-Time  
/rollup_files/SCED_Gen_Resources/{year}.parquet       # RT Gen Dispatch
/rollup_files/SCED_Load_Resources/{year}.parquet      # RT Load Dispatch

# Prices
/rollup_files/DA_prices/{year}.parquet                # DA SPP
/rollup_files/RT_prices/{year}.parquet                # RT SPP (15-min)

# Ancillary Services
/rollup_files/AS_prices/{year}.parquet                # AS MCPC
```

### Settlement Point Mapping
```
/Settlement_Points_List_and_Electrical_Buses_Mapping/
  └── Settlement_Points_YYYYMMDD_HHMMSS.csv
```

## Conclusion

The current BESS revenue calculation has **fundamental flaws**, particularly in:

1. **RT Energy Settlement** - Using wrong formula (dispatch × price instead of imbalance × price)
2. **Load Resource Omission** - Missing half of the BESS operation data
3. **DA Charging Attribution** - Not properly filtering settlement point bids

These issues likely lead to **significant overestimation** of RT revenues and **incomplete** charging cost accounting.

## Your Specific Questions Answered

### Q: Should RT energy revenue look at DA settlement point and energy bids to compute cleared amount and price for base point?

**A: Partially correct, but missing key concept:**
- Yes, you need DA cleared amounts (awards)
- But RT revenue is **not** BasePoint × Price
- RT revenue is **(Actual/BasePoint - DA_Award) × RT_Price**
- This is **imbalance settlement**, not full energy payment

### Q: Can you do this for RT settlement points and offer curves in 60-day disclosure?

**A: Yes, and this is critical:**
- RT settlement uses the same settlement points as DA
- Offer curves in SCED data show bid prices
- BasePoint should align with offer curve prices
- But revenue is still **imbalance-based**, not total output

### Key Takeaway
ERCOT RT market is an **imbalance market** - you only get paid/charged for the **difference** between your DA position and RT delivery, not for your total RT output. This is the most critical correction needed in the current implementation.
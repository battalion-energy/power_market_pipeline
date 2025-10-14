# BESS Charging Data Interpretations Analysis

## Background
ERCOT splits BESS into two resources (until Dec 2025):
- **Gen Resources**: Handle discharging (selling energy)
- **Load Resources**: Handle charging (buying energy)

## Critical Discovery
DAM Load Resource files DO NOT contain an "Awarded Quantity" or "Energy Bid Award" column for energy charging. They only contain:
- Max/Low Power Consumption limits (capacity constraints)
- Ancillary Service awards (RegUp, RegDown, RRS, etc.)

## 5 Proposed Interpretations of BESS Charging Data

### 1. **Implicit Charging from Gen Awards (MOST LIKELY)**
**Theory**: When DAM Gen Resource has 0 or negative awards, the BESS is charging
- **Evidence FOR**: 
  - Gen Resources DO have "Awarded Quantity" column
  - Some Gen awards are 0 during off-peak (potential charging hours)
  - Physical reality: If not discharging, battery must be charging or idle
- **Evidence AGAINST**: 
  - Gen awards are rarely negative in the data
  - Doesn't explain how much charging is happening
- **Implementation**: `charging_mw = max_consumption_limit when gen_award == 0`

### 2. **Charging in Energy Bid Files (PLAUSIBLE)**
**Theory**: DAM charging schedules are in separate Energy Bid files, not Load Resource files
- **Evidence FOR**:
  - There are "60d_DAM_Energy_Bids" files in the data
  - Load Resources might submit bids separately from awards
  - Would match how Gen Resources have both bids and awards
- **Evidence AGAINST**:
  - Need to verify these files exist and contain Load bids
  - More complex data integration required
- **Implementation**: Join Energy Bid files with Load Resource names

### 3. **Self-Scheduled Charging (UNLIKELY)**
**Theory**: BESS charging is self-scheduled and not awarded by ERCOT
- **Evidence FOR**:
  - Explains why no energy awards in Load Resource files
  - Max/Low Power limits define allowed charging range
  - BESS operators decide when to charge within limits
- **Evidence AGAINST**:
  - ERCOT typically clears all energy through market
  - Would be economically inefficient
- **Implementation**: Assume charging at economic price thresholds

### 4. **RT-Only Charging via SCED (PARTIALLY TRUE)**
**Theory**: All BESS charging happens in real-time through SCED, not DAM
- **Evidence FOR**:
  - SCED Gen Resources show negative BasePoints (charging)
  - Some BESS like FLOWERII have $0 DA revenue (RT-only operation)
  - SCED Load Resources exist but have no negative values
- **Evidence AGAINST**:
  - Doesn't explain DAM Load Resource purpose
  - Risk management suggests some DAM positioning
- **Implementation**: Use negative SCED Gen BasePoints as charging

### 5. **Charging Data in COP or Separate Reports (LEAST LIKELY)**
**Theory**: BESS charging schedules are in Current Operating Plans or other reports
- **Evidence FOR**:
  - COP files exist in the data
  - Could be operational data not in market files
- **Evidence AGAINST**:
  - COP is for real-time operations, not DAM
  - Would be inconsistent with other resource types
- **Implementation**: Parse COP files for BESS state-of-charge data

## Recommended Approach (Best Interpretation)

**PRIMARY: Interpretation #4 (RT-Only via SCED negative BasePoints)**
**SECONDARY: Check Interpretation #2 (Energy Bid files)**

### Rationale:
1. **Data Evidence**: SCED Gen Resources DO show negative BasePoints, which represents actual charging
2. **Market Reality**: Many BESS operate primarily in RT for arbitrage flexibility
3. **Explains Observations**: 
   - Why FLOWERII/SWOOSEII have $0 DA revenue
   - Why Load Resources have no energy awards
   - Why all revenue calculations work with just Gen Resources

### Implementation Strategy:
```python
# 1. For RT charging: Use SCED Gen negative BasePoints
rt_charging = sced_gen[sced_gen['BasePoint'] < 0]

# 2. For DA positioning: Check if Energy Bid files exist
# If not, assume minimal DA charging

# 3. For AS provision: Use Load Resource AS awards
# RegDown on Load = ability to reduce charging when needed

# 4. Calculate net position:
# Net Revenue = (Gen discharge revenue) - (Gen charging cost) + (AS revenue)
```

## Data Validation Needed
1. Check 60d_DAM_Energy_Bids files for Load Resource bids
2. Verify SCED Gen negative BasePoints match known charging patterns
3. Cross-check total MWh discharged vs charged for energy balance
4. Validate against known BESS operational patterns

## Conclusion
The most likely interpretation is that BESS charging in ERCOT is primarily captured through:
- **Real-time**: Negative BasePoints in SCED Gen Resources (confirmed in data)
- **Day-ahead**: Either implicit (when Gen award = 0) or in separate Energy Bid files
- **Load Resources**: Primarily for AS provision (RegDown when charging)

This explains why the current calculator works with just Gen Resources and why some BESS show $0 DA revenue.
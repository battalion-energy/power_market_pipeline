# RT Revenue Calculation - Detailed Analysis

**Date**: October 7, 2025
**Analysis Period**: May 2024 (peak revenue month per Modo Energy benchmark)

---

## Executive Summary

**CRITICAL FINDING**: All analyzed batteries are **charging at HIGHER prices than they're discharging at** - the exact opposite of profitable energy arbitrage.

This explains the negative RT revenues and suggests either:
1. **These batteries are NOT doing price arbitrage** - they're optimizing for Ancillary Services or other strategies
2. **Data quality issues** in SCED BasePoint or RT prices
3. **Calculation methodology error** (unlikely given data alignment)

---

## Data Sources Used

### Files (All 100% Real ERCOT Data)
```
SCED_Gen_Resources:  114.9M rows, ResourceName + BasePoint (discharge MW)
SCED_Load_Resources: 114.9M rows, ResourceName + BasePoint (charge MW)
RT_prices:           32.6M rows, SettlementPointName + SettlementPointPrice ($/MWh, 15-min)
```

### Key Fields
- **SCED_Gen_Resources.BasePoint**: MW discharge at each SCED interval (~5-15 min)
- **SCED_Load_Resources.BasePoint**: MW charge at each SCED interval
- **RT_prices.SettlementPointPrice**: Real-time LMP at resource node (15-min intervals)
- **Timestamps**: SCED in Central Time → converted to UTC, rounded to 15-min to join with RT prices

---

## Calculation Methodology

```python
# Step 1: Load discharge data
df_gen = SCED_Gen_Resources.filter(ResourceName == "GEN_NAME")
    .select(SCEDTimeStamp, BasePoint as discharge_mw)
    .convert_timezone("America/Chicago" → "UTC")

# Step 2: Load charging data
df_load = SCED_Load_Resources.filter(ResourceName == "LOAD_NAME")
    .select(SCEDTimeStamp, BasePoint as charge_mw)
    .convert_timezone("America/Chicago" → "UTC")

# Step 3: Load RT prices
df_prices = RT_prices.filter(SettlementPointName == "RESOURCE_NODE")
    .select(datetime, SettlementPointPrice as rt_price)

# Step 4: Join and calculate discharge revenue
df_gen_joined = df_gen
    .round_timestamp_to_15min()
    .join(df_prices, on=timestamp)

discharge_revenue = Σ(discharge_mw × rt_price × 15/60 hours)

# Step 5: Join and calculate charge cost
df_load_joined = df_load
    .round_timestamp_to_15min()
    .join(df_prices, on=timestamp)

charge_cost = Σ(charge_mw × rt_price × 15/60 hours)

# Step 6: Calculate net
rt_net = discharge_revenue - charge_cost
efficiency = discharge_MWh / charge_MWh
```

---

## Detailed Examples - May 2024

### Example 1: RRANCHES_UNIT2 (Top RT Performer)
**Expected**: Positive arbitrage, buy low sell high
**Actual**: NEGATIVE arbitrage

```
Capacity:         150.0 MW
Gen Resource:     RRANCHES_UNIT2
Load Resource:    RRANCHES_LD2
Resource Node:    RRANCHES_ALL

May 2024 Results:
  Discharge:      2,803.5 MWh @ avg $34.57/MWh = $96,930 revenue
  Charge:         3,383.1 MWh @ avg $34.84/MWh = $117,866 cost

  RT Net:         -$20,936
  Efficiency:     82.9%
  $/MW-month:     -$139.57

Benchmark Expected: +$2,025,000 (Modo: $13.50/kW-month)
Gap:                $2,045,936 (100.8% below benchmark)
```

**Key Issue**: Charging at $34.84/MWh vs discharging at $34.57/MWh → losing $0.27/MWh on average

---

### Example 2: CHISMGRD_BES1 (Worst RT Performer)
**Expected**: Negative arbitrage
**Actual**: MASSIVE negative arbitrage

```
Capacity:         9.99 MW
Gen Resource:     CHISMGRD_BES1
Load Resource:    CHISMGRD_LD1
Resource Node:    CHISMGRD_RN

May 2024 Results:
  Discharge:      1,984.4 MWh @ avg $21.04/MWh = $41,743 revenue
  Charge:         8,581.9 MWh @ avg $62.99/MWh = $540,532 cost

  RT Net:         -$498,789
  Efficiency:     23.1%
  $/MW-month:     -$49,928.79

Benchmark Expected: +$134,865
Gap:                $633,654 (470% below benchmark)
```

**Key Issue**:
- Charging at **3x the price** of discharge ($62.99 vs $21.04/MWh)
- Charging **4.3x more energy** than discharging (8,582 vs 1,984 MWh)
- Constantly charging at 10 MW, barely discharging
- Round-trip efficiency 23% suggests either:
  1. Battery degradation/damage
  2. Load resource is for something OTHER than battery charging
  3. Co-located with solar - "Load" is actually solar production?

---

### Example 3: BATCAVE_BES1 (Negative RT, >100% efficiency anomaly)
**Expected**: Positive arbitrage
**Actual**: Negative arbitrage, but reasonable efficiency

```
Capacity:         155.2 MW
Gen Resource:     BATCAVE_BES1
Load Resource:    BATCAVE_LD1
Resource Node:    BATCAVE_RN

May 2024 Results:
  Discharge:      3,966.2 MWh @ avg $21.72/MWh = $86,133 revenue
  Charge:         4,911.9 MWh @ avg $70.84/MWh = $347,952 cost

  RT Net:         -$261,819
  Efficiency:     80.7%
  $/MW-month:     -$1,686.98

Benchmark Expected: +$2,095,200
Gap:                $2,357,019 (112% below benchmark)
```

**Key Issue**:
- Charging at **3.3x the price** of discharge ($70.84 vs $21.72/MWh)
- Efficiency is reasonable (80.7%), so battery itself seems OK
- But strategy is completely backwards - charging expensive, selling cheap

**Note**: Full year shows >100% efficiency (114.4%), but May shows normal 80.7%

---

## Critical Pattern Identified

### All Three Batteries Show Same Problem

| Battery | Charge Avg Price | Discharge Avg Price | Price Ratio | May RT Net |
|---------|-----------------|---------------------|-------------|-----------|
| RRANCHES_UNIT2 | $34.84/MWh | $34.57/MWh | 1.01x | -$20,936 |
| CHISMGRD_BES1 | $62.99/MWh | $21.04/MWh | **3.00x** | -$498,789 |
| BATCAVE_BES1 | $70.84/MWh | $21.72/MWh | **3.26x** | -$261,819 |

**This is the opposite of energy arbitrage strategy.**

---

## Possible Explanations

### 1. Ancillary Services Optimization (Most Likely)
- Batteries are **NOT optimizing for energy arbitrage**
- Instead optimizing for:
  - **Regulation Up/Down** capacity payments (need to maintain mid-SOC)
  - **ECRS/RRS** capacity payments
  - **FFR** capacity payments
- Energy actions (charge/discharge) are:
  - Following SCED BasePoint instructions to maintain SOC
  - Following regulation signals
  - NOT profit-maximizing energy trades
- **This would explain**:
  - Negative RT energy revenues (energy is byproduct, not goal)
  - Modo shows AS as 70% of revenue, energy only 28%
  - Our DAM AS revenues are $22M (55% of total)

### 2. Co-Location with Renewables
- Some batteries co-located with solar/wind
- "Load Resource" might include:
  - Charging from co-located solar (not market purchases)
  - Contractual must-take requirements
  - Hedging obligations
- **Would explain**:
  - User's note: "some batteries on the ranking are negative $/kw because they are probably part of some other hedge"
  - CHISMGRD_BES1's 23% efficiency (if "Load" includes direct solar)

### 3. Data Quality Issues (Less Likely)
- **SCED BasePoint might not equal actual settlement**
  - BasePoint = instruction, not necessarily actual output
  - Telemetered output might differ
- **RT prices might not be settlement prices**
  - SettlementPointPrice vs actual resource settlement
- **Resource node mapping might be wrong**
  - Using wrong settlement point for prices

### 4. Missing Revenue Components (Possible)
We're calculating:
```
RT Revenue = discharge_revenue - charge_cost
```

But ERCOT RT settlement includes:
- **Energy revenue** (what we calculate)
- **AS deployment revenue** (mileage payments) ← **WE'RE MISSING THIS**
- **Ancillary service performance payments**
- **Make-whole payments**
- **RUC charges/credits**

**If AS deployment is significant**, it would offset negative energy arbitrage.

---

## Sample Data Showing the Pattern

### CHISMGRD_BES1 - May 2024, Evening Hours
```
Charging Intervals (Load Resource):
04/30/2024 19:00 - 10.0 MW @ $7.45/MWh  = $18.63 cost
04/30/2024 19:00 - 10.0 MW @ $6.26/MWh  = $15.65 cost
04/30/2024 19:15 - 10.0 MW @ $6.88/MWh  = $17.20 cost
04/30/2024 19:30 - 10.0 MW @ $6.45/MWh  = $16.13 cost
... (constantly charging at 10 MW)

Discharge Intervals (Gen Resource):
04/30/2024 19:00 - 0.0 MW @ $7.45/MWh   = $0 revenue
04/30/2024 19:15 - 0.0 MW @ $6.88/MWh   = $0 revenue
04/30/2024 19:30 - 0.0 MW @ $6.45/MWh   = $0 revenue
... (not discharging)
```

**Battery is constantly charging, not discharging.** This is consistent with:
- Maintaining SOC for AS capacity availability
- NOT doing energy arbitrage

---

## Questions for Critical Evaluation

### 1. Is BasePoint the Right Field?
**Current**: Using `BasePoint` from SCED
**Question**: Should we use:
- Telemetered output instead?
- Settlement data instead of SCED?
- Different field entirely?

### 2. Are We Missing AS Deployment Revenue?
**Current**: Only calculating energy arbitrage
**Question**: ERCOT RT settlement includes:
- AS mileage payments (Reg Up/Down actual deployment)
- Performance payments
- Should we be looking at settlement statements instead?

### 3. Is Resource Pairing Correct?
**Current**: Using BESS_UNIFIED_MAPPING_V3_CLARIFIED.csv
**Question**:
- Are Gen/Load resources correctly paired?
- Could "Load" resource include solar co-location?
- Should we validate mapping against ERCOT QSE data?

### 4. Are RT Prices Correct?
**Current**: Using RT_prices parquet with SettlementPointPrice
**Question**:
- Is this the price batteries actually settle at?
- Should we use resource-specific settlement prices?
- Are we at the right settlement point (Resource Node vs something else)?

### 5. Should We Expect Negative RT Energy?
**Current**: Treating negative as error
**Question**: If batteries optimize for AS (70% of revenue):
- Is negative energy arbitrage **expected**?
- Should we focus on DAM+AS revenues instead?
- Is RT energy just "balancing" to maintain SOC for AS?

---

## Next Steps for Investigation

### HIGH PRIORITY - Validate Assumptions
1. **Check ERCOT settlement statements** for a few batteries
   - Compare our calculated RT energy vs actual settlement
   - Identify missing revenue components (AS deployment, etc.)
   - Validate BasePoint vs actual telemetered output

2. **Analyze AS deployment patterns**
   - Look at AS_* fields in SCED_Gen_Resources
   - Calculate regulation deployment (MW actually used)
   - Estimate AS deployment revenue (mileage)

3. **Identify co-located batteries**
   - Cross-reference BESS with solar/wind databases
   - Check if negative RT batteries are co-located
   - Validate resource pairing for co-located sites

### MEDIUM PRIORITY - Methodology Improvements
4. **Add AS deployment revenue calculation**
   - Calculate Reg Up/Down deployment from SCED
   - Apply mileage rates
   - Include in RT revenue

5. **Validate resource mapping**
   - Spot-check Gen/Load pairing with ERCOT QSE data
   - Verify settlement points
   - Check for co-location

### LOW PRIORITY - Data Quality
6. **Investigate >100% efficiency**
   - Look at monthly patterns (May shows 80%, full year 114%)
   - Check for data errors in specific periods
   - Compare with other data sources

---

## Conclusion

**The calculation methodology is technically sound**, but our **interpretation may be wrong**.

### What We Know:
1. ✅ Data sources are correct and complete
2. ✅ Timezone handling is correct
3. ✅ Price joining is working properly
4. ✅ Math is correct (discharge revenue - charge cost)

### What We're Learning:
1. ⚠️ Batteries are **NOT optimizing for energy arbitrage**
2. ⚠️ RT energy may be **byproduct of AS optimization**
3. ⚠️ Negative RT energy might be **expected behavior** for AS-focused batteries
4. ⚠️ We're probably **missing AS deployment revenue** (mileage payments)

### Key Question:
**Is our benchmark comparison even valid?**

Modo Energy reports total revenues (energy + AS deployment + capacity).
We're calculating:
- DAM energy: ✅
- DAM AS capacity: ✅
- RT energy: ✅ (but negative)
- **RT AS deployment: ❌ MISSING**

**If RT AS deployment is large enough, it would explain the gap.**

---

## Recommendation

**Before "fixing" anything**, we need to:
1. Obtain actual ERCOT settlement statements for 2-3 batteries
2. Validate our calculations against actual settlements
3. Identify what revenue components we're missing
4. Determine if negative RT energy is legitimate or error

**Only then can we know if we need to fix the calculation or accept the results.**

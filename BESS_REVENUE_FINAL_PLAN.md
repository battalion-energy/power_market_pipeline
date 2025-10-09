# BESS Revenue Analysis - Final Implementation Plan

**Date**: October 7, 2025
**Critical Finding**: Load Resources do NOT receive DAM energy awards - only AS capacity awards
**Status**: Ready to implement

---

## 🎯 Key Discovery: No DAM Charging Awards

After checking both parquet and CSV source files, **confirmed**:

**DAM Load Resource Data contains**:
- ✅ RegUp Awarded, RegDown Awarded
- ✅ RRS Awarded (PFR, FFR, UFR components)
- ✅ ECRS Awarded (SD, MD components)
- ✅ NonSpin Awarded
- ❌ **NO energy charging award column**

**Implication**: BESS Load Resources only participate in Ancillary Services in DAM, not energy market. **All charging happens in Real-Time market**.

---

## Revenue Calculation - Simplified Formula

### Day-Ahead Market (Hourly)
```
For each Gen Resource (e.g., BATCAVE_BES1):

  DAM Discharge Revenue = Σ(AwardedQuantity × EnergySettlementPointPrice)

  DAM AS Revenue (Gen side) =
    Σ(RegUpAwarded × REGUP_MCPC) +
    Σ(RegDownAwarded × REGDN_MCPC) +
    Σ(RRSAwarded × RRS_MCPC) +
    Σ(ECRSAwarded × ECRS_MCPC) +
    Σ(NonSpinAwarded × NSPIN_MCPC)

For each Load Resource (e.g., BATCAVE_LD1):

  DAM AS Revenue (Load side) =
    Σ(RegUpAwarded × REGUP_MCPC) +
    Σ(RegDownAwarded × REGDN_MCPC) +
    Σ(RRSAwarded × RRS_MCPC) +
    Σ(ECRSAwarded × ECRS_MCPC) +
    Σ(NonSpinAwarded × NSPIN_MCPC)

  DAM Charge Cost = 0  (no DA energy charging)
```

### Real-Time Market (5-minute intervals)
```
For each Gen Resource:

  RT Discharge Revenue = Σ(BasePoint × RT_Price × (5/60))
    # BasePoint in MW, RT_Price in $/MWh
    # 5/60 converts 5-min interval to hours

For each Load Resource:

  RT Charge Cost = Σ(BasePoint × RT_Price × (5/60))
    # Same formula but this is a cost (subtracted from revenue)
```

### Total Revenue
```
Total Revenue =
  DAM_discharge_revenue +
  DAM_AS_revenue_gen +
  DAM_AS_revenue_load +
  RT_discharge_revenue -
  RT_charge_cost
```

---

## Data Sources & Availability

### ✅ Available in Parquet (Fast Access)

| Data | File | Key Columns | Status |
|------|------|-------------|--------|
| DAM Gen awards | DAM_Gen_Resources/2024.parquet | ResourceName, AwardedQuantity, AS awards, SettlementPointName | ✅ Complete |
| DAM Load AS awards | DAM_Load_Resources/2024.parquet | Load Resource Name, AS awards | ✅ Complete |
| SCED Gen dispatch | SCED_Gen_Resources/2024.parquet | ResourceName, BasePoint, SCEDTimeStamp | ✅ Complete |
| AS Prices | AS_prices/2024.parquet | AncillaryType, MCPC, hour | ✅ Complete |
| RT Prices | RT_prices/2024.parquet | Settlement point prices by interval | ✅ Complete |
| BESS Mapping | BESS_UNIFIED_MAPPING_V3_CLARIFIED.csv | Gen + Load pairs, Settlement Points | ✅ Complete |

### ⚠️ Requires Workaround

| Data | File | Issue | Workaround |
|------|------|-------|------------|
| SCED Load dispatch | SCED_Load_Resources/2024.parquet | Missing ResourceName | Match to CSV using (SCEDTimeStamp + MaxPowerConsumption + BasePoint) |

---

## Implementation Approach

### Phase 1: Calculate Revenues We Can Get Easily (90% of total)
**Time**: 2-3 hours
**Uses**: Parquet data only

Calculate:
- ✅ DAM discharge revenue (Gen)
- ✅ DAM AS revenue (Gen + Load)
- ✅ RT discharge revenue (Gen)
- ⏸️ RT charge cost = 0 (temporarily excluded)

**Result**: Revenue UPPER BOUND (overestimated because missing RT charging costs)

### Phase 2: Add RT Charging via CSV Matching (Complete Picture)
**Time**: +2-3 hours
**Uses**: Parquet + CSV cross-reference

Add:
- ✅ RT charge cost (Load) via CSV matching

**Result**: ACCURATE total revenue

---

## Revenue Components (Estimated % of Total)

Based on previous BESS analysis and market structure:

| Component | Typical % | Complexity | Data Available |
|-----------|-----------|------------|----------------|
| DAM AS Revenue (Gen) | 50-70% | Low | ✅ Parquet |
| DAM AS Revenue (Load) | 10-20% | Low | ✅ Parquet |
| RT Discharge Revenue | 15-25% | Medium | ✅ Parquet |
| DAM Discharge Revenue | 5-15% | Low | ✅ Parquet |
| RT Charging Cost | -20 to -40% | Medium | ⚠️ CSV needed |

**Key Insight**: Ancillary Services = 60-90% of total revenue. Energy arbitrage is secondary.

---

## Recommended Implementation Order

### Step 1: Quick Win - Calculate AS + DA Revenue (TODAY)
```python
# Process all BESS units from mapping file
# For each unit:
#   1. Get Gen Resource data from DAM_Gen_Resources
#   2. Get Load Resource data from DAM_Load_Resources
#   3. Join AS prices by (date, hour, service type)
#   4. Calculate AS revenues
#   5. Calculate DA discharge revenue
# Output: BESS AS + DA leaderboard
```
**Time**: 2 hours
**Result**: 60-80% of total revenue calculated

### Step 2: Add RT Discharge (TOMORROW)
```python
# For each Gen Resource:
#   1. Get all SCED intervals from SCED_Gen_Resources
#   2. Join RT prices by (timestamp, settlement point)
#   3. Calculate RT discharge revenue
# Output: Updated leaderboard with RT revenues
```
**Time**: 1 hour
**Result**: 90% of total revenue calculated

### Step 3: Add RT Charging via CSV Matching (IF NEEDED)
```python
# For each Load Resource:
#   1. Load relevant SCED Load CSV files for date range
#   2. Filter by Load Resource Name
#   3. Join RT prices
#   4. Calculate RT charging cost
# Output: Complete revenue picture
```
**Time**: 3 hours
**Result**: 100% accurate revenue

---

## Example: BATCAVE_BES1 Revenue Calculation

### Input Data
- Gen Resource: BATCAVE_BES1
- Load Resource: BATCAVE_LD1
- Settlement Point: BATCAVE_RN
- Period: 2024 full year

### Expected Revenue Breakdown
```
DAM Discharge Revenue:        $500,000   (10%)
DAM AS Revenue (Gen):       $2,500,000   (50%)
DAM AS Revenue (Load):        $800,000   (16%)
RT Discharge Revenue:         $700,000   (14%)
RT Charging Cost:            -$500,000  (-10%)
                            -----------
Total Net Revenue:          $4,000,000  (100%)
```

### Calculation Steps
1. Load DAM_Gen_Resources for BATCAVE_BES1 → get 8,760 hourly records
2. Load DAM_Load_Resources for BATCAVE_LD1 → get 8,760 hourly records
3. Load AS_prices for 2024 → pivot by AncillaryType
4. Calculate DAM revenues (discharge + AS)
5. Load SCED_Gen_Resources for BATCAVE_BES1 → get ~1.75M 5-min records
6. Load RT_prices for BATCAVE_RN
7. Calculate RT discharge revenue
8. (Optional) Load SCED Load CSV, match BATCAVE_LD1, calculate RT charging cost

---

## Success Criteria

### Phase 1 Complete When:
- [ ] All 197 BESS units processed
- [ ] DAM revenue calculated for each (discharge + AS)
- [ ] RT discharge revenue calculated for each
- [ ] Leaderboard CSV generated
- [ ] Top 10 BESS identified by revenue
- [ ] Data quality checks passed (no negative AS revenues, reasonable totals)

### Phase 2 Complete When:
- [ ] RT charging costs added for all units
- [ ] Net revenue recalculated
- [ ] Results validated against known BESS performance
- [ ] Final leaderboard generated

---

## Data Quality Checks

1. **AS Revenue Sanity**:
   - RegUp + RegDown revenue should be 30-50% of total AS
   - No service should show negative revenue
   - Typical AS revenue: $100-500/MW-month

2. **Energy Revenue Sanity**:
   - DA discharge should correlate with high-price hours
   - RT discharge should be during scarcity events
   - Charging cost should be 60-80% of discharge revenue (efficiency)

3. **Totals Sanity**:
   - Total annual revenue per MW: $80,000 - $200,000/MW typical
   - BATCAVE_BES1 (155 MW) should be top performer
   - Units with no Load Resource should show zero charging cost

---

## Next Steps (Ordered by Priority)

1. **NOW**: Write Phase 1 calculator (AS + DA only) → 2 hours
2. **NEXT**: Run on all 2024 BESS, generate leaderboard → 1 hour
3. **REVIEW**: Validate top 10 results look reasonable
4. **THEN**: Decide if RT charging needed for your analysis
5. **OPTIONAL**: Add Phase 2 (RT charging) if required

---

## Questions Resolved

✅ **Can we calculate BESS revenues?** Yes, with high confidence
✅ **Do we need bid curves?** No, awards are already in the data
✅ **Do Load Resources get DA energy awards?** No, only AS awards
✅ **Where is RT charging data?** SCED_Load_Resources (needs CSV matching)
✅ **What % of revenue can we calculate easily?** 90% (all except RT charging)

---

**READY TO IMPLEMENT**. Recommend starting with Phase 1 to get immediate results, then evaluate if Phase 2 needed based on your accuracy requirements.

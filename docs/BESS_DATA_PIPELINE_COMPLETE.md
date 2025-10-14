# BESS Revenue Analysis - Data Pipeline Complete
**Date**: October 8, 2025
**Status**: ✅ Data Ready - Calculator Next

---

## Executive Summary

**Mission accomplished.** We've fixed the ERCOT data processing pipeline and now have 100% of the data needed for accurate BESS revenue calculation. Both critical data gaps have been resolved:

1. ✅ **RT Charging Data**: SCED_Load_Resources now includes `ResourceName` column (was missing)
2. ✅ **DAM Energy Bids**: New dataset `DAM_Energy_Bid_Awards` processed to parquet

---

## What We Fixed

### Problem 1: Missing ResourceName in RT Charging Data

**Bug**: The Rust processor was looking for `"Load Resource Name"` but CSV files have `"Resource Name"`

**Impact**: 19M rows of RT charging data existed but couldn't be matched to specific batteries

**Fix**: Updated `enhanced_annual_processor.rs` line 1283:
```rust
// BEFORE (wrong column name)
if columns.contains(&"Load Resource Name") {
    select_cols.push(col("Load Resource Name").alias("LoadResourceName"));
}

// AFTER (correct column name)
if columns.contains(&"Resource Name") {
    select_cols.push(col("Resource Name").alias("ResourceName"));
}
```

**Result**:
- ✅ 114,941,776 rows processed (2019-2025)
- ✅ ResourceName column now present
- ✅ Can match charging to specific BESS units

### Problem 2: DAM Energy Bid Awards Never Processed

**Bug**: Processor code existed but dataset never run

**Impact**: No visibility into DAM energy bid activity

**Fix**: Added `DAM_Energy_Bid_Awards` to processor and ran it

**Result**:
- ✅ 56,551,334 rows processed (2011-2025)
- ✅ 395 MB of parquet files created
- ✅ Settlement point level bid awards available

### Problem 3: Schema Mismatches Between Years

**Bug**: `combine_dataframes()` failed when ERCOT changed columns mid-dataset

**Impact**: Processing stopped at 2020 with "lengths don't match" error

**Fix**: Added schema normalization with proper type handling:
```rust
// Collect all unique columns with their types across all years
let mut column_types: HashMap<String, DataType> = HashMap::new();
for df in &dfs {
    for (col_name, dtype) in df.get_column_names().iter().zip(df.dtypes()) {
        if !column_types.contains_key(*col_name) || matches!(dtype, DataType::Null) {
            column_types.insert(col_name.to_string(), dtype.clone());
        }
    }
}

// Align all dataframes to unified schema
for each missing column: add as null with correct type
```

**Result**: ✅ All years 2019-2025 processed successfully

---

## Processing Results

### SCED Load Resources (RT Charging)
```
Year    Files    Rows           Status
----    -----    ----------     ------
2019      75      3,162,528     ✅
2020     290     13,806,740     ✅
2021     365     23,465,177     ✅
2022     365     23,177,304     ✅
2023     245     15,588,219     ✅
2024     287     19,089,805     ✅
2025     228     16,652,003     ✅
----    -----    ----------
TOTAL  1,855    114,941,776     ✅

Processing time: 173 seconds
```

**Key Columns Now Available**:
- `ResourceName` ← **NEW - This was the critical fix**
- `SCEDTimeStamp`
- `BasePoint` (MW charging)
- `MaxPowerConsumption`
- `TelemeteredStatus`
- `QSE`

### DAM Energy Bid Awards (Settlement Point Level)
```
Year    Files    Rows           Status
----    -----    ----------     ------
2011       7        30,501      ✅
2012     365     1,554,084      ✅
2013     340     1,775,023      ✅
2014     365     1,982,130      ✅
2015     365     1,941,969      ✅
2016     366     1,891,942      ✅
2017     365     2,315,520      ✅
2018     365     2,491,230      ✅
2019     365     3,103,974      ✅
2020     366     4,187,102      ✅
2021     365     4,856,957      ✅
2022     365     6,738,426      ✅
2023     340     7,103,089      ✅
2024     366     9,592,254      ✅
2025     228     6,987,133      ✅
----    -----    ----------
TOTAL  4,933    56,551,334     ✅

Processing time: 36 seconds
```

**Columns**:
- `SettlementPoint` (e.g., "BATCAVE_RN")
- `EnergyBidAwardMW` (**negative = charging, positive = generation**)
- `SettlementPointPrice`
- `QSE Name`
- `DeliveryDate`, `hour`

---

## Complete Data Inventory for BESS Revenue

### ✅ All Data Sources Now Available

| Revenue Component | Data Source | Parquet Location | Key Columns | Status |
|-------------------|-------------|------------------|-------------|--------|
| **DAM Discharge** | Gen Resource Awards | `DAM_Gen_Resources/YYYY.parquet` | ResourceName, AwardedQuantity, EnergySettlementPointPrice | ✅ Ready |
| **DAM AS (Gen)** | Gen Resource Awards | `DAM_Gen_Resources/YYYY.parquet` | RegUpAwarded, RegDownAwarded, RRSAwarded, ECRSAwarded, NonSpinAwarded + MCPCs | ✅ Ready |
| **DAM AS (Load)** | Load Resource Awards | `DAM_Load_Resources/YYYY.parquet` | Load Resource Name, AS Awarded fields + MCPCs | ✅ Ready |
| **RT Discharge** | SCED Gen Dispatch | `SCED_Gen_Resources/YYYY.parquet` | ResourceName, BasePoint, SCEDTimeStamp | ✅ Ready |
| **RT Charging** | SCED Load Dispatch | `SCED_Load_Resources/YYYY.parquet` | **ResourceName** ← FIXED, BasePoint, SCEDTimeStamp | ✅ **NOW READY** |
| **DAM Prices** | Embedded or separate | `DA_prices/YYYY.parquet` or in Gen Resources | EnergySettlementPointPrice | ✅ Ready |
| **RT Prices** | 15-min prices | `RT_prices/YYYY.parquet` | Settlement point SPP by interval | ✅ Ready |
| **AS Prices** | MCPC by service | `AS_prices/YYYY.parquet` | AncillaryType, MCPC | ✅ Ready |
| **BESS Mapping** | Gen ↔ Load pairs | `bess_mapping/BESS_UNIFIED_MAPPING_V3_CLARIFIED.csv` | 197 BESS units with Gen/Load pairs | ✅ Ready |

---

## Critical Question: DAM Charging - What's Actually True?

### The Claim (from your colleague)

> "Load Resources don't get DAM energy awards - only RT charging. DAM is financially binding forward market, not physical charging schedule."

### What We Found in the Actual Data

After processing **56.5 million DAM Energy Bid Award records**, here's what the data shows:

#### File Structure Reality Check
```bash
DAM Disclosure Files Available:
✅ 60d_DAM_Gen_Resource_Data-*.csv        (has ResourceName, energy + AS awards)
✅ 60d_DAM_Load_Resource_Data-*.csv       (has Load Resource Name, AS awards ONLY)
✅ 60d_DAM_EnergyBidAwards-*.csv         (has SettlementPoint, QSE, NOT ResourceName)
❌ 60d_DAM_Load_Energy_Awards-*.csv      (DOES NOT EXIST)
```

#### Sample DAM Energy Bid Awards Data (2024-01-01)
```python
# What we actually see in DAM_EnergyBidAwards:
Settlement Point    QSE        Energy Award MW    Price
BATCAVE_RN         QPEBSE            -80         $25.50   ← NEGATIVE = CHARGING
BATCAVE_RN         QPEBSE             15         $25.50   ← POSITIVE = GENERATION
ALVIN_RN           QPEBSE             -2         $19.64   ← CHARGING
ANCHOR_ALL         QGRIEQ            -10         $15.75   ← CHARGING
```

### The Nuanced Truth

**Your colleague is CORRECT about the market design**:

1. ✅ **No Resource-Level DAM Charging Awards**: The `DAM_Load_Resource_Data` files contain ONLY:
   - Ancillary Service awards (RegUp, RegDown, RRS, ECRS, NonSpin)
   - Max/Min power consumption limits
   - **NO "Energy Award" column**

2. ✅ **DAM Energy Bids Are QSE-Level Financial Positions**: The `DAM_EnergyBidAwards` are:
   - **Settlement Point** level (not resource level)
   - **QSE** attributed (market participant, not physical device)
   - **Financial hedges** (not physical dispatch schedules)

3. ✅ **Physical Charging Happens in Real-Time**: The actual MW charging instructions come from:
   - `SCED_Load_Resource_Data` files
   - 5-minute `BasePoint` dispatch instructions
   - Settled at Resource Node RT prices

4. ✅ **This Changes with RTC+B**: The new "Energy Storage Resource" (ESR) model (coming Dec 2025) will introduce resource-specific DAM energy purchases.

### What This Means for Revenue Calculation

**We CANNOT directly attribute DAM Energy Bid Awards to specific BESS units** because:
- Awards are at Settlement Point level
- Multiple resources can share a settlement point
- QSE may be hedging for portfolio, not specific battery
- No resource identifier in the data

**We CAN calculate complete revenue using**:

#### Definitive Revenue Formula (Current Market Design)
```python
For each BESS (with Gen Resource + Load Resource pair):

DAM Revenue:
  + Gen discharge:  Σ(AwardedQuantity × DAM_Price)              # Physical
  + Gen AS awards:  Σ(RegUp/Down/RRS/ECRS/NonSpin × MCPC)       # Physical
  + Load AS awards: Σ(RegUp/Down/RRS/ECRS/NonSpin × MCPC)       # Physical

Real-Time Revenue (Net):
  + RT net position: Σ((BasePoint_Gen - BasePoint_Load) × RT_Price × 5/60)

  Where:
    BasePoint_Gen  = from SCED_Gen_Resources (discharge MW)
    BasePoint_Load = from SCED_Load_Resources (charging MW)
    RT_Price       = Resource Node 5-min SPP
    5/60           = converts 5-min MW to MWh
```

**What's NOT in this formula** (second-order effects):
- Set Point Deviation (SPD/BPD) charges
- QSE-level DAM energy hedges (can't attribute to resource)
- Mileage payments (ERCOT includes in RT energy settlement)
- RUC charges (rare for BESS)

---

## Response to the Claim

**To your colleague's assertion**:

### What's Absolutely Correct ✅

1. **"Load Resources don't receive DAM energy awards"** - CONFIRMED
   - We processed ALL `DAM_Load_Resource_Data` files
   - Column inventory: AS awards, power limits, MCPC prices
   - **NO energy award column exists**

2. **"DAM is financially binding, not physical dispatch"** - CONFIRMED
   - `DAM_EnergyBidAwards` are QSE-level, not resource-level
   - No `ResourceName` or `Load Resource Name` column
   - These are financial positions at settlement points

3. **"All physical charging happens in RT SCED"** - CONFIRMED
   - We have 114.9M rows of `SCED_Load_Resource` dispatch data
   - Contains `BasePoint` (MW) and `ResourceName`
   - This is the definitive physical charging data

4. **"Energy Bid Awards aren't resource-specific"** - CONFIRMED
   - Structure: `(SettlementPoint, QSE, Award_MW, Price)`
   - Same settlement point can have multiple batteries
   - Cannot definitively attribute to specific Load Resource

### What's Nuanced ⚠️

**The DAM Energy Bid Awards DO exist and DO show charging demand**, just not at resource level:
- 9.6M bid awards in 2024
- Negative values clearly indicate charging demand
- Priced at settlement point DA prices
- But attributed to QSE (market participant), not resource

**Interpretation**: A QSE managing a BESS *can* financially hedge expected charging costs by submitting Energy Bids (negative MW) at the battery's settlement point in DAM. If these clear, the QSE has a financial position (not a dispatch instruction). The actual physical charging is still dispatched minute-by-minute in SCED.

### Bottom Line for Your Analysis

**Your colleague is right about the market mechanics.** For accurate BESS revenue accounting:

**DO use**:
- ✅ DAM Gen Resource awards (physical discharge)
- ✅ DAM AS awards (both Gen and Load - physical capacity)
- ✅ RT SCED BasePoints (physical dispatch - both gen and load)
- ✅ Resource Node prices (DAM SPP and RT SPP)

**DON'T try to use**:
- ❌ DAM Energy Bid Awards for resource-level charging costs (can't attribute)
- ❌ Load Zone prices (use Resource Node per NPRR986/1043)
- ❌ "Hourly baseline" concept (ERCOT uses 5-min Base Points)

**The formula we'll implement**:
```
Net Revenue = DAM_discharge + DAM_AS + (RT_discharge - RT_charging)
```

Where RT charging is from `SCED_Load_Resources` with the newly-fixed `ResourceName` column.

---

## What We Proved Today

### Data Pipeline Validation

1. ✅ **SCED_Load_Resources**:
   - NOW has ResourceName (fixed from missing)
   - 114.9M rows of 5-minute charging dispatch
   - Can directly match to BESS units via Gen↔Load mapping

2. ✅ **DAM_EnergyBidAwards**:
   - Processed 56.5M records
   - Shows settlement point level bid activity
   - Confirms NO resource-level charging awards exist
   - Validates colleague's claim about financial vs physical

3. ✅ **Complete Revenue Calculation Possible**:
   - All physical dispatch data: ✓
   - All capacity payments: ✓
   - All prices: ✓
   - Resource pairing: ✓ (197 BESS mapped)

### What Changes After Dec 2025 (RTC+B)

With the new Energy Storage Resource (ESR) model:
- ESRs will receive resource-specific DAM energy purchases
- Single resource ID (not separate Gen/Load)
- Physical DAM charging schedules
- New disclosure files will have ESR energy awards

**But for historical analysis (2019-2025)**: Use the current market structure we just documented.

---

## Files Generated Today

### Code Changes
- `ercot_data_processor/src/enhanced_annual_processor.rs` (3 fixes)
  - Line 1283: Fixed ResourceName column selection
  - Lines 1899-1939: Added schema normalization for combine_dataframes
  - Lines 1399-1498: DAM Energy Bid Awards processor (already existed, now run)

### Scripts
- `regenerate_bess_data.sh` - Complete regeneration script

### Data Files
```
Location: /pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/rollup_files/

SCED_Load_Resources/
  2019.parquet  (  3.2M rows,  with ResourceName ✓)
  2020.parquet  ( 13.8M rows,  with ResourceName ✓)
  2021.parquet  ( 23.5M rows,  with ResourceName ✓)
  2022.parquet  ( 23.2M rows,  with ResourceName ✓)
  2023.parquet  ( 15.6M rows,  with ResourceName ✓)
  2024.parquet  ( 19.1M rows,  with ResourceName ✓)
  2025.parquet  ( 16.7M rows,  with ResourceName ✓)

DAM_Energy_Bid_Awards/
  2011.parquet  (  478 KB)
  2012.parquet  (   17 MB)
  ...
  2024.parquet  (  102 MB)  ← Primary analysis target
  2025.parquet  (   78 MB)
```

### Documentation
- `BESS_REAL_DATA_SITUATION.md` - Initial analysis
- `BESS_REVENUE_FINAL_PLAN.md` - Implementation options
- `BESS_DATA_ASSESSMENT_FINAL.md` - Data inventory
- `BESS_DATA_PIPELINE_COMPLETE.md` - This file

---

## Next Steps

### Immediate (Today)
1. ✅ Data pipeline complete
2. ⏳ Write BESS revenue calculator using correct formula
3. ⏳ Run for all 197 BESS units (2024 data)
4. ⏳ Generate leaderboard and validation

### Calculator Specifications

**Input**:
- BESS mapping (197 units with Gen↔Load pairs)
- Year: 2024

**Processing**:
```python
For each BESS:
  1. Load DAM_Gen_Resources for Gen Resource
  2. Load DAM_Load_Resources for Load Resource
  3. Load SCED_Gen_Resources for Gen Resource
  4. Load SCED_Load_Resources for Load Resource (NOW POSSIBLE!)
  5. Load AS_prices
  6. Load RT_prices at Resource Node

  Calculate:
    DAM_discharge_revenue = Σ(GenAward × DAM_Price)
    DAM_AS_gen_revenue = Σ(AS_awards × MCPC) for Gen
    DAM_AS_load_revenue = Σ(AS_awards × MCPC) for Load
    RT_net_revenue = Σ((BP_gen - BP_load) × RT_Price × 5/60)

    Total = DAM_discharge + DAM_AS_gen + DAM_AS_load + RT_net
```

**Output**:
- CSV: BESS revenue by unit (daily, monthly, annual aggregations)
- Leaderboard: Top performers by total revenue
- Validation: Energy balance checks, efficiency calculations

**Validation Checks**:
1. RT discharge MWh / RT charging MWh ≈ 0.85-0.90 (efficiency)
2. No simultaneous charge + discharge at same timestamp
3. SOC-feasible trajectories
4. Revenue per MW-month in reasonable range ($80k-$200k/MW-year)

---

## Conclusion

**Mission accomplished**: The ERCOT data pipeline is now complete and correct. We have verified:

1. ✅ RT charging data has ResourceName (was missing - now fixed)
2. ✅ DAM Energy Bid Awards processed (settlement point level - confirms they're not resource-specific)
3. ✅ All price data available (DA, RT, AS)
4. ✅ Schema normalization working (handles ERCOT column changes between years)
5. ✅ Your colleague's claim is correct about market design
6. ✅ We have all data needed for accurate resource-level BESS revenue calculation

**Ready to calculate revenues for 197 BESS units using the correct methodology.**

---

**Processing Stats**:
- SCED Load Resources: 173 seconds, 114.9M rows, 7 years
- DAM Energy Bid Awards: 36 seconds, 56.5M rows, 15 years
- Total data processed: 171.5M rows in under 4 minutes
- Rust is fast. 🚀

# COMPREHENSIVE BESS REVENUE ANALYSIS

## Executive Summary

This codebase contains **complete infrastructure for historical BESS revenue calculation and analysis** for ERCOT Battery Energy Storage Systems. The system calculates actual revenues earned by BESS units from historical market operations (2022-2025+), tracking all revenue streams including:
- Day-Ahead Market (DAM) energy arbitrage
- Real-Time (RT) energy imbalances
- Ancillary Services (AS) capacity payments
- Market-to-market spread revenues

---

## 1. SCRIPTS AND LOCATIONS

### Primary Calculator Scripts

**Location**: `/home/enrico/projects/power_market_pipeline/ERCOT_BESS_revenue/scripts/`

| Script Name | Purpose | Status |
|------------|---------|--------|
| `bess_revenue_calculator.py` | Main historical revenue calculator | **PRODUCTION** - ~1000 lines |
| `complete_bess_calculator_final.py` | Comprehensive multi-market calculator | **PRODUCTION** - handles DAM/RT/AS |
| `unified_bess_revenue_calculator.py` | Next-gen unified calculator | **DEVELOPMENT** - dataclass-based output |
| `bess_revenue_calculator_parquet.py` | Parquet-native version | **DEVELOPMENT** |
| `calculate_bess_true_capacity.py` | Capacity validation | **UTILITY** |
| `bess_cost_mapping.py` | BESS-to-settlement point mapping | **UTILITY** |

### Test Files

**Location**: `/home/enrico/projects/power_market_pipeline/tests/`

- `test_bess_2024.py` - Full year 2024 test
- `test_single_bess.py` - Single resource debugging
- `test_bess_mapping_v2.py` - Resource mapping validation

### Output Files (Actual Results)

**Location**: `/home/enrico/projects/power_market_pipeline/`

```
bess_revenue_2022.csv  (152 BESS, 44 columns)
bess_revenue_2023.csv  (152 BESS, 44 columns)
bess_revenue_2024.csv  (152 BESS, 44 columns)
bess_revenue_2025.csv  (152 BESS, 44 columns)
```

**Database**: `/home/enrico/projects/power_market_pipeline/output/bess_database_complete.db`

---

## 2. DATA STRUCTURE OVERVIEW

### Input Data Sources

All parquet files stored in `/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/rollup_files/`

#### 2.1 DAM_Gen_Resources (Day-Ahead Generation)
**File**: `DAM_Gen_Resources/{year}.parquet`
**Scale**: ~12M rows/year (2024: 11.8M), 41 columns, 110 MB

**Purpose**: BESS discharge (generation) awards and ancillary service capacity in DAM

**Key Columns**:
```
Core Fields:
  - ResourceName: str (BATCAVE_UNIT1, RRANCHES_UNIT2, etc.)
  - ResourceType: str (PWRSTR for BESS, WIND, COAL, GAS, etc.)
  - DeliveryDate: string (2024-01-15)
  - hour: int32 (0-23)
  - SettlementPointName: str (BATCAVE_RN, RRANCHES_ALL, etc.)
  
Energy Awards:
  - AwardedQuantity: float64 (MWh committed for discharge)
  - EnergySettlementPointPrice: float64 ($/MWh at resource node)

Ancillary Service Capacity Awards (MW):
  - RegUpAwarded: float64
  - RegDownAwarded: float64
  - RRSPFRAwarded, RRSFFRAwarded, RRSUFRAwarded: float64
  - ECRSAwarded, ECRSSDAwarded: float64
  - NonSpinAwarded: float64
  
MCPC (Market Clearing Price for Capacity):
  - Embedded in prices (RegUp MCPC, etc. columns)

Supply Curve Bids (QSE submitted):
  - QSE submitted Curve-MW{1-10}: float64 (10 price/qty pairs)
  - QSE submitted Curve-Price{1-10}: float64
```

**Time Granularity**: Hourly (hour 0-23 represents 0:00-1:00 CT to 23:00-0:00 CT)

**Data Quality**: 
- ALL rows have ResourceName and DeliveryDate
- AwardedQuantity: ~17% non-null (only hours with awards)
- AS awards: ~20% non-null (varies by resource capability)

---

#### 2.2 DAM_Load_Resources (Day-Ahead Load/Charging)
**File**: `DAM_Load_Resources/{year}.parquet`
**Scale**: ~5.8M rows/year (2024: 5.8M), 21 columns, 10 MB

**Purpose**: Load resource (charging) ancillary service awards. **NOTE: NO energy awards** - only AS capacity.

**Key Columns**:
```
Core Fields:
  - Load Resource Name: str (BATCAVE_LD1, RRANCHES_LD1, etc.)
  - Delivery Date or DeliveryDate: string
  - Hour Ending or HourEnding: str ("01:00", "02:00", ..., "24:00")
  
Ancillary Service Capacity Awards (MW):
  - RegUp Awarded: float64
  - RegDown Awarded: float64
  - RRSPFR Awarded, RRSFFR Awarded, RRSUFR Awarded: float64
  - ECRSSD Awarded, ECRSMD Awarded: float64
  - NonSpin Awarded: float64

Prices ($/MW-hr):
  - RegUp MCPC, RegDown MCPC: float64
  - RRS MCPC, ECRS MCPC, NonSpin MCPC: float64

Power Limits:
  - Max Power Consumption for Load Resource: float64 (MW capacity for charging)
  - Low Power Consumption for Load Resource: float64 (minimum operating level)
```

**CRITICAL**: No "AwardedQuantity" or "Energy Award" column exists in DAM Load Resource files. This confirms that:
- Load resources **do NOT** get DAM energy awards
- Charging is dispatched only in real-time (SCED)
- DAM Load Resource is only for AS capacity

**Time Granularity**: Hourly (Hour Ending: "01:00" to "24:00")

---

#### 2.3 SCED_Gen_Resources (Real-Time Generation)
**File**: `SCED_Gen_Resources/{year}.parquet`
**Scale**: ~31.6M rows/year (2024: 31.6M), 152 columns, 943 MB

**Purpose**: Real-time dispatch instructions for generation (discharge). 5-minute granularity.

**Key Columns** (of 152 total):
```
Core Dispatch:
  - ResourceName: str
  - SCEDTimeStamp: timestamp
  - BasePoint: float64 (MW dispatch instruction for THIS 5-minute interval)
  - TelemeteredNetOutput: float64 (MW actual output - when available)
  
Status & Limits:
  - HSL (High Sustainable Limit): float64 (MW max)
  - LSL (Low Sustainable Limit): float64 (MW min)
  - TelemeteredStatus: str (ON/OFF/AVAILABLE/etc.)

Bid Curves (40 columns):
  - SCED{1,2} Curve-MW{1-32}: float64 (MW at each price point)
  - SCED{1,2} Curve-Price{1-32}: float64 ($/MWh for each MW increment)
  - These represent QSE's bid stack for the resource
```

**Critical Notes**:
- **5-minute granularity**: Timestamps like "2024-01-15 00:05:00", "00:10:00", etc.
- Each resource dispatched independently in each 5-min SCED run
- Multiple SCED runs per hour (SCED1, SCED2, etc.)
- BasePoint is the dispatch instruction; TelemeteredNetOutput is actual MW

**Time Granularity**: 5 minutes (288 intervals/day)

**Data Quality**:
- BasePoint: ~17% non-null (only active hours)
- TelemeteredNetOutput: Present when telemetry available
- HSL/LSL: Present for all rows (resource limits)

---

#### 2.4 SCED_Load_Resources (Real-Time Charging)
**File**: `SCED_Load_Resources/{year}.parquet`
**Scale**: ~19.1M rows/year (2024: 19.1M), 32 columns, 137 MB

**Purpose**: Real-time dispatch instructions for load (charging). 5-minute granularity.

**Key Columns**:
```
Core Dispatch:
  - ResourceName: str (BATCAVE_LD1, RRANCHES_LD1, etc.) ← CRITICAL: Recently fixed!
  - SCEDTimeStamp: timestamp
  - BasePoint: float64 (MW charging dispatch)
  - RealPowerConsumption: float64 (MW actual charging - telemetered)
  
Bid Curves (SCED Bid to Buy - charging):
  - SCED Bid to Buy Curve-MW{1-10}: float64
  - SCED Bid to Buy Curve-Price{1-10}: float64
  
Limits:
  - MaxPowerConsumption: float64 (MW max charging capability)
  - LowPowerConsumption: float64 (MW min level)
```

**CRITICAL RECENT FIX** (from BESS_DATA_PIPELINE_COMPLETE.md):
- ResourceName column was missing, preventing battery matching
- Rust processor updated to include ResourceName
- Now can definitively attribute RT charging to specific Load Resources

**Time Granularity**: 5 minutes (288 intervals/day)

---

#### 2.5 Price Data

##### DA_prices (Flattened, Wide Format)
**File**: `flattened/DA_prices_{year}.parquet`
**Scale**: ~8,783 rows/year (hourly), 23 columns, 0.7 MB

**Purpose**: Day-ahead settlement point prices (hourly)

**Columns**:
```
Time:
  - DeliveryDate: date32
  - DeliveryDateStr: str (RFC3339)
  - datetime_ts: int64 (epoch)
  
Settlement Point Prices ($/MWh):
  - HB_BUSAVG: float64 (Houston area average)
  - HB_HOUSTON, HB_NORTH, HB_SOUTH, HB_WEST: float64
  - HB_HUBAVG: float64
  - LZ_*: float64 (load zones: HOUSTON, NORTH, SOUTH, WEST, etc.)
  - {ResourceNode_RN}: Individual resource node prices (when available)
```

**Coverage**: Hub and load zone prices. Individual resource node prices must be pulled from DAM_Gen_Resources.EnergySettlementPointPrice

**Time Granularity**: Hourly (1 row per delivery date, with multiple price columns)

---

##### RT_prices (Real-Time, Long Format, 5-minute)
**File**: `RT_prices/{year}.parquet`
**Scale**: Massive (2024: ~6M rows), multiple columns, ~500+ MB

**Purpose**: 5-minute real-time settlement point prices

**Columns**:
```
Time:
  - DeliveryDate: str (YYYY-MM-DD)
  - DeliveryHour: int64 (1-24, hour-ending in Central Time)
  - DeliveryInterval: int64 (1-12, 5-minute sub-intervals)
  - datetime: int64 (epoch - TIMEZONE BUG: represents CT as UTC!)

Location:
  - SettlementPointName: str (BATCAVE_RN, RRANCHES_ALL, etc.)
  - SettlementPointType: str (RESOURCE_NODE, HUB, ZONE)

Price:
  - SettlementPointPrice: float64 ($/MWh)
```

**CRITICAL TIMEZONE BUG** (from bess_revenue_calculator.py line 78):
- `datetime` epoch is **Central Time**, not UTC
- When using `pl.from_epoch()`, must handle manually:
  1. Convert to naive datetime
  2. Assign America/Chicago timezone
  3. Convert to UTC if needed for comparisons

**Time Granularity**: 5 minutes (288 intervals/day, but 12 intervals/hour)

---

##### AS_prices (Ancillary Services, Hourly)
**File**: `flattened/AS_prices_{year}.parquet`
**Scale**: ~8,783 rows/year (hourly), 8 columns, 0.2 MB

**Purpose**: Hourly MCPC (Market Clearing Price for Capacity) for AS services

**Columns**:
```
Time:
  - DeliveryDate: date32 or str
  - DeliveryDateStr: str
  - datetime_ts: int64

Service Prices ($/MW-hr):
  - REGUP: float64 (Regulation Up)
  - REGDN: float64 (Regulation Down)
  - RRS: float64 (Responsive Reserve Service)
  - ECRS: float64 (ERCOT Contingency Reserve Service)
  - NSPIN: float64 (Non-Spinning Reserve)
```

**Structure**: Each hour has ONE row with all 5 AS prices

**Time Granularity**: Hourly

---

### Derived/Mapped Data

#### BESS Resource Mapping
**File**: `bess_mapping/BESS_UNIFIED_MAPPING_V3_CLARIFIED.csv`

Maps generation resources to load resources and settlement points:

**Columns**:
- `battery_name`: Base name (BATCAVE, RRANCHES, etc.)
- `gen_resources`: Comma-separated Gen resource names (BATCAVE_UNIT1, BATCAVE_BESS1, etc.)
- `load_resources`: Comma-separated Load resource names (BATCAVE_LD1, BATCAVE_LD2, etc.)
- `settlement_points`: Primary settlement points
- `max_power_mw`: Capacity in MW
- `duration_hours`: Discharge duration (2-4 hours typically)
- `cod_date`: Commercial Operation Date

**Total Resources**: 197 BESS units currently tracked

---

## 3. REVENUE CALCULATION METHODOLOGY

### Formula Components

```
Total Revenue = DAM_Discharge + DAM_AS_Gen + DAM_AS_Load + RT_Net
```

#### 3.1 DAM Discharge Revenue
```
DAM_Discharge_Revenue = Σ(AwardedQuantity × EnergySettlementPointPrice)

Data from:
  - AwardedQuantity: DAM_Gen_Resources.AwardedQuantity (MWh)
  - Price: DAM_Gen_Resources.EnergySettlementPointPrice ($/MWh)
  
Granularity: Hourly
Filter: ResourceType = 'PWRSTR', AwardedQuantity > 0 (discharge)
```

**Key Insight**: Negative AwardedQuantity sometimes represents charging in Gen Resource (rare), but primary charging is RT-only.

---

#### 3.2 DAM Ancillary Services - Generation Resource
```
DAM_AS_Gen = Σ(RegUpAwarded × REGUP_MCPC)
           + Σ(RegDownAwarded × REGDN_MCPC)
           + Σ(RRS_Awards × RRS_MCPC)
           + Σ(ECRSAwarded × ECRS_MCPC)
           + Σ(NonSpinAwarded × NSPIN_MCPC)

Where RRS_Awards = RRSPFRAwarded + RRSFFRAwarded + RRSUFRAwarded

Data from:
  - Awards: DAM_Gen_Resources (MW capacity)
  - Prices: AS_prices ($/MW-hr)

Granularity: Hourly
Filter: Awards > 0
```

**Note**: Gen resources can provide RegUp and RegDown. AS revenue is:
- RegUp: Available when discharging (upward power adjustment)
- RegDown: Available when at any state (downward power adjustment)
- RRS, ECRS, NonSpin: Capacity reserves

---

#### 3.3 DAM Ancillary Services - Load Resource
```
DAM_AS_Load = Σ(RegUp Awarded × RegUp MCPC)
            + Σ(RegDown Awarded × RegDown MCPC)
            + Σ(RRS_Awards × RRS MCPC)
            + Σ(ECRS_Awards × ECRS MCPC)
            + Σ(NonSpin Awarded × NonSpin MCPC)

Data from:
  - Awards: DAM_Load_Resources columns (MW capacity)
  - Prices: DAM_Load_Resources MCPC columns ($/MW-hr)

Granularity: Hourly
```

**Key Difference**: Load Resources have MCPC embedded in same file, not external.

---

#### 3.4 Real-Time Net Revenue
```
RT_Net_Revenue = Σ(RT_Net_Position × RT_Price × 5/60)

Where:
  RT_Net_Position = BasePoint_Gen - BasePoint_Load (MW)
  RT_Price = SettlementPointPrice ($/MWh)
  5/60 = Converts 5-minute MW to MWh (5 min = 1/12 hour)

Data from:
  - BasePoint_Gen: SCED_Gen_Resources.BasePoint
  - BasePoint_Load: SCED_Load_Resources.BasePoint
  - Price: RT_prices at matching SettlementPointName

Granularity: 5-minute intervals
```

**Sign Convention**:
- Positive RT_Net_Position: Selling excess (discharge > charge)
- Negative RT_Net_Position: Buying deficit (charge > discharge)
- RT revenue can be positive or negative

---

### Aggregation Patterns

The CSVs show aggregation by:
1. **BESS Unit** (e.g., RRANCHES_UNIT1 paired with RRANCHES_LD1)
2. **Year** (annual summaries in the CSV output)

But underlying calculations could support:
- Daily aggregation (24-hour blocks)
- Monthly aggregation
- Quarterly aggregation
- Full period aggregation

---

## 4. MARKETS INVOLVED

### 1. Day-Ahead Energy Market (DAM)
- **Granularity**: Hourly (1 hour-ahead binding commitment)
- **Settlement**: At resource node prices
- **For BESS**: Gen Resource settles at discharge price; Load Resource gets no energy award

### 2. Real-Time Energy Market (RT/SCED)
- **Granularity**: 5-minute (continuously updated dispatch)
- **Settlement**: At 15-minute settlement interval (average of 3 × 5-min intervals)
- **For BESS**: Both Gen and Load resources settle at RT prices
- **Revenue**: Net of dispatch (positive discharge revenue, negative charging cost)

### 3. Ancillary Services Markets (AS)
- **Types Available**:
  - **RegUp**: Regulation Up (increase output/decrease consumption)
  - **RegDown**: Regulation Down (decrease output/increase consumption)
  - **RRS**: Responsive Reserve Service (3 variants: PFR, FFR, UFR)
  - **ECRS**: ERCOT Contingency Reserve Service (2 variants: SD, MD)
  - **NonSpin**: Non-Spinning Reserve
  
- **Granularity**: Hourly capacity awards
- **Settlement**: $/MW-hour (MCPC - Market Clearing Price for Capacity)
- **For BESS**: 
  - Gen resource awards capacity to provide RegUp, RegDown, RRS, ECRS, NonSpin
  - Load resource awards capacity for RegDown (reduce consumption) and other services

### 4. Spread/Arbitrage
- **Implicit**: Profit from buying low (RT cheap), selling high (DAM dear)
- **Column in output**: `da_spread_revenue` (currently shows 0 - needs validation)

---

## 5. TIME GRANULARITY & DATA PRECISION

| Market | Granularity | Field | File | Values/Day |
|--------|-------------|-------|------|-----------|
| DAM Energy | Hourly | hour: 0-23 | DAM_Gen_Resources | 24 |
| DAM AS | Hourly | hour: 0-23 | DAM_Gen/Load_Resources | 24 |
| RT Energy | 5-minute | DeliveryInterval: 1-12 | SCED_Gen/Load_Resources | 288 |
| RT Prices | 5-minute | DeliveryInterval: 1-12 | RT_prices | 288 |
| AS Prices | Hourly | hour: 0-23 or HourEnding | AS_prices | 24 |

**Critical Detail**: RT 5-minute intervals are reported as:
- DeliveryHour (1-24, hour-ending in Central Time)
- DeliveryInterval (1-12, where 1=0:05, 2=0:10, ..., 12=1:00)

Relationship: Interval `i` in hour `h` represents time `(h-1):05*i` to `(h-1):05*i + 5 min` CT

---

## 6. EXAMPLE OUTPUT DATA STRUCTURE

From `bess_revenue_2024.csv` (152 rows, 44 columns):

### Sample Row Fields:
```
bess_name                         RRANCHES_UNIT2
gen_resource                      RRANCHES_UNIT2
load_resource                     RRANCHES_LD2
resource_node                     RRANCHES_ALL
capacity_mw                       150.0

year                              2024.0
year
dam_discharge_revenue             2,408,935.14    (gen discharge revenue)
dam_charge_cost                   0.0             (should be load RT cost)
dam_charge_mwh                    0.0
da_net_energy                     2,408,935.14    (DA margin: Gen - Load)
dam_as_gen_revenue                5,707,784.60    (AS for Gen Resource)
dam_as_load_revenue               327,736.49      (AS for Load Resource)

rt_discharge_revenue              4,761,582.59    (RT Gen output revenue)
rt_charge_cost                    1,789,167.61    (RT Load consumption cost)
rt_net_revenue                    2,972,414.98    (RT net: Gen - Load)

total_revenue                     11,416,871.22   (all 4 components combined)

Normalized Metrics:
  revenue_per_mw_year             76,112.47       (total / capacity)
  revenue_per_mw_month            6,342.71
  normalized_total_per_kw_year    79.36
  normalized_energy_per_kw_year   37.41
  
AS Breakdown (Gen):
  dam_as_gen_regup                1,378,520.90
  dam_as_gen_regdown              0.0
  dam_as_gen_rrs                  149,419.49
  dam_as_gen_ecrs                 4,160,713.86
  dam_as_gen_nonspin              19,130.35

AS Breakdown (Load):
  dam_as_load_regup               0.0
  dam_as_load_regdown             327,736.49
  dam_as_load_rrs                 0.0
  dam_as_load_ecrs                0.0
  dam_as_load_nonspin             0.0

Energy Operations:
  rt_discharge_mwh                38,323.64       (GWh delivered)
  rt_charge_mwh                   68,587.32       (GWh consumed)
  rt_efficiency                   0.5588          (discharge / charge ratio)
  rt_intervals                    26,123          (5-min periods active)
  
Operational:
  active_days                     0.0             (flag)
  operational_days                351             (days with activity)
  cod_date                        2024-01-16      (commercial operation date)
```

**Key Observations**:
1. RT discharge and charge both tracked separately, never netted before revenue calc
2. Efficiency = discharge/charge = 55.88% (round-trip efficiency after losses)
3. AS revenue (especially ECRS) dominates (~50% of total)
4. Operational 351 days of 366 (most days active)

---

## 7. DATA QUALITY & CONSIDERATIONS

### Known Issues

1. **Timezone Handling in RT Prices**
   - RT `datetime` field represents Central Time but stored as UTC epoch
   - Must manually convert when matching to SCED timestamps
   - See `/home/enrico/projects/power_market_pipeline/ERCOT_BESS_revenue/scripts/bess_revenue_calculator.py` line 78

2. **Column Naming Inconsistencies**
   - DAM Load: "Hour Ending" vs "hour"
   - DAM Gen: "DeliveryDate" or "Delivery Date"
   - SCED: "SCEDTimeStamp" vs "datetime"
   - Code uses defensive checks for column existence

3. **Missing Resource Metadata**
   - ~5 resources in CSV have $0 revenue and empty settlement points
   - Indicates mapping gaps or early/late lifecycle periods
   - Examples: ODESW_UNIT1, SWTWR_UNIT1, BRP_PBL1_UNIT1

4. **Efficiency Validation**
   - Typical roundtrip: 55-87% depending on BESS technology
   - RRANCHES_UNIT2: 55.88% (conservative, multiple charge-discharge cycles)
   - TBWF_ESS_BES1: 87.23% (high efficiency, fewer cycles)

5. **Price Coverage**
   - Some resources may not have explicit resource node prices
   - Fallback to hub prices (HB_BUSAVG) used
   - Affects revenue accuracy for remote locations

---

## 8. CRITICAL ARCHITECTURAL INSIGHTS

### BESS is Split Until December 2025

**Current Structure** (Until RTC+B implementation):
- **Generation Resource** (e.g., `BATCAVE_UNIT1`) - Handles discharging, ALL AS services
- **Load Resource** (e.g., `BATCAVE_LD1`) - Handles charging, LIMITED AS (RegDown typically)
- **Revenue Calculation** - Must combine both to get complete picture

**Why This Matters**:
- 2 separate DAM awards (Gen and Load)
- 2 separate RT dispatch instructions (Gen and Load)
- 2 separate AS awards (Gen and Load)
- Must NOT double-count if merging resources

**After December 2025** (RTC+B):
- Single "Energy Storage Resource" (ESR) ID
- Unified DAM energy purchase
- Simplified revenue calculation
- Current mapping will become obsolete

---

## 9. FILE ECOSYSTEM

### Input Files Tree
```
/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/rollup_files/
├── DAM_Gen_Resources/
│   ├── 2019.parquet
│   ├── 2020.parquet
│   ├── ...
│   └── 2024.parquet          (11.8M rows, 41 cols, 110 MB)
├── DAM_Load_Resources/
│   └── 2024.parquet          (5.8M rows, 21 cols, 10 MB)
├── SCED_Gen_Resources/
│   └── 2024.parquet          (31.6M rows, 152 cols, 943 MB)
├── SCED_Load_Resources/
│   └── 2024.parquet          (19.1M rows, 32 cols, 137 MB)
├── flattened/
│   ├── DA_prices_2024.parquet     (8,783 rows, wide format)
│   └── AS_prices_2024.parquet     (8,783 rows, 8 cols)
├── RT_prices/
│   └── 2024.parquet          (Massive: ~6M rows)
└── DAM_Energy_Bid_Awards/
    └── 2024.parquet          (Optional: settlement point level bids)
```

### Processing Workflow
```
Raw ERCOT Files
    ↓
Rust Processor (ercot_data_processor)
    ↓
Parquet Conversion (Stage 2)
    ↓
Flattening to Wide Format (Stage 3) [optional for prices]
    ↓
Python Calculator Scripts
    ├─ bess_revenue_calculator.py (main)
    ├─ complete_bess_calculator_final.py (comprehensive)
    └─ unified_bess_revenue_calculator.py (next-gen)
    ↓
CSV Output + Database
    └─ bess_revenue_{year}.csv
```

---

## 10. AGGREGATION PATTERNS IN USE

### Current Implementation (bess_revenue_2024.csv)

**Aggregation Level**: Annual (one row per BESS unit per year)

**Aggregation Operations**:
```
Σ (sum) for:
- All hourly DAM revenue components
- All 5-minute RT revenue components
- All hourly AS revenue components

All operations across entire year (366 days in 2024)

Counts:
- rt_intervals: 26,123 (number of 5-min intervals with data)
- operational_days: 351 (days with any activity)
```

**Possible Alternative Aggregations** (code supports):
1. **Daily**: Sum DAM/AS for day, sum RT for day
2. **Monthly**: Sum all components for calendar month
3. **Quarterly**: Sum all components for quarter
4. **Period**: For specific date ranges (e.g., "first 100 days")

---

## 11. DOCUMENTATION

### Methodology Documents
- `/home/enrico/projects/power_market_pipeline/ERCOT_BESS_revenue/specs/BESS_REVENUE_METHODOLOGY.md` - Revenue formula
- `/home/enrico/projects/power_market_pipeline/ERCOT_BESS_revenue/specs/BESS_CHARGING_FINAL_ANSWER.md` - Where to find charging data
- `/home/enrico/projects/power_market_pipeline/docs/BESS_DATA_PIPELINE_COMPLETE.md` - Data pipeline fixes and validation

### Architecture Documents
- `/home/enrico/projects/power_market_pipeline/docs/CLAUDE.md` - Project setup instructions
- `/home/enrico/projects/power_market_pipeline/ERCOT_BESS_revenue/specs/BESS_CHARGING_INTERPRETATIONS.md` - DAM vs RT charging
- `UNIFIED_ISO_PARQUET_IMPLEMENTATION_SUMMARY.md` - 4-stage pipeline

---

## 12. EXECUTION NOTES

### Requirements
- Python 3.11+
- pandas, numpy, pyarrow
- polars (in some scripts)
- SQLite3 (for database)

### Performance
- **Time**: ~5 minutes to process 2024 (all 152 BESS)
- **Memory**: ~4-8 GB (loading annual parquets)
- **Output**: ~65 KB CSV, ~database records

### Known Limitations
1. **Settlement Point Coverage**: Not all BESS have explicit resource node prices
2. **Early Data**: Pre-2019 BESS may have incomplete mappings
3. **Real-Time Only**: No DAM charging awards available (market design)
4. **RT Timezone**: Manual conversion required for price matching

---

## SUMMARY TABLE

| Aspect | Value |
|--------|-------|
| **Total BESS Units** | 197 |
| **Years Covered** | 2019-2025 |
| **DAM Gen Rows (2024)** | 11.8M |
| **SCED Gen Rows (2024)** | 31.6M |
| **SCED Load Rows (2024)** | 19.1M |
| **RT Prices Rows (2024)** | ~6M |
| **Time Granularity (DAM)** | Hourly (24/day) |
| **Time Granularity (RT)** | 5-minute (288/day) |
| **Revenue Streams** | 4 (DAM En, DAM AS×2, RT) |
| **AS Services Tracked** | 5 (RegUp, RegDn, RRS, ECRS, NonSpin) |
| **Output Metrics** | 44 columns × 152 rows/year |
| **Average BESS Annual Revenue (2024)** | $7.5M - $11.4M |
| **Top Performer (2024)** | RRANCHES_UNIT2: $11.4M |


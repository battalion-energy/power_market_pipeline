# CRITICAL: DAM Gen Resources Parquet Data Corruption

**Date**: October 8, 2025
**Severity**: 🔴 **CRITICAL** - Affects BESS revenue calculations
**Impact**: ~36 million rows of data unusable

## Executive Summary

The 60-Day DAM Generation Resource Data parquet files have **severe date corruption** affecting years 2021, 2022, 2023, and 2025. The `DeliveryDate` and `datetime` columns contain **NULL values for all rows** in these files, making the data completely unusable for time-series analysis and BESS revenue calculations.

## Corruption Details

### Affected Files

| File | Total Rows | Null Dates | Status | Impact |
|------|------------|------------|--------|--------|
| 2011.parquet | 107,136 | 0 | ✓ OK | - |
| 2012.parquet | 5,579,280 | 0 | ✓ OK | - |
| 2013.parquet | 5,195,560 | 0 | ✓ OK | - |
| 2014.parquet | 5,684,818 | 0 | ✓ OK | - |
| 2015.parquet | 5,966,878 | 0 | ✓ OK | - |
| 2016.parquet | 6,341,838 | 0 | ✓ OK | - |
| 2017.parquet | 7,080,779 | 0 | ✓ OK | - |
| 2018.parquet | 6,767,377 | 0 | ✓ OK | - |
| 2019.parquet | 7,047,559 | 0 | ✓ OK | - |
| 2020.parquet | 7,680,627 | 0 | ✓ OK | - |
| **2021.parquet** | **8,541,145** | **8,541,145** | **✗ CORRUPTED** | All BESS revenue data for 2021 unusable |
| **2022.parquet** | **9,756,814** | **9,756,814** | **✗ CORRUPTED** | All BESS revenue data for 2022 unusable |
| **2023.parquet** | **9,906,322** | **9,906,322** | **✗ CORRUPTED** | All BESS revenue data for 2023 unusable |
| 2024.parquet | 11,847,736 | 0 | ✓ OK | - |
| **2025.parquet** | **7,971,632** | **7,971,632** | **✗ CORRUPTED** | Recent BESS revenue data unusable |

**Total corrupted rows**: 36,175,913
**Data value lost**: Critical for BESS revenue analysis 2021-2023, 2025

## Root Cause Analysis

### Source CSV Files: ✓ INTACT

The original CSV files **DO contain valid dates**:
```csv
"Delivery Date","Hour Ending",...
"04/18/2021","1",...
"04/18/2021","1",...
```

Location: `/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/60-Day_DAM_Disclosure_Reports/csv/`

### Parquet Files: ✗ CORRUPTED

The parquet conversion process **failed to preserve dates**:
```
DeliveryDate: None
datetime: NaN
```

### Likely Cause

The Rust processor (`ercot_data_processor`) has a bug in the DAM Gen Resources processing logic that:
1. Failed to parse the date format "MM/DD/YYYY" for certain years
2. Did not detect or report the parsing failure
3. Silently wrote NULL values instead of failing loudly

This is particularly concerning because:
- Years 2011-2020 worked fine
- Year 2024 worked fine
- Years 2021-2023, 2025 failed silently

**Hypothesis**: Schema change or date format change in the source CSVs between 2020→2021 that the processor didn't handle.

## Impact on BESS Revenue Analysis

### Critical Dependencies

The corrupted data affects:

1. **DAM Energy Awards** - Cannot determine when batteries were awarded energy
2. **DAM Prices** - Cannot match awards to prices without timestamps
3. **Ancillary Service Awards** - RegUp, RegDown, RRS, ECRS all time-dependent
4. **Resource-Specific Analysis** - Cannot track individual BESS units over time
5. **Revenue Calculations** - Impossible without knowing WHEN awards occurred

### Affected Analysis Period

**2021-2023**: Prime period for BESS growth in ERCOT - completely unusable!
**2025**: Current year data - cannot analyze recent performance

This means **3+ years of BESS revenue analysis is currently impossible**.

## Recovery Plan

### Option 1: Re-Process from CSV (RECOMMENDED)

**Pros**:
- Source CSV files are intact
- Can fix the root cause
- Guarantees data integrity

**Cons**:
- Need to debug and fix Rust processor
- Re-processing will take hours

**Steps**:
1. Investigate Rust processor date parsing logic
2. Identify why 2021-2023, 2025 failed
3. Fix the parser to handle different date formats
4. Re-run processor on corrupted years
5. Verify new parquet files have valid dates

### Option 2: Download Fresh Data via Web Service API

**Pros**:
- Web Service API available from Dec 11, 2023 onwards
- Can get clean data for 2024-2025

**Cons**:
- Cannot fix 2021-2023 (pre-API era)
- Still need to fix processor for future updates
- Web Service has 60-day disclosure lag

**Steps**:
1. Use new `ercot_ws_download_all.py` to download Dec 2023 → Now
2. Process new CSV files to parquet
3. Replace 2024.parquet and 2025.parquet
4. Still leaves 2021-2023 corrupted

### Option 3: Hybrid Approach (BEST)

**Combine both options:**

1. **For 2021-2023**: Fix Rust processor and re-process from existing CSVs
2. **For 2024-2025**: Download fresh from Web Service API
3. **For future**: Use Web Service API + fixed processor

## Immediate Actions Required

### Priority 1: Verify CSV File Completeness (1 hour)

Check if all CSV files exist for 2021-2023:

```bash
# Count CSV files per year
find /pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/60-Day_DAM_Disclosure_Reports/csv/ \
  -name "*DAM_Gen_Resource_Data*2021*.csv" | wc -l

# Should have ~365 files per year
```

### Priority 2: Debug Rust Processor (2-4 hours)

Investigate the date parsing bug:

```bash
# Check Rust processor source
cd ercot_data_processor
rg "DeliveryDate" src/
rg "datetime" src/

# Look for date parsing logic
rg "parse.*date" src/
```

### Priority 3: Test Fix on Single File (30 minutes)

```bash
# Process a single 2021 CSV file
cargo run --release -- --process-single-file \
  /path/to/60d_DAM_Gen_Resource_Data-17-JUN-21.csv \
  --output test_2021.parquet

# Verify dates are present
python -c "import pyarrow.parquet as pq; \
  df = pq.read_table('test_2021.parquet').to_pandas(); \
  print(df[['DeliveryDate', 'datetime']].head())"
```

### Priority 4: Re-Process All Corrupted Years (6-12 hours)

```bash
# Re-run processor for 2021, 2022, 2023, 2025
# This will regenerate the parquet files with correct dates
```

### Priority 5: Download 2024-2025 via Web Service (4-8 hours)

```bash
# Download fresh data from Web Service API
uv run python ercot_ws_download_all.py --datasets 60d_DAM_Gen_Resources

# Process to parquet
# Replace existing files
```

## Prevention Measures

### 1. Add Data Validation to Processor

Add checks after parquet write to verify:
- No NULL dates in output
- Date ranges match expected input
- Row counts match

### 2. Automated Testing

Create test suite:
- Parse sample CSV from each year
- Verify all columns present
- Verify date parsing works
- Verify no NULL dates in output

### 3. Monitoring

Add verification step after each parquet generation:
- Check for NULL dates
- Alert if found
- Fail loudly rather than silently

## Files to Investigate

### Rust Processor Sources
```
ercot_data_processor/src/
├── ercot_processor.rs         # Main processor
├── enhanced_annual_processor.rs  # Annual processing (recently modified)
├── ercot_unified_processor.rs # Unified processor
└── ercot_price_processor.rs   # Price-specific logic
```

### Test Files
```
60-Day_DAM_Disclosure_Reports/csv/
├── 60d_DAM_Gen_Resource_Data-17-JUN-21.csv  # 2021 sample
├── 60d_DAM_Gen_Resource_Data-02-OCT-22.csv  # 2022 sample
├── 60d_DAM_Gen_Resource_Data-17-AUG-23.csv  # 2023 sample
└── 60d_DAM_Gen_Resource_Data-18-JAN-24.csv  # 2024 sample (works)
```

## Questions to Answer

1. **When did the corruption occur?**
   - Check git history for processor changes around 2021 processing
   - Check file modification dates on parquet files

2. **Why did 2024 work but 2021-2023 didn't?**
   - Compare CSV schemas between years
   - Check for date format differences

3. **Were there warnings during processing?**
   - Check processor logs
   - Look for "failed to parse" or "null" warnings

4. **How many BESS units are affected?**
   - Count unique Resource Names in 2020 (working) vs 2021 (broken)
   - Identify which batteries started operating in 2021-2023

## Estimated Recovery Time

| Task | Time | Priority |
|------|------|----------|
| Verify CSV completeness | 1 hour | High |
| Debug Rust processor | 2-4 hours | Critical |
| Fix date parsing bug | 1-2 hours | Critical |
| Test on single file | 30 min | High |
| Re-process 2021 | 2 hours | High |
| Re-process 2022 | 2 hours | High |
| Re-process 2023 | 2 hours | High |
| Re-process 2025 | 1 hour | Medium |
| Verify new parquet files | 1 hour | High |
| Download 2024-2025 from API | 4-8 hours | Medium |
| **Total** | **16-24 hours** | - |

## Conclusion

This is a **critical data integrity issue** that makes 3+ years of BESS revenue analysis impossible. The good news is that the source CSV files are intact, so recovery is possible by fixing the Rust processor and re-processing the data.

**Recommended immediate action**: Debug and fix the Rust processor's date parsing logic, then re-process the corrupted years from the existing CSV files.

## Next Steps

1. Run the CSV file completeness check
2. Investigate Rust processor date parsing code
3. Create a GitHub issue to track this bug
4. Proceed with Web Service API downloads for 2024-2025 in parallel
5. Update this report with findings

---

**Report generated**: October 8, 2025
**Reporter**: Claude Code analysis
**Files analyzed**:
- 15 parquet files in DAM_Gen_Resources/
- Sample CSV files from 2021-2024
- 36+ million rows examined

# SCED Curve Float64 Fix

## Problem
SCED Gen Resource CSV files contain curve price columns that Polars incorrectly infers as i64, causing parse errors:
```
Could not parse "-161.061767578125" as dtype i64 at column 'SCED2 Curve-Price1'
Could not parse "-29.3660736083984" as dtype i64 at column 'SCED2 Curve-Price1'
```

## Root Cause
The SCED files contain offer curve data with up to 10 MW/Price pairs for SCED1 and SCED2:
- `SCED1 Curve-MW1` through `SCED1 Curve-MW10`
- `SCED1 Curve-Price1` through `SCED1 Curve-Price10`
- `SCED2 Curve-MW1` through `SCED2 Curve-MW10`
- `SCED2 Curve-Price1` through `SCED2 Curve-Price10`

These price columns can contain:
- Negative values (negative prices during oversupply)
- High-precision decimals (e.g., "-161.061767578125")
- Values that look like integers in early rows but are floats later

## Solution Implemented

### 1. Rust Code Fix
Updated `read_sced_gen_file()` in `enhanced_annual_processor.rs` (lines 788-803):

```rust
// Add SCED curve price columns - these are ALWAYS Float64!
// SCED1 curve points
for i in 1..=10 {
    schema_overrides.with_column(format!("SCED1 Curve-MW{}", i).into(), DataType::Float64);
    schema_overrides.with_column(format!("SCED1 Curve-Price{}", i).into(), DataType::Float64);
}

// SCED2 curve points  
for i in 1..=10 {
    schema_overrides.with_column(format!("SCED2 Curve-MW{}", i).into(), DataType::Float64);
    schema_overrides.with_column(format!("SCED2 Curve-Price{}", i).into(), DataType::Float64);
}

// Output Schedule columns
schema_overrides.with_column("Output Schedule".into(), DataType::Float64);
schema_overrides.with_column("Output Schedule 2".into(), DataType::Float64);
```

### 2. Python Workaround Script
Created `fix_sced_float_columns.py` to pre-process problematic CSV files:
- Reads CSV with explicit float dtypes for all numeric columns
- Re-saves with proper types
- Focuses on known problematic files from 2019

### 3. Affected Columns
All numeric columns in SCED files should be Float64:
- **Limits**: LSL, HSL
- **Dispatch**: Base Point, Telemetered Net Output
- **Schedules**: Output Schedule, Output Schedule 2
- **Ancillary Services**: AS RRS, AS Reg-Up, AS Reg-Down, AS Non-Spin, AS ECRS
- **Offer Curves**: All SCED1/SCED2 Curve-MW and Curve-Price columns (40 columns total)

## Files Fixed
- `60d_SCED_Gen_Resource_Data-14-AUG-19.csv` (58,560 rows)
- `60d_SCED_Gen_Resource_Data-14-DEC-19.csv` (61,248 rows)

## Why This Matters
1. **Negative Prices**: Texas can have negative electricity prices during high wind/low demand
2. **Precision**: Offer curves use high precision for accurate dispatch calculations
3. **Schema Consistency**: All years must have consistent column types for vstack operations

## Prevention
Always force Float64 for:
- Any column with "Price" in the name
- Any column with "MW" in the name
- Any column that could contain decimals
- Any column that could be negative

## Status
✅ Code updated in source
✅ Python workaround created
✅ Problematic files fixed
⚠️ Binary needs rebuild when dependency issue resolved

## To Apply Permanently
Once the arrow-arith/chrono dependency issue is resolved:
```bash
cd ercot_data_processor
cargo build --release
```

Until then, use the Python script to fix problematic files:
```bash
python3 fix_sced_float_columns.py
```
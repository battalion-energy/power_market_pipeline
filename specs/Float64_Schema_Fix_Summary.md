# Float64 Schema Fix Summary

## Problem
ERCOT 60-day disclosure CSV files contain numeric columns with decimal values (e.g., "84.6", "54.1") that Polars was incorrectly inferring as i64 integer type, causing parse errors:
- `Could not parse "84.6" as dtype i64 at column 'LSL'`
- `Could not parse "54.1" as dtype i64 at column 'Ancillary Service RRS'`

## Root Cause
Polars' schema inference was only looking at initial rows which might have contained whole numbers (0.0, 1.0), leading to incorrect i64 type inference instead of Float64.

## Solution Implemented

### 1. Enhanced Annual Processor Updates
Modified `read_dam_gen_file()` and `read_sced_gen_file()` functions to:
- Create explicit schema overrides forcing Float64 for all numeric columns
- Use `infer_schema(Some(10000))` to sample more rows for better inference
- Check column existence before selecting (handling schema evolution)

### 2. Key Code Changes

#### DAM File Reader (lines 568-713)
```rust
// Force critical numeric columns to be Float64
let mut schema_overrides = Schema::new();
schema_overrides.with_column("LSL".into(), DataType::Float64);
schema_overrides.with_column("HSL".into(), DataType::Float64);
schema_overrides.with_column("Awarded Quantity".into(), DataType::Float64);
// ... more columns

let df = CsvReader::from_path(file)?
    .has_header(true)
    .with_dtypes(Some(Arc::new(schema_overrides)))
    .infer_schema(Some(10000))  // Sample more rows
    .finish()?;

// Check for optional columns before selecting
if columns.contains(&"RRSFFR Awarded") {
    select_cols.push(col("RRSFFR Awarded").cast(DataType::Float64));
}
```

#### SCED File Reader (lines 715-779)
Similar pattern with Float64 overrides for:
- LSL, HSL, Base Point
- Telemetered Net Output
- Ancillary Service columns (RRS, Reg-Up, Reg-Down, Non-Spin, ECRS)

### 3. Schema Evolution Handling
Both readers now handle columns that were added over time:
- **RRSFFR, RRSUFR**: Added to DAM files in later years
- **ECRS columns**: Added around 2023
- **Telemetered Net Output**: Missing in some older SCED files

### 4. Testing
Created `test_disclosure_schema.rs` to verify Float64 handling:
```
✅ LSL dtype: Float64
✅ HSL dtype: Float64
✅ Samples: [Some(0.0), Some(0.0), Some(0.0)]
```

## Files Modified
1. `/ercot_data_processor/src/enhanced_annual_processor.rs`
   - `read_dam_gen_file()` - lines 568-713
   - `read_sced_gen_file()` - lines 715-779

2. `/ercot_data_processor/src/bin/test_disclosure_schema.rs` (new test utility)

## Impact
- ✅ 60-day disclosure files now process without parse errors
- ✅ All numeric columns correctly typed as Float64
- ✅ Schema evolution handled gracefully
- ✅ BESS revenue analysis can use accurate dispatch data

## Related Issues Fixed
- DST flag schema evolution (2011 data)
- Missing columns in older files
- Type mismatches between years
- Parse errors on decimal values
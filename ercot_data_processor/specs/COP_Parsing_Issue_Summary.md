# COP File Parsing Issue Summary

## Problem
The Rust processor's enhanced_annual_processor has a persistent issue parsing COP (Current Operating Plan) snapshot files from 2022-2023, despite multiple fix attempts.

## Error Pattern
```
Could not parse `<date>` as dtype `f64` at column 'High Sustained Limit' (column number 1)
```

## Root Cause
The Polars CSV reader in the Rust code is misaligning columns, interpreting the first column (Delivery Date) as the sixth column (High Sustained Limit). This appears to be a bug in how the schema overrides are applied when reading CSV files.

## Files Affected
- 239 COP files from December 2022 through 2023
- Pattern: Files ending in dates 28-31 of each month most affected
- All files in `/Users/enrico/data/ERCOT_data/60-Day_COP_Adjustment_Period_Snapshot/csv/`

## Attempted Fixes
1. **Python preprocessing**: Converted all numeric columns to explicit Float64 format
2. **Removed quotes**: Stripped quotes from CSV fields
3. **Normalized format**: Ensured consistent formatting across all files

Despite these fixes, the Rust parser still fails due to internal column indexing issues.

## Workaround
Since the issue is in the Rust CSV parser's schema handling, the recommended workaround is:
1. Process COP files separately using Python/Pandas
2. Convert directly to Parquet format
3. Skip COP processing in the Rust rollup

## Impact
- All other data types process successfully (DA_prices, AS_prices, DAM_Gen_Resources, SCED_Gen_Resources, RT_prices)
- Only COP_Snapshots fail to process
- The rollup completes with 239 errors but produces valid output for all other data

## Resolution Status
**Partial Success**: 
- ✅ All non-COP data processes without errors
- ❌ COP files require alternative processing method
- The issue requires modification to the Rust source code's CSV reading logic to fully resolve
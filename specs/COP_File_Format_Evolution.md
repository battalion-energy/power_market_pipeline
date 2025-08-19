# COP File Format Evolution and Regression Prevention

## Problem Statement
The ERCOT COP (Current Operating Plan) files have undergone multiple format changes over the years, causing recurring processing failures. This document comprehensively tracks all known variations and the robust solution implemented.

## Format Evolution Timeline

### Phase 1: Early 2014 (January - August)
**Filename Pattern**: `CompleteCOP_MMDDYYYY.csv`
**Has Headers**: YES ✅
**Columns**: 13
**Header Row**:
```
"Delivery Date","QSE Name","Resource Name","Hour Ending","Status","High Sustained Limit","Low Sustained Limit","High Emergency Limit","Low Emergency Limit","Reg Up","Reg Down","RRS","NSPIN"
```

### Phase 2: Late 2014 (September - December) ⚠️ CRITICAL
**Filename Pattern**: `CompleteCOP_MMDDYYYY.csv`
**Has Headers**: NO ❌
**Columns**: 13
**First Row Example**:
```
"09/25/2014","QAEN","DECKER_DPG1","01:00","OFF","325","50","326","50","0","0","0","0"
```
**Special Handling Required**: Must detect missing headers and provide column names programmatically!

### Phase 3: 2015 - December 12, 2022
**Filename Pattern**: `CompleteCOP_MMDDYYYY.csv`
**Has Headers**: YES ✅
**Columns**: 13
**Same format as Phase 1**

### Phase 4: December 13, 2022 - Present
**Filename Pattern**: `60d_COP_Adjustment_Period_Snapshot-DD-MMM-YY.csv`
**Has Headers**: YES ✅
**Columns**: 19 (RRS split + BESS SOC columns added)
**New Header Row**:
```
"Delivery Date","QSE Name","Resource Name","Hour Ending","Status","High Sustained Limit","Low Sustained Limit","High Emergency Limit","Low Emergency Limit","Reg Up","Reg Down","RRSPFR","RRSFFR","RRSUFR","NSPIN","ECRS","Minimum SOC","Maximum SOC","Hour Beginning Planned SOC"
```

## Key Changes
1. **RRS Column Split** (Dec 13, 2022): RRS → RRSPFR, RRSFFR, RRSUFR
2. **BESS Support Added**: Minimum SOC, Maximum SOC, Hour Beginning Planned SOC
3. **ECRS Added**: New ancillary service type
4. **Filename Format Change**: CompleteCOP_ → 60d_COP_Adjustment_Period_Snapshot-

## Robust Solution Implementation

### Detection Logic
```rust
// File: cop_file_reader.rs

fn detect_format(file_path: &Path) -> Result<(bool, usize, bool)> {
    let filename = file_path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    
    // New format always has headers
    if filename.starts_with("60d_COP_Adjustment_Period_Snapshot") {
        return Ok((true, 19, false));
    }
    
    // For CompleteCOP files, check date to identify late 2014
    if filename.starts_with("CompleteCOP_") {
        let date_part = filename.replace("CompleteCOP_", "").replace(".csv", "");
        if date_part.len() == 8 {
            let month = date_part[0..2].parse::<u32>().unwrap_or(0);
            let year = date_part[4..8].parse::<u32>().unwrap_or(0);
            
            // CRITICAL: Late 2014 files have NO headers!
            if year == 2014 && month >= 9 {
                return Ok((false, 13, true));
            }
        }
    }
    
    // Double-check by reading first line
    // ...
}
```

### Key Features
1. **Automatic Format Detection**: Based on filename and content inspection
2. **Header Injection**: Provides column names for headerless files
3. **Type Normalization**: Forces all numeric columns to Float64
4. **Schema Evolution**: Handles missing columns with appropriate defaults
5. **Null Value Handling**: Comprehensive handling of empty strings and NA values

## Testing Checklist

### Critical Test Cases
- [ ] CompleteCOP_09252014.csv (Late 2014, NO headers)
- [ ] CompleteCOP_01012015.csv (2015, has headers)
- [ ] CompleteCOP_12122022.csv (Last old format)
- [ ] 60d_COP_Adjustment_Period_Snapshot-13-DEC-22.csv (First new format)
- [ ] Recent 2025 files with SOC columns

### Validation Points
1. **Column Count**: Verify 13 vs 19 columns
2. **Header Detection**: Confirm presence/absence
3. **Date Parsing**: MM/DD/YYYY vs DD-MMM-YY
4. **Type Consistency**: All numeric columns as Float64
5. **SOC Columns**: Present in new format, null in old

## Error Messages and Solutions

### Error: "unable to find column 'Delivery Date'"
**Cause**: Late 2014 files have no headers
**Solution**: Detect and inject headers programmatically

### Error: "dtype mismatch: i64 and f64"
**Cause**: Type inference inconsistency
**Solution**: Force all numeric columns to Float64

### Error: "RRSFFR column not found"
**Cause**: Pre-Dec 2022 files have single RRS column
**Solution**: Schema normalization with column splitting/defaults

## Integration Points

### Files Using This Logic
1. `cop_file_reader.rs` - Core implementation
2. `schema_detector.rs` - Delegates to cop_file_reader for CompleteCOP files
3. `enhanced_annual_processor.rs` - Uses for COP processing
4. `enhanced_annual_processor_validated.rs` - Schema-validated version

### Import Statement
```rust
use crate::cop_file_reader;

// Usage
let df = cop_file_reader::read_cop_file(file_path)?;
```

## Future-Proofing

### Principles
1. **Never Assume Headers**: Always check first line
2. **Flexible Column Detection**: Don't hardcode column positions
3. **Type Safety**: Force Float64 for all numeric columns
4. **Graceful Degradation**: Missing columns get sensible defaults
5. **Comprehensive Logging**: Track format detection decisions

### Monitoring
- Log format detection results
- Track error patterns by date range
- Alert on new unrecognized formats

## Regression Prevention

### Code Review Checklist
- [ ] Does change handle ALL four format phases?
- [ ] Are late 2014 files specifically tested?
- [ ] Is Float64 type enforcement maintained?
- [ ] Are new columns added with backwards compatibility?
- [ ] Is the cop_file_reader module being used?

### Automated Tests
```rust
#[test]
fn test_late_2014_no_headers() {
    // Must handle CompleteCOP_09252014.csv format
}

#[test]
fn test_rrs_column_split() {
    // Must handle RRS → RRSPFR/RRSFFR/RRSUFR transition
}

#[test]
fn test_soc_columns_optional() {
    // Old files without SOC columns must still process
}
```

## Contact for Issues
If processing fails after following this guide:
1. Check the actual CSV file headers/content
2. Verify the date detection logic
3. Ensure cop_file_reader module is imported
4. Add new format variation to this document

Last Updated: 2025
Validated Against: ERCOT data through 2025
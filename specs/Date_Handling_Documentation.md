# ERCOT Data Pipeline Date Handling Documentation

Generated: 2025-08-18

## Overview

This document describes the date handling conventions used throughout the ERCOT data processing pipeline, ensuring consistency between Rust and Python components.

## Date Format Standards

### 1. Primary Date Storage: Date32

All parquet files in the pipeline use **Date32** format for date columns:
- **Physical Type**: INT32
- **Logical Type**: Date
- **Representation**: Days since Unix epoch (1970-01-01)
- **Column Name**: `DeliveryDate`

### 2. Timezone-Aware String Representation

Alongside the Date32 column, we include a human-readable string with timezone:
- **Column Name**: `DeliveryDateStr`
- **Format**: ISO 8601 with timezone offset
- **Example**: `2024-01-02T14:30:00-06:00` (Central Time)
- **Timezone**: `-06:00` for CST (Central Standard Time) or `-05:00` for CDT (Central Daylight Time)

### 3. Timestamp for Compatibility

For backward compatibility and time-series operations:
- **Column Name**: `datetime_ts`
- **Type**: Timestamp (INT64, nanoseconds since epoch)
- **Usage**: Full datetime with hour/minute precision

## Data Flow and Transformations

### Raw CSV Files (Source)
```
DeliveryDate: MM/DD/YYYY format (string)
HourEnding: HH:00 format (string)
```

### Rust Processor Output
```rust
// Calculate days since Unix epoch
let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
let days_since_epoch = (date - epoch).num_days() as i32;

// Parquet Schema
DeliveryDate: Date32 (days since epoch)
DeliveryDateStr: String ("YYYY-MM-DDTHH:MM:SS-06:00")
```

### Python Processing (Flatten/Combine)
```python
# Convert to Date32 for parquet storage
epoch = pd.Timestamp('1970-01-01')
df['DeliveryDate'] = ((df['datetime'] - epoch).dt.total_seconds() / 86400).astype('int32')

# Add timezone string
df['DeliveryDateStr'] = df['datetime'].dt.strftime('%Y-%m-%dT%H:%M:%S-06:00')

# Keep timestamp for compatibility
df['datetime_ts'] = df['datetime']

# Write with proper schema
table = pa.Table.from_pandas(df)
new_fields = []
for field in table.schema:
    if field.name == 'DeliveryDate':
        new_fields.append(pa.field('DeliveryDate', pa.date32()))
    else:
        new_fields.append(field)
table = table.cast(pa.schema(new_fields))
pq.write_table(table, output_file)
```

## File Types and Their Date Columns

### 1. Day-Ahead (DA) Prices
- **DeliveryDate**: Date32 - The delivery date for energy
- **DeliveryDateStr**: String - ISO format with timezone
- **datetime_ts**: Timestamp - Full datetime for each hour

### 2. Ancillary Services (AS) Prices
- **DeliveryDate**: Date32 - The delivery date for services
- **DeliveryDateStr**: String - ISO format with timezone
- **datetime_ts**: Timestamp - Full datetime for each hour

### 3. Real-Time (RT) Prices
- **DeliveryDate**: Date32 - The delivery date (date part only)
- **DeliveryDateStr**: String - ISO format with timezone (includes time)
- **datetime_ts**: Timestamp - Full datetime for each 15-minute interval
- **DeliveryInterval**: Integer - Interval within hour (1-4)

### 4. Combined Files
All combined files preserve the Date32 format from their source files:
- DA_AS_combined: Hourly data with Date32
- DA_AS_RT_combined: Hourly aggregated with Date32
- DA_AS_RT_15min_combined: 15-minute intervals with Date32

## Schema Evolution Handling

The pipeline handles multiple date formats for backward compatibility:

1. **Legacy Format** (pre-2024 data):
   - CSV files with MM/DD/YYYY strings
   - Converted to Date32 during processing

2. **Intermediate Format** (Python rollup):
   - Some files may have YYYY-MM-DD strings
   - Automatically detected and converted

3. **Target Format** (all output):
   - Date32 in parquet files
   - Consistent across Rust and Python

## Date Parsing Priority

When reading date columns, the pipeline tries formats in this order:
1. Date object (if already parsed)
2. ISO format (YYYY-MM-DD)
3. US format (MM/DD/YYYY)
4. Auto-detection (pandas inference)

## Benefits of Date32 Format

1. **Storage Efficiency**: 4 bytes per date vs 8 bytes for timestamp
2. **Cross-Platform Compatibility**: Standard parquet type
3. **Fast Date Operations**: Integer arithmetic for date calculations
4. **Clear Semantics**: Explicitly represents dates without time components

## Query Examples

### Reading with Pandas
```python
import pandas as pd

# Date32 automatically converts to datetime.date objects
df = pd.read_parquet('DA_prices_2024.parquet')
print(df['DeliveryDate'].iloc[0])  # datetime.date(2024, 1, 2)
```

### Reading with PyArrow
```python
import pyarrow.parquet as pq

table = pq.read_table('DA_prices_2024.parquet')
# DeliveryDate column has type date32[day]
print(table.schema.field('DeliveryDate'))  # date32[day]
```

### Converting Date32 to Datetime
```python
# If you need datetime64 for time series operations
df['datetime'] = pd.to_datetime(df['DeliveryDate'])

# Or use the pre-computed datetime_ts column
df['datetime'] = df['datetime_ts']
```

## Validation

To verify correct date format in any parquet file:
```python
import pyarrow.parquet as pq

pf = pq.ParquetFile('your_file.parquet')
for field in pf.schema:
    if 'DeliveryDate' == field.name:
        assert str(field.logical_type) == 'Date'
        assert str(field.physical_type) == 'INT32'
        print(f"✓ DeliveryDate is correctly stored as Date32")
```

## Migration Notes

### For Existing Scripts
Scripts expecting datetime64 columns should:
1. Use the `datetime_ts` column for backward compatibility
2. Or convert Date32 to datetime: `pd.to_datetime(df['DeliveryDate'])`

### For New Development
1. Always use Date32 for date-only columns
2. Include timezone-aware string representations
3. Preserve Date32 type when combining/transforming data

## Common Issues and Solutions

### Issue: Date shows as year 3992
**Cause**: Using `num_days_from_ce()` instead of Unix epoch
**Solution**: Calculate days from 1970-01-01

### Issue: 2023 file contains 2020 data
**Cause**: Filename pattern matching (e.g., "2023" in timestamp "120230")
**Solution**: Parse date from specific filename position

### Issue: Date parsing fails with "coerce"
**Cause**: Unexpected date format
**Solution**: Try multiple formats in sequence

## Related Files

- `ercot_data_processor/src/enhanced_annual_processor.rs` - Rust date handling
- `flatten_ercot_prices.py` - Python flattening with Date32
- `combine_ercot_prices.py` - Python combining preserving Date32
- `extract_parquet_schemas.py` - Schema documentation generator
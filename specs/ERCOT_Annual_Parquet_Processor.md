# ERCOT Annual Parquet Processor Design

## Overview

This document describes the design and implementation of a Rust program that processes ERCOT price data into annual Parquet files. The program creates organized directories for Real-Time (RT), Day-Ahead (DA), and Ancillary Services (AS) price data, with accompanying JSON metadata files describing the column schemas.

## Data Sources

The program processes data from the following ERCOT directories:

### Real-Time Prices
- **Source**: `/Users/enrico/data/ERCOT_data/Settlement_Point_Prices_at_Resource_Nodes,_Hubs_and_Load_Zones`
- **File Pattern**: `cdr.00012301.*.SPPHLZNP6905_*.csv`
- **Description**: Real-time settlement point prices at 15-minute intervals

### Day-Ahead Prices
- **Source**: `/Users/enrico/data/ERCOT_data/DAM_Settlement_Point_Prices`
- **File Pattern**: `cdr.00012331.*.DAMSPNP4190.csv`
- **Description**: Day-ahead market settlement point prices (hourly)

### Ancillary Services Prices
- **Source**: `/Users/enrico/data/ERCOT_data/DAM_Clearing_Prices_for_Capacity`
- **File Pattern**: `cdr.00012329.*.DAMCPCNP4188.csv`
- **Description**: Day-ahead ancillary services clearing prices

## Output Structure

The program creates the following directory structure:

```
/Users/enrico/data/ERCOT_data/rollup_files/
├── RT_prices/
│   ├── 2019.parquet
│   ├── 2020.parquet
│   ├── ...
│   ├── 2025.parquet
│   └── schema.json
├── DA_prices/
│   ├── 2019.parquet
│   ├── 2020.parquet
│   ├── ...
│   ├── 2025.parquet
│   └── schema.json
└── AS_prices/
    ├── 2019.parquet
    ├── 2020.parquet
    ├── ...
    ├── 2025.parquet
    └── schema.json
```

## Processing Logic

### 1. Real-Time Price Processing
- Parse CSV files containing 15-minute interval data
- Extract fields: DeliveryDate, DeliveryHour, DeliveryInterval, SettlementPointName, SettlementPointType, SettlementPointPrice, DSTFlag
- Create datetime column from DeliveryDate + DeliveryHour + DeliveryInterval
- Group by year and save to annual Parquet files
- Handle missing data and type conversions

### 2. Day-Ahead Price Processing
- Parse CSV files containing hourly data
- Extract fields: DeliveryDate, HourEnding, SettlementPoint, SettlementPointPrice, DSTFlag
- Create datetime column from DeliveryDate + HourEnding
- Group by year and save to annual Parquet files
- Ensure price columns are Float64 type

### 3. Ancillary Services Processing
- Parse CSV files containing hourly AS prices
- Extract fields: DeliveryDate, HourEnding, AncillaryType, MCPC, DSTFlag
- Create datetime column from DeliveryDate + HourEnding
- Group by year and save to annual Parquet files
- Maintain separate records for each ancillary service type (REGUP, REGDN, RRS, ECRS, NSPIN)

## Schema Definitions

### RT_prices/schema.json
```json
{
  "description": "Real-time settlement point prices at 15-minute intervals",
  "columns": {
    "datetime": {
      "type": "Datetime",
      "description": "Timestamp of the price interval"
    },
    "DeliveryDate": {
      "type": "Date",
      "description": "Operating day (YYYY-MM-DD)"
    },
    "DeliveryHour": {
      "type": "UInt32",
      "description": "Hour of the day (1-24)"
    },
    "DeliveryInterval": {
      "type": "UInt32",
      "description": "15-minute interval within hour (1-4)"
    },
    "SettlementPointName": {
      "type": "Utf8",
      "description": "Name of the settlement point"
    },
    "SettlementPointType": {
      "type": "Utf8",
      "description": "Type: RN (Resource Node), HB (Hub), LZ (Load Zone)"
    },
    "SettlementPointPrice": {
      "type": "Float64",
      "description": "Price in $/MWh (includes scarcity adders)"
    },
    "DSTFlag": {
      "type": "Utf8",
      "description": "Daylight Saving Time flag (Y/N)"
    }
  }
}
```

### DA_prices/schema.json
```json
{
  "description": "Day-ahead market settlement point prices (hourly)",
  "columns": {
    "datetime": {
      "type": "Datetime",
      "description": "Timestamp of the price hour"
    },
    "DeliveryDate": {
      "type": "Date",
      "description": "Operating day (YYYY-MM-DD)"
    },
    "HourEnding": {
      "type": "Utf8",
      "description": "Hour ending time (01:00 - 24:00)"
    },
    "SettlementPoint": {
      "type": "Utf8",
      "description": "Settlement point name"
    },
    "SettlementPointPrice": {
      "type": "Float64",
      "description": "Price in $/MWh (includes scarcity adders)"
    },
    "DSTFlag": {
      "type": "Utf8",
      "description": "Daylight Saving Time flag (Y/N)"
    }
  }
}
```

### AS_prices/schema.json
```json
{
  "description": "Day-ahead ancillary services clearing prices",
  "columns": {
    "datetime": {
      "type": "Datetime",
      "description": "Timestamp of the price hour"
    },
    "DeliveryDate": {
      "type": "Date",
      "description": "Operating day (YYYY-MM-DD)"
    },
    "HourEnding": {
      "type": "Utf8",
      "description": "Hour ending time (01:00 - 24:00)"
    },
    "AncillaryType": {
      "type": "Utf8",
      "description": "Type of ancillary service (REGUP, REGDN, RRS, ECRS, NSPIN)"
    },
    "MCPC": {
      "type": "Float64",
      "description": "Market Clearing Price for Capacity in $/MW"
    },
    "DSTFlag": {
      "type": "Utf8",
      "description": "Daylight Saving Time flag (Y/N)"
    }
  }
}
```

## Implementation Details

### Error Handling
- Skip corrupted or missing files with warnings
- Handle schema evolution (e.g., missing DSTFlag in older files)
- Continue processing on individual file failures
- Log all errors to console with file names

### Performance Optimizations
- Use parallel processing with Rayon for file parsing
- Process files in batches by year
- Use lazy evaluation with Polars LazyFrame
- Memory-efficient streaming for large datasets
- Progress bars for user feedback

### Data Quality Checks
- Verify date parsing and consistency
- Check for duplicate records (datetime + location)
- Validate price ranges (flag extreme values)
- Ensure all required columns are present
- Force Float64 type for all price columns

### Command Line Interface
```bash
# Process all price types
cargo run --release -- --process-ercot-prices /Users/enrico/data/ERCOT_data

# Process specific price type
cargo run --release -- --process-rt-prices /Users/enrico/data/ERCOT_data
cargo run --release -- --process-da-prices /Users/enrico/data/ERCOT_data
cargo run --release -- --process-as-prices /Users/enrico/data/ERCOT_data

# Skip existing files
cargo run --release -- --process-ercot-prices /Users/enrico/data/ERCOT_data --skip-existing
```

## Testing Strategy

1. **Unit Tests**: Test individual parsing functions with sample data
2. **Integration Tests**: Process small date ranges and verify output
3. **Data Validation**: Compare aggregated statistics with source data
4. **Performance Tests**: Measure processing time for full year datasets
5. **Schema Tests**: Verify JSON metadata matches actual Parquet schemas

## Future Enhancements

1. Add support for additional price types (LMP, shadow prices)
2. Implement incremental updates (process only new files)
3. Add data quality reports with statistics
4. Support for compressed output formats
5. REST API for querying processed data
6. Integration with TimescaleDB for real-time queries
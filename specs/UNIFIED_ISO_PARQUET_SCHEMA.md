# Unified ISO Parquet Schema Design

## Overview

This document defines the unified parquet file structure for all ISO market data (ERCOT, PJM, CAISO, MISO, NYISO, ISONE, SPP). The design supports:

- Day-Ahead (DA) and Real-Time (RT) energy prices
- Ancillary Services (AS) prices
- Multiple temporal resolutions: 5-minute, 15-minute, hourly
- Nodal and hub-level data
- Atomic updates with year-based partitioning

## Design Principles

1. **DO NOT modify existing ERCOT parquet files** - This is a new unified structure
2. **Preserve all data** - No data loss during conversion
3. **Consistent schema** - Same column names/types across all ISOs
4. **Year-based partitioning** - Separate files per year for scalability
5. **UTC timezone** - All timestamps normalized to UTC
6. **Float64 prices** - All price columns must be Float64
7. **Atomic updates** - New file created, then atomic mv over old file
8. **Metadata registry** - JSON files for hubs, nodes, market info

## Directory Structure

```
/home/enrico/data/unified_iso_data/
├── metadata/
│   ├── hubs/
│   │   ├── ercot_hubs.json
│   │   ├── pjm_hubs.json
│   │   ├── caiso_hubs.json
│   │   ├── miso_hubs.json
│   │   ├── nyiso_hubs.json
│   │   ├── isone_hubs.json
│   │   └── spp_hubs.json
│   ├── nodes/
│   │   ├── pjm_nodes.json          # 22,528 pnodes
│   │   ├── caiso_nodes.json
│   │   ├── miso_nodes.json
│   │   └── ...
│   ├── zones/
│   │   ├── ercot_zones.json
│   │   ├── nyiso_zones.json
│   │   └── ...
│   ├── ancillary_services/
│   │   ├── ercot_as_products.json
│   │   ├── pjm_as_products.json
│   │   └── ...
│   └── market_info.json            # Global market metadata
├── parquet/
│   ├── ercot/
│   │   ├── da_energy_hourly/
│   │   │   ├── da_energy_hourly_2019.parquet
│   │   │   ├── da_energy_hourly_2020.parquet
│   │   │   └── ...
│   │   ├── rt_energy_15min/
│   │   │   ├── rt_energy_15min_2019.parquet
│   │   │   └── ...
│   │   └── as_hourly/
│   │       ├── as_hourly_2019.parquet
│   │       └── ...
│   ├── pjm/
│   │   ├── da_energy_hourly_hub/
│   │   │   ├── da_energy_hourly_hub_2019.parquet
│   │   │   └── ...
│   │   ├── da_energy_hourly_nodal/
│   │   │   ├── da_energy_hourly_nodal_2019.parquet
│   │   │   └── ...
│   │   ├── rt_energy_5min_nodal/
│   │   │   ├── rt_energy_5min_nodal_2024.parquet
│   │   │   └── ...
│   │   ├── rt_energy_hourly_nodal/
│   │   │   ├── rt_energy_hourly_nodal_2019.parquet
│   │   │   └── ...
│   │   └── as_hourly/
│   │       ├── as_hourly_2024.parquet
│   │       └── ...
│   ├── caiso/
│   │   ├── da_energy_hourly_nodal/
│   │   ├── rt_energy_15min_nodal/   # CAISO uses 15-min, not 5-min
│   │   └── as_hourly/
│   ├── miso/
│   │   ├── da_energy_hourly_nodal/
│   │   ├── rt_energy_5min_nodal/
│   │   ├── rt_energy_hourly_nodal/
│   │   └── as_hourly/
│   ├── nyiso/
│   │   ├── da_energy_hourly/
│   │   ├── rt_energy_5min/
│   │   └── as_hourly/
│   ├── isone/
│   │   ├── da_energy_hourly/
│   │   ├── rt_energy_hourly/
│   │   └── as_hourly/
│   └── spp/
│       ├── da_energy_hourly/
│       ├── rt_energy_hourly/
│       └── as_hourly/
└── schemas/
    ├── energy_prices_schema.json
    ├── ancillary_services_schema.json
    └── metadata_schema.json
```

## Unified Energy Prices Schema

All energy price files (DA and RT, all ISOs) use this schema:

```json
{
  "name": "unified_energy_prices",
  "version": "1.0.0",
  "columns": {
    "datetime_utc": {
      "type": "Datetime[ns, UTC]",
      "description": "Timestamp in UTC (start of interval)",
      "required": true,
      "primary_key": true
    },
    "datetime_local": {
      "type": "Datetime[ns]",
      "description": "Timestamp in local timezone (for reference)",
      "required": false
    },
    "interval_start_utc": {
      "type": "Datetime[ns, UTC]",
      "description": "Interval start time in UTC",
      "required": true
    },
    "interval_end_utc": {
      "type": "Datetime[ns, UTC]",
      "description": "Interval end time in UTC",
      "required": true
    },
    "delivery_date": {
      "type": "Date",
      "description": "Delivery date (local timezone)",
      "required": true
    },
    "delivery_hour": {
      "type": "UInt8",
      "description": "Hour ending (1-24 or 0-23 depending on ISO)",
      "required": true
    },
    "delivery_interval": {
      "type": "UInt8",
      "description": "Interval within hour (1-4 for 15min, 1-12 for 5min, 0 for hourly)",
      "required": true
    },
    "interval_minutes": {
      "type": "UInt8",
      "description": "Resolution in minutes (5, 15, or 60)",
      "required": true
    },
    "iso": {
      "type": "Utf8",
      "description": "ISO identifier (ERCOT, PJM, CAISO, MISO, NYISO, ISONE, SPP)",
      "required": true,
      "primary_key": true
    },
    "market_type": {
      "type": "Utf8",
      "description": "Market type (DA, RT)",
      "required": true
    },
    "settlement_location": {
      "type": "Utf8",
      "description": "Settlement point/node/hub name",
      "required": true,
      "primary_key": true
    },
    "settlement_location_type": {
      "type": "Utf8",
      "description": "Location type (HUB, ZONE, NODE, PNODE, RESOURCE_NODE)",
      "required": true
    },
    "settlement_location_id": {
      "type": "Utf8",
      "description": "Unique location identifier (if different from name)",
      "required": false
    },
    "zone": {
      "type": "Utf8",
      "description": "Load zone or pricing zone",
      "required": false
    },
    "voltage_kv": {
      "type": "Float64",
      "description": "Voltage level in kV (for nodal data)",
      "required": false
    },
    "lmp_total": {
      "type": "Float64",
      "description": "Total Locational Marginal Price ($/MWh)",
      "required": true
    },
    "lmp_energy": {
      "type": "Float64",
      "description": "Energy component of LMP ($/MWh)",
      "required": false
    },
    "lmp_congestion": {
      "type": "Float64",
      "description": "Congestion component of LMP ($/MWh)",
      "required": false
    },
    "lmp_loss": {
      "type": "Float64",
      "description": "Loss component of LMP ($/MWh)",
      "required": false
    },
    "system_lambda": {
      "type": "Float64",
      "description": "System lambda or system energy price ($/MWh)",
      "required": false
    },
    "dst_flag": {
      "type": "Utf8",
      "description": "Daylight Saving Time flag (Y/N)",
      "required": false
    },
    "data_source": {
      "type": "Utf8",
      "description": "Original data source file or API",
      "required": false
    },
    "version": {
      "type": "UInt32",
      "description": "Data version number (for revisions)",
      "required": false
    },
    "is_current": {
      "type": "Boolean",
      "description": "Whether this is the current version",
      "required": false
    }
  },
  "primary_key": ["datetime_utc", "iso", "settlement_location"],
  "partition_by": ["iso", "year(delivery_date)"],
  "sort_by": ["datetime_utc", "settlement_location"]
}
```

## Unified Ancillary Services Schema

All ancillary services files use this schema:

```json
{
  "name": "unified_ancillary_services",
  "version": "1.0.0",
  "columns": {
    "datetime_utc": {
      "type": "Datetime[ns, UTC]",
      "description": "Timestamp in UTC (start of interval)",
      "required": true,
      "primary_key": true
    },
    "datetime_local": {
      "type": "Datetime[ns]",
      "description": "Timestamp in local timezone (for reference)",
      "required": false
    },
    "interval_start_utc": {
      "type": "Datetime[ns, UTC]",
      "description": "Interval start time in UTC",
      "required": true
    },
    "interval_end_utc": {
      "type": "Datetime[ns, UTC]",
      "description": "Interval end time in UTC",
      "required": true
    },
    "delivery_date": {
      "type": "Date",
      "description": "Delivery date (local timezone)",
      "required": true
    },
    "delivery_hour": {
      "type": "UInt8",
      "description": "Hour ending (1-24 or 0-23)",
      "required": true
    },
    "interval_minutes": {
      "type": "UInt8",
      "description": "Resolution in minutes (5, 15, or 60)",
      "required": true
    },
    "iso": {
      "type": "Utf8",
      "description": "ISO identifier",
      "required": true,
      "primary_key": true
    },
    "market_type": {
      "type": "Utf8",
      "description": "Market type (DA, RT)",
      "required": true
    },
    "as_product": {
      "type": "Utf8",
      "description": "Ancillary service product name",
      "required": true,
      "primary_key": true
    },
    "as_product_standard": {
      "type": "Utf8",
      "description": "Standardized AS product (REG_UP, REG_DOWN, SPIN, NON_SPIN, etc.)",
      "required": true
    },
    "as_region": {
      "type": "Utf8",
      "description": "AS pricing region or zone",
      "required": false
    },
    "market_clearing_price": {
      "type": "Float64",
      "description": "Market clearing price ($/MW)",
      "required": true
    },
    "cleared_quantity_mw": {
      "type": "Float64",
      "description": "Cleared quantity in MW",
      "required": false
    },
    "unit": {
      "type": "Utf8",
      "description": "Unit of measurement ($/MW, $/MWh, etc.)",
      "required": false
    },
    "data_source": {
      "type": "Utf8",
      "description": "Original data source",
      "required": false
    },
    "version": {
      "type": "UInt32",
      "description": "Data version number",
      "required": false
    },
    "is_current": {
      "type": "Boolean",
      "description": "Whether this is the current version",
      "required": false
    }
  },
  "primary_key": ["datetime_utc", "iso", "as_product"],
  "partition_by": ["iso", "year(delivery_date)"],
  "sort_by": ["datetime_utc", "as_product"]
}
```

## Metadata JSON Schemas

### Hub Metadata (`metadata/hubs/{iso}_hubs.json`)

```json
{
  "iso": "PJM",
  "last_updated": "2025-10-25T00:00:00Z",
  "hubs": [
    {
      "hub_id": "51292",
      "hub_name": "AEP_DAYTON",
      "hub_type": "TRADING_HUB",
      "zone": "AEP",
      "description": "AEP Dayton Hub",
      "active": true,
      "effective_date": "2010-01-01",
      "retirement_date": null
    }
  ]
}
```

### Node Metadata (`metadata/nodes/{iso}_nodes.json`)

```json
{
  "iso": "PJM",
  "last_updated": "2025-10-25T00:00:00Z",
  "total_nodes": 22528,
  "nodes": [
    {
      "node_id": "1",
      "node_name": "PJM-RTO",
      "node_type": "ZONE",
      "zone": "RTO",
      "voltage_kv": null,
      "latitude": null,
      "longitude": null,
      "active": true
    }
  ]
}
```

### Ancillary Service Products (`metadata/ancillary_services/{iso}_as_products.json`)

```json
{
  "iso": "ERCOT",
  "last_updated": "2025-10-25T00:00:00Z",
  "products": [
    {
      "product_name": "REGUP",
      "product_standard": "REG_UP",
      "product_type": "REGULATION",
      "description": "Regulation Up",
      "unit": "$/MW",
      "resolution_minutes": 60,
      "active": true
    },
    {
      "product_name": "REGDN",
      "product_standard": "REG_DOWN",
      "product_type": "REGULATION",
      "description": "Regulation Down",
      "unit": "$/MW",
      "resolution_minutes": 60,
      "active": true
    },
    {
      "product_name": "RRS",
      "product_standard": "SPIN",
      "product_type": "SPINNING_RESERVE",
      "description": "Responsive Reserve Service",
      "unit": "$/MW",
      "resolution_minutes": 60,
      "active": true
    },
    {
      "product_name": "ECRS",
      "product_standard": "NON_SPIN",
      "product_type": "NON_SPINNING_RESERVE",
      "description": "ERCOT Contingency Reserve Service",
      "unit": "$/MW",
      "resolution_minutes": 60,
      "active": true
    },
    {
      "product_name": "NSPIN",
      "product_standard": "NON_SPIN",
      "product_type": "NON_SPINNING_RESERVE",
      "description": "Non-Spinning Reserve",
      "unit": "$/MW",
      "resolution_minutes": 60,
      "active": true
    }
  ]
}
```

### Market Info (`metadata/market_info.json`)

```json
{
  "version": "1.0.0",
  "last_updated": "2025-10-25T00:00:00Z",
  "markets": {
    "ERCOT": {
      "full_name": "Electric Reliability Council of Texas",
      "timezone": "America/Chicago",
      "timezone_offset_hours": -6,
      "dst_observed": true,
      "da_resolution_minutes": 60,
      "rt_resolution_minutes": 15,
      "as_resolution_minutes": 60,
      "supports_nodal": true,
      "supports_hub": true,
      "supports_zone": true,
      "settlement_location_types": ["RESOURCE_NODE", "HUB", "LOAD_ZONE"],
      "data_start_date": "2019-01-01",
      "coverage": {
        "states": ["TX"],
        "regions": ["ERCOT"]
      }
    },
    "PJM": {
      "full_name": "PJM Interconnection",
      "timezone": "America/New_York",
      "timezone_offset_hours": -5,
      "dst_observed": true,
      "da_resolution_minutes": 60,
      "rt_resolution_minutes": [5, 60],
      "as_resolution_minutes": 60,
      "supports_nodal": true,
      "supports_hub": true,
      "supports_zone": true,
      "settlement_location_types": ["PNODE", "HUB", "ZONE"],
      "total_pnodes": 22528,
      "total_hubs": 24,
      "data_start_date": "2019-01-01",
      "coverage": {
        "states": ["PA", "NJ", "MD", "DE", "VA", "WV", "OH", "KY", "NC", "TN", "IL", "IN", "MI"],
        "regions": ["PJM"]
      }
    },
    "CAISO": {
      "full_name": "California Independent System Operator",
      "timezone": "America/Los_Angeles",
      "timezone_offset_hours": -8,
      "dst_observed": true,
      "da_resolution_minutes": 60,
      "rt_resolution_minutes": 15,
      "as_resolution_minutes": 60,
      "supports_nodal": true,
      "supports_hub": true,
      "supports_zone": true,
      "settlement_location_types": ["NODE", "HUB", "ZONE"],
      "data_start_date": "2019-01-01",
      "coverage": {
        "states": ["CA"],
        "regions": ["CAISO"]
      }
    },
    "MISO": {
      "full_name": "Midcontinent Independent System Operator",
      "timezone": "America/Chicago",
      "timezone_offset_hours": -6,
      "dst_observed": true,
      "da_resolution_minutes": 60,
      "rt_resolution_minutes": [5, 60],
      "as_resolution_minutes": [5, 60],
      "supports_nodal": true,
      "supports_hub": true,
      "supports_zone": true,
      "settlement_location_types": ["NODE", "HUB", "ZONE"],
      "data_start_date": "2024-01-01",
      "coverage": {
        "states": ["ND", "SD", "MN", "WI", "MI", "IA", "IL", "IN", "MO", "AR", "LA", "MS", "TX"],
        "regions": ["MISO"]
      }
    },
    "NYISO": {
      "full_name": "New York Independent System Operator",
      "timezone": "America/New_York",
      "timezone_offset_hours": -5,
      "dst_observed": true,
      "da_resolution_minutes": 60,
      "rt_resolution_minutes": [5, 60],
      "as_resolution_minutes": 60,
      "supports_nodal": true,
      "supports_hub": false,
      "supports_zone": true,
      "settlement_location_types": ["NODE", "ZONE"],
      "total_zones": 11,
      "data_start_date": "2019-01-01",
      "coverage": {
        "states": ["NY"],
        "regions": ["NYISO"]
      }
    },
    "ISONE": {
      "full_name": "ISO New England",
      "timezone": "America/New_York",
      "timezone_offset_hours": -5,
      "dst_observed": true,
      "da_resolution_minutes": 60,
      "rt_resolution_minutes": 60,
      "as_resolution_minutes": 60,
      "supports_nodal": true,
      "supports_hub": true,
      "supports_zone": true,
      "settlement_location_types": ["NODE", "HUB", "ZONE"],
      "data_start_date": "2019-01-01",
      "coverage": {
        "states": ["ME", "NH", "VT", "MA", "RI", "CT"],
        "regions": ["ISONE"]
      }
    },
    "SPP": {
      "full_name": "Southwest Power Pool",
      "timezone": "America/Chicago",
      "timezone_offset_hours": -6,
      "dst_observed": true,
      "da_resolution_minutes": 60,
      "rt_resolution_minutes": 60,
      "as_resolution_minutes": 60,
      "supports_nodal": true,
      "supports_hub": true,
      "supports_zone": true,
      "settlement_location_types": ["NODE", "HUB", "SETTLEMENT_LOCATION"],
      "data_start_date": "2019-01-01",
      "coverage": {
        "states": ["KS", "OK", "AR", "LA", "TX", "NM", "MO", "ND", "SD", "NE", "MT", "WY"],
        "regions": ["SPP"]
      }
    }
  }
}
```

## Standardized AS Product Mapping

Map each ISO's ancillary service products to standard categories:

| Standard Product | Description | ERCOT | PJM | CAISO | MISO |
|-----------------|-------------|-------|-----|-------|------|
| REG_UP | Regulation Up | REGUP | Regulation | RU | REGUP |
| REG_DOWN | Regulation Down | REGDN | Regulation | RD | REGDN |
| SPIN | Spinning Reserve | RRS | Synchronized Reserve | SR | SPIN |
| NON_SPIN | Non-Spinning | ECRS, NSPIN | Primary Reserve | NR | NSPIN |
| RAMP | Ramping Product | - | - | FRU, FRD | RAMP |

## Data Quality Checks

All parquet files must pass these validations:

1. **No duplicates**: No duplicate (datetime_utc, iso, settlement_location) tuples
2. **Sorted**: Data sorted by datetime_utc ascending
3. **Type enforcement**: All price columns are Float64
4. **Timezone**: All datetime_utc columns are UTC-aware
5. **Completeness**: No unexpected gaps in time series
6. **Range validation**: Prices within reasonable bounds (e.g., -1000 to 10000 $/MWh)
7. **Schema consistency**: All columns match expected schema

## Atomic File Update Process

When regenerating parquet files (e.g., for current year):

```python
# 1. Create new file with .tmp suffix
output_file = f"da_energy_hourly_{year}.parquet"
temp_file = f"da_energy_hourly_{year}.parquet.tmp"

# 2. Write data to temp file
df.to_parquet(temp_file, engine='pyarrow', compression='snappy')

# 3. Verify temp file integrity
verify_parquet_file(temp_file)

# 4. Atomic move (replaces old file)
os.replace(temp_file, output_file)
```

This ensures:
- No partial/corrupted files
- No downtime during update
- Automatic rollback if verification fails

## Compression Strategy

- **Primary**: Snappy compression (fast read/write, ~85% reduction)
- **Archival**: Gzip compression (higher ratio, slower)
- **Index columns**: datetime_utc, settlement_location
- **Row group size**: 1M rows (balance between compression and query performance)

## File Naming Convention

```
{market_type}_energy_{resolution}_{granularity}_{year}.parquet

Examples:
- da_energy_hourly_2024.parquet
- da_energy_hourly_hub_2024.parquet
- da_energy_hourly_nodal_2024.parquet
- rt_energy_5min_nodal_2024.parquet
- rt_energy_15min_2024.parquet
- as_hourly_2024.parquet
```

## ISO-Specific Notes

### ERCOT
- RT prices: 15-minute resolution
- DA prices: Hourly with hour-ending (1-24)
- Scarcity pricing included in LMP
- Settlement types: RN, HB, LZ

### PJM
- RT prices: Both 5-minute and hourly
- DA prices: Hourly
- LMP components available: energy, congestion, loss
- Largest nodal dataset (22,528 pnodes)

### CAISO
- RT prices: 15-minute (not 5-minute like PJM/MISO)
- Interval timestamps with start/end
- Node-focused (not hub-centric)

### MISO
- RT prices: Both 5-minute and hourly
- Ex-post vs ex-ante prices (DA)
- Weekly 5-minute files (large)
- ~7000 nodes

### NYISO
- 11 pricing zones
- Hourly and 5-minute data
- Zone-focused structure

### ISONE
- Hourly resolution only
- Hub and node pricing

### SPP
- Hourly resolution
- Settlement location terminology
- Hub and node pricing

## Version History

- **v1.0.0** (2025-10-25): Initial unified schema design

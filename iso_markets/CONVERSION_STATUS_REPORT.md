# ISO Parquet Conversion Status Report
**Date**: October 26, 2025
**Session**: 2024 DA-only conversion test

## ✅ Successfully Completed

### 1. PJM - COMPLETE ✅
- **Status**: SUCCESS
- **Rows**: 55,070,598 nodal + 79,056 hub = 55,149,654 total
- **File Size**: 532.9 MB
- **Memory Peak**: ~21GB (safe)
- **Time**: ~5 minutes
- **Output**:
  - `da_energy_hourly_nodal_2024.parquet` (527 MB, 55M rows, 6,435 nodes)
  - `da_energy_hourly_hub_2024.parquet` (2.8 MB, 79K rows, 9 hubs)
- **Metadata**: Hub and node JSON files created

### 2. CAISO - COMPLETE ✅
- **Status**: SUCCESS
- **Rows**: 144,606,781
- **File Size**: 891.8 MB
- **Output**: `da_energy_hourly_nodal_2024.parquet`
- **Unique Locations**: 16,972
- **Notes**: Very large dataset, converter handled it well

## ❌ Failed Conversions

### 3. MISO - FAILED ❌
- **Error**: `KeyError: 'MarketDateTime'`
- **Root Cause**: MISO CSV files use **pivoted format** with 24 hour-ending columns
- **File Structure**:
  ```
  Line 1: Title
  Line 2: Date
  Line 3: Blank
  Line 4: Note about EST
  Line 5: Headers - Node,Type,Value,HE 1,HE 2,...,HE 24
  Line 6+: Data rows
  ```
- **Fix Needed**: Complete rewrite of MISO converter to:
  1. Skip header rows (first 4 lines)
  2. Parse the pivoted format
  3. Unpivot 24 hour columns into rows
  4. Handle EST timezone (not CT!)
  5. Combine date from line 2 with hour-ending columns

### 4. SPP - FAILED ❌
- **Error**: `pytz.exceptions.AmbiguousTimeError: Cannot infer dst time from 2024-11-03 01:00:00`
- **Root Cause**: DST transition handling
- **Fix Needed**: Add `ambiguous='infer'` or `ambiguous='NaT'` to tz_localize calls

### 5. NYISO - Status Unknown
- **Log Size**: 670 bytes (very small)
- **Status**: Need to check log

### 6. ISONE - Status Unknown
- **Log Size**: 576 bytes (very small)
- **Status**: Need to check log

### 7. ERCOT - Not Yet Run
- **Status**: Updated path to `/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data`
- **Fix Applied**: Updated to use `dam_prices_*.csv` files instead of CDR format
- **Ready**: YES, syntax validated
- **Files Available**: 13 CSV files for 2024

## 📊 Summary Stats

| ISO | Status | Rows | Size | Issues |
|-----|--------|------|------|--------|
| **PJM** | ✅ | 55.1M | 533 MB | None |
| **CAISO** | ✅ | 144.6M | 892 MB | None |
| **ERCOT** | ⏳ | - | - | Path fixed, ready to test |
| **MISO** | ❌ | - | - | Pivoted format not supported |
| **NYISO** | ❓ | - | - | Need to check |
| **ISONE** | ❓ | - | - | Need to check |
| **SPP** | ❌ | - | - | DST handling |

**Total Successful**: 2 / 7 (29%)
**Total Data Converted**: 199.7M rows, 1.39 GB

## 🔧 Required Fixes

### Priority 1: MISO Converter Rewrite
MISO requires a completely different parsing approach due to pivoted format.

**Example Implementation Needed**:
```python
def convert_da_energy_miso_pivot(self, year=None):
    # Skip first 4 rows
    df = pd.read_csv(file, skiprows=4)

    # Unpivot hour columns
    df_melted = df.melt(
        id_vars=['Node', 'Type', 'Value'],
        value_vars=[f'HE {i}' for i in range(1, 25)],
        var_name='hour_ending',
        value_name='lmp'
    )

    # Parse hour from 'HE 1' -> 1
    # Combine with date from line 2
    # Handle EST timezone
    # Convert to unified format
```

### Priority 2: DST Handling (SPP, potentially others)
Add ambiguous time handling to all timezone conversions:
```python
df['datetime_utc'] = df['datetime_local'].dt.tz_localize(
    self.iso_timezone,
    ambiguous='infer',  # or 'NaT' to mark as invalid
    nonexistent='shift_forward'
)
```

### Priority 3: Check NYISO and ISONE logs
Investigate why conversions appear to have failed quickly.

## 📝 Data Path Updates Applied

All converters now correctly point to:
- **Input**: `/pool/ssd8tb/data/iso/{ISO}_data/csv_files/`
- **Output**: `/pool/ssd8tb/data/iso/unified_iso_data/parquet/`
- **Logs**: `/pool/ssd8tb/data/iso/unified_iso_data/logs/`

**Special Cases**:
- ERCOT: `/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data`

## 🎯 Next Steps

1. **Check NYISO and ISONE logs** to diagnose failures
2. **Add DST handling** to SPP (and verify others)
3. **Rewrite MISO converter** for pivoted format (significant work)
4. **Test ERCOT** with updated path
5. **Once all working**: Run full 2019-2025 conversion

## 💾 Current Output Location

All parquet files being written to:
```
/pool/ssd8tb/data/iso/unified_iso_data/parquet/
├── caiso/
│   └── da_energy_hourly_nodal/
│       └── da_energy_hourly_nodal_2024.parquet (892 MB)
└── pjm/
    ├── da_energy_hourly_hub/
    │   └── da_energy_hourly_hub_2024.parquet (2.8 MB)
    └── da_energy_hourly_nodal/
        └── da_energy_hourly_nodal_2024.parquet (527 MB)
```

## ⚡ Performance Notes

- **Memory Management**: Working excellently (21GB peak vs 256GB before fix)
- **Sequential Processing**: Preventing memory issues successfully
- **Chunked Processing**: BATCH_SIZE=50, CHUNK_SIZE=100k working well
- **Large Datasets**: CAISO's 144M rows processed without issues

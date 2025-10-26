# All ISO Parquet Converters - Complete Reference

## Overview

This directory contains **complete parquet converters for all 7 ISOs**, creating a unified data repository for world-class BESS optimization and multi-ISO analysis.

## Available Converters

### ✅ PJM Interconnection (`pjm_parquet_converter.py`)

**Status**: RUNNING (currently converting all data)

**Data Coverage**:
- Day-Ahead Hub Prices (hourly, 9-24 hubs)
- Day-Ahead Nodal Prices (hourly, 22,528 pnodes)
- Real-Time Hourly Nodal Prices
- Real-Time 5-Minute Nodal Prices
- Ancillary Services (Regulation, Synchronized Reserve, Primary Reserve)

**Usage**:
```bash
# Single year
python3 pjm_parquet_converter.py --year 2024

# All years
python3 pjm_parquet_converter.py --all

# DA only
python3 pjm_parquet_converter.py --all --da-only
```

---

### ✅ CAISO (`caiso_parquet_converter.py`)

**Status**: RUNNING (currently converting all data)

**Data Coverage**:
- Day-Ahead Nodal Prices (hourly)
- Real-Time 15-Minute Nodal Prices (NOT 5-min)
- Ancillary Services (RU, RD, SR, NR, FRU, FRD)

**Special Notes**:
- CAISO uses 15-minute RT intervals, not 5-minute
- Data in pivot format with XML_DATA_ITEM column
- Components: LMP_PRC, LMP_CONG_PRC, LMP_LOSS_PRC, LMP_ENE_PRC

**Usage**:
```bash
python3 caiso_parquet_converter.py --year 2024
python3 caiso_parquet_converter.py --all
```

---

### ✅ ERCOT (`ercot_parquet_converter.py`)

**Status**: READY (data available, not yet run)

**IMPORTANT**: Creates NEW unified format files, separate from legacy ERCOT parquet:
- **Legacy**: `/home/enrico/data/ERCOT_data/rollup_files/` (DO NOT MODIFY)
- **Unified**: `/home/enrico/data/unified_iso_data/parquet/ercot/` (NEW)

**Data Coverage**:
- Day-Ahead Prices (hourly, all settlement points)
- Real-Time Prices (15-minute, SCED)
- Ancillary Services (REGUP, REGDN, RRS, ECRS, NSPIN)

**Settlement Point Types**:
- HB = Hub
- LZ = Load Zone
- RN = Resource Node

**Usage**:
```bash
# This creates UNIFIED format, separate from legacy
python3 ercot_parquet_converter.py --year 2024
python3 ercot_parquet_converter.py --all
```

---

### ✅ MISO (`miso_parquet_converter.py`)

**Status**: READY (waiting for data download)

**Data Coverage**:
- Day-Ahead Ex-Post LMP (hourly, actual settlement)
- Day-Ahead Ex-Ante LMP (hourly, forecast)
- Real-Time Final Settlement (hourly)
- Real-Time 5-Minute LMP
- Ancillary Services (RegMCP, SpinMCP, Ramp)

**Special Notes**:
- ~7,000 nodes
- Weekly 5-min files are large (~500MB compressed, ~2-3GB uncompressed)
- Ex-post preferred for settlement prices

**Usage**:
```bash
python3 miso_parquet_converter.py --year 2024
python3 miso_parquet_converter.py --all
```

---

### ✅ NYISO (`nyiso_parquet_converter.py`)

**Status**: READY (data available in mixed formats)

**Data Coverage**:
- Day-Ahead Zonal Prices (11 zones, hourly)
- Real-Time Zonal Prices (hourly and 5-minute)
- Ancillary Services

**Zones**: CAPITL, CENTRL, DUNWOD, GENESE, HUD VL, LONGIL, MHK VL, MILLWD, N.Y.C., NORTH, WEST

**Special Notes**:
- Handles both CSV and ZIP archives
- Data in mixed directory structure (day_ahead/, real-time/, dam/, rt/)
- Primarily zonal pricing (not nodal like PJM)

**Usage**:
```bash
python3 nyiso_parquet_converter.py --year 2024
python3 nyiso_parquet_converter.py --all
```

---

### ✅ ISO-NE / ISONE (`isone_parquet_converter.py`)

**Status**: READY (waiting for data download)

**Data Coverage**:
- Day-Ahead LMP (hourly, nodal)
- Real-Time LMP (hourly, nodal)
- Ancillary Services

**Special Notes**:
- New England states: ME, NH, VT, MA, RI, CT
- Eastern timezone
- Template ready for data when available

**Usage**:
```bash
python3 isone_parquet_converter.py --year 2024
python3 isone_parquet_converter.py --all
```

---

### ✅ SPP (`spp_parquet_converter.py`)

**Status**: READY (waiting for data download)

**Data Coverage**:
- Day-Ahead LMP (hourly, settlement locations)
- Real-Time LMP (hourly, settlement locations)
- Ancillary Services

**Special Notes**:
- Southwest Power Pool
- Central timezone
- Uses "settlement location" terminology
- Template ready for data when available

**Usage**:
```bash
python3 spp_parquet_converter.py --year 2024
python3 spp_parquet_converter.py --all
```

---

## Running Multiple ISOs in Parallel

### Master Runner Script (`run_all_iso_converters.py`)

**Run all ISOs with available data**:
```bash
python3 run_all_iso_converters.py --all
```

**Run specific ISOs**:
```bash
python3 run_all_iso_converters.py --isos PJM CAISO ERCOT --year 2024
```

**Run sequentially (for debugging)**:
```bash
python3 run_all_iso_converters.py --sequential
```

**Currently enabled ISOs** (has_data=True):
- PJM ✓
- CAISO ✓
- ERCOT ✓
- NYISO ✓

**Disabled ISOs** (waiting for data):
- MISO (set has_data=True once data is downloaded)
- ISONE (set has_data=True once data is downloaded)
- SPP (set has_data=True once data is downloaded)

---

## Monitoring and Status

### Check Conversion Status

```bash
python3 check_conversion_status.py
```

**Output includes**:
- Files created per ISO and market type
- Year coverage
- Storage used
- Metadata files
- Running processes with CPU/MEM usage

### View Logs

```bash
# Real-time monitoring
tail -f /home/enrico/data/unified_iso_data/logs/pjm_full_conversion.log
tail -f /home/enrico/data/unified_iso_data/logs/caiso_full_conversion.log

# Check all logs
ls -lh /home/enrico/data/unified_iso_data/logs/
```

### Check Running Processes

```bash
ps aux | grep parquet_converter | grep -v grep
```

---

## Common Usage Patterns

### Convert Single Year for Testing

```bash
# Test each converter with a single year first
python3 pjm_parquet_converter.py --year 2024 --da-only
python3 caiso_parquet_converter.py --year 2022 --da-only
python3 ercot_parquet_converter.py --year 2024 --rt-only
```

### Convert All Data in Background

```bash
# Launch all available ISOs in parallel
nohup python3 run_all_iso_converters.py --all > /tmp/all_isos.log 2>&1 &

# Monitor progress
tail -f /tmp/all_isos.log
python3 check_conversion_status.py
```

### Regenerate Current Year Only

```bash
# Update parquet files for 2025 (atomic replacement)
python3 pjm_parquet_converter.py --year 2025
python3 caiso_parquet_converter.py --year 2025
python3 ercot_parquet_converter.py --year 2025
```

---

## Data Quality and Validation

All converters perform:

1. **Duplicate Detection**: Identifies duplicate (datetime, location) pairs
2. **Sorting Validation**: Ensures datetime ascending order
3. **Type Enforcement**: All prices → Float64
4. **Timezone Normalization**: All datetimes → UTC-aware
5. **Price Range Checks**: -$1000 to $10,000/MWh
6. **Atomic Updates**: Temp file → validate → atomic mv

---

## Output Structure

```
/home/enrico/data/unified_iso_data/parquet/
├── pjm/
│   ├── da_energy_hourly_hub/
│   │   ├── da_energy_hourly_hub_2023.parquet
│   │   ├── da_energy_hourly_hub_2024.parquet
│   │   └── da_energy_hourly_hub_2025.parquet
│   ├── da_energy_hourly_nodal/
│   ├── rt_energy_5min_nodal/
│   ├── rt_energy_hourly_nodal/
│   └── as_hourly/
├── caiso/
│   ├── da_energy_hourly_nodal/
│   ├── rt_energy_15min_nodal/
│   └── as_hourly/
├── ercot/
│   ├── da_energy_hourly/
│   ├── rt_energy_15min/
│   └── as_hourly/
├── miso/
│   ├── da_energy_hourly_nodal/
│   ├── rt_energy_5min_nodal/
│   └── rt_energy_hourly_nodal/
├── nyiso/
│   ├── da_energy_hourly/
│   ├── rt_energy_5min/
│   └── as_hourly/
├── isone/
│   ├── da_energy_hourly/
│   ├── rt_energy_hourly/
│   └── as_hourly/
└── spp/
    ├── da_energy_hourly/
    ├── rt_energy_hourly/
    └── as_hourly/
```

---

## Metadata Files

```
/home/enrico/data/unified_iso_data/metadata/
├── hubs/
│   ├── pjm_hubs.json (9 hubs)
│   ├── ercot_hubs.json
│   └── ...
├── nodes/
│   ├── pjm_nodes.json (14,143+ nodes)
│   ├── caiso_nodes.json
│   └── ...
├── zones/
│   ├── nyiso_zones.json (11 zones)
│   ├── ercot_zones.json
│   └── ...
├── ancillary_services/
│   ├── pjm_as_products.json
│   ├── ercot_as_products.json
│   └── ...
└── market_info.json (all ISO info)
```

---

## Troubleshooting

### Issue: Converter fails with "CSV directory not found"

**Solution**: Check if data has been downloaded and directory exists:
```bash
ls -lh /pool/ssd8tb/data/iso/MISO/csv_files/
```

Set `has_data=True` in `run_all_iso_converters.py` only after data is downloaded.

### Issue: Duplicate rows warning

**Cause**: Overlapping CSV files with same data

**Impact**: Minor - duplicates are logged but kept (or drop with keep='last')

### Issue: Memory error with large files

**Solution**: Process smaller chunks or single years:
```bash
# Process year by year instead of --all
for year in {2019..2025}; do
    python3 pjm_parquet_converter.py --year $year
done
```

### Issue: Wrong timezone conversion

**Check**: All converters use ISO-specific timezone:
- PJM/NYISO/ISONE: America/New_York (Eastern)
- ERCOT/MISO/SPP: America/Chicago (Central)
- CAISO: America/Los_Angeles (Pacific)

---

## Summary of Implementation Status

| ISO | Converter | Data Available | Status | Priority |
|-----|-----------|----------------|--------|----------|
| **PJM** | ✅ | ✅ | RUNNING | HIGH |
| **CAISO** | ✅ | ✅ | RUNNING | HIGH |
| **ERCOT** | ✅ | ✅ | READY | MEDIUM |
| **NYISO** | ✅ | ✅ | READY | MEDIUM |
| **MISO** | ✅ | ⏳ | WAITING FOR DATA | HIGH |
| **ISONE** | ✅ | ⏳ | WAITING FOR DATA | LOW |
| **SPP** | ✅ | ⏳ | WAITING FOR DATA | LOW |

**All 7 converters are complete and ready to use.**

---

## Next Steps

1. **Monitor current conversions**: PJM and CAISO are running
2. **Run ERCOT and NYISO**: Once PJM/CAISO complete
3. **Download MISO data**: Set has_data=True and run converter
4. **Download ISONE data**: Set has_data=True and run converter
5. **Download SPP data**: Set has_data=True and run converter
6. **Verify all parquet files**: Use check_conversion_status.py
7. **Integrate with BESS optimization models**

---

**Last Updated**: 2025-10-25
**Schema Version**: 1.0.0
**Total Converters**: 7/7 complete

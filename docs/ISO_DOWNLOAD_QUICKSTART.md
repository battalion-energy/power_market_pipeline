# ISO Download Quickstart Guide

Complete guide to downloading all ISO market data in parallel.

## Setup Complete ✓

All downloaders have been implemented:
- ✓ **NYISO** - Complete (CSV downloads, no auth required)
- ✓ **CAISO** - Complete (OASIS API, requires credentials)
- ✓ **IESO** - Complete (REST API, no auth required)
- ✓ **AESO** - Complete (file downloads, no auth required)
- ⚠️ **SPP** - Skeleton only (requires digital certificates - 2-4 week approval)

## Required Environment Variables

Create or update your `.env` file:

```bash
# CAISO OASIS API (required for CAISO downloads)
CAISO_USERNAME=your_username
CAISO_PASSWORD=your_password

# SPP (optional - only if you have certificates)
SPP_CERT_PATH=/path/to/certificate.pem
SPP_CERT_PASSWORD=cert_password
SPP_API_KEY=your_api_key

# Data directory (optional - defaults to /pool/ssd8tb/data/iso)
ISO_DATA_DIR=/pool/ssd8tb/data/iso
```

## Quick Start - Download All ISOs

### Test with Small Date Range (Recommended First)
```bash
# Download 1 week of data from all ISOs
python download_all_isos.py \
  --start-date 2024-01-01 \
  --end-date 2024-01-07 \
  --output-dir /pool/ssd8tb/data/iso
```

### Full Historical Download (2019-2025)
```bash
# Download full 6+ years of data (will take hours/days)
python download_all_isos.py \
  --start-date 2019-01-01 \
  --end-date 2025-10-10 \
  --output-dir /pool/ssd8tb/data/iso
```

### Download Specific ISOs Only
```bash
# Only download NYISO and CAISO
python download_all_isos.py \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --isos NYISO CAISO
```

## Download Time Estimates

Based on data volume and API rate limits:

| ISO | Data Types | Daily Files | Rate Limit | Est. Time (2019-2025) |
|-----|-----------|-------------|------------|---------------------|
| **NYISO** | DA/RT LMP + AS | ~8 files/day | None | 2-3 hours |
| **CAISO** | DA/RT LMP + AS | ~15 files/day | 10 req/min | 6-8 hours |
| **IESO** | LMP + Zonal + AS | ~10 files/day | None | 3-4 hours |
| **AESO** | Pool Price + Gen | Bulk files | None | 1 hour |
| **SPP** | DA/RT LMP + AS | N/A | N/A | N/A (needs certs) |

**Total estimated time for full historical download: 12-16 hours**

## Output Directory Structure

All downloads will be organized as:

```
/pool/ssd8tb/data/iso/
├── NYISO_data/csv_files/
│   ├── dam/zone/               # Day-ahead zonal LMP
│   ├── dam/gen/                # Day-ahead generator LMP
│   ├── rt5m/zone/              # Real-time zonal LMP
│   ├── rt5m/gen/               # Real-time generator LMP
│   ├── ancillary_services/dam/ # DA ancillary services
│   ├── ancillary_services/rtm/ # RT ancillary services
│   └── load/actual/            # Actual load
│
├── CAISO_data/csv_files/
│   ├── dam/lmp/                # Day-ahead LMP
│   ├── rt5m/lmp/               # Real-time LMP
│   ├── ancillary_services/dam/ # DA ancillary services
│   ├── ancillary_services/rtm/ # RT ancillary services
│   └── load/forecast/          # Load forecast
│
├── IESO_data/csv_files/
│   ├── da_lmp/                 # Day-ahead LMP (post-May 2025)
│   ├── rt_lmp/                 # Real-time LMP (post-May 2025)
│   ├── zonal_prices/           # Ontario zonal prices
│   ├── oemp/                   # Ontario Energy Market Price
│   ├── hoep_legacy/            # Legacy HOEP (pre-May 2025)
│   ├── ancillary_services/10s/ # 10-min sync reserves
│   ├── ancillary_services/10ns/# 10-min non-sync reserves
│   └── ancillary_services/30or/# 30-min operating reserves
│
├── AESO_data/csv_files/
│   ├── pool_price/             # Alberta pool prices
│   ├── ancillary_services/     # Operating reserves
│   ├── generation/             # Generation by fuel type
│   └── load/actual/            # Actual load
│
└── SPP_data/csv_files/
    └── (requires certificates)
```

## Monitoring Progress

The script provides detailed logging with timestamps:
- INFO level: Progress updates, file counts
- WARNING level: Missing files (404), rate limit delays
- ERROR level: Download failures, API errors

Watch the console output to monitor progress for each ISO.

## Resume Capability

All downloaders automatically skip existing files, so you can:
1. Stop the script at any time (Ctrl+C)
2. Re-run with the same date range
3. It will resume from where it left off

## Next Steps After Download

### 1. Convert CSV to Parquet (Annual Files)
```bash
# Process each ISO's CSV files into annual parquet files
cd ercot_data_processor
cargo run --release -- --process-iso NYISO --output-parquet
cargo run --release -- --process-iso CAISO --output-parquet
# etc.
```

### 2. Verify Data Integrity
```bash
python verify_parquet_files.py --iso NYISO
python verify_parquet_files.py --iso CAISO
# etc.
```

### 3. Create Combined Market Files
```bash
# Combine DA + RT + AS into unified datasets
python create_combined_market_files.py --iso NYISO --year 2024
```

## Troubleshooting

### CAISO Authentication Errors
```
Error: CAISO_USERNAME and CAISO_PASSWORD must be set in environment
```
**Solution**: Add credentials to `.env` file or export as environment variables

### CAISO Rate Limiting
```
HTTP 429 Too Many Requests
```
**Solution**: The script has built-in 6-second delays. If you still hit limits, increase retry_delay in the code.

### IESO Report Code Errors
```
Data not found (404) for PUB_DALMPEnergy
```
**Solution**: Some IESO report codes may need verification. Check [IESO Reports API documentation](https://www.ieso.ca/sector-participants/technical-interfaces).

### AESO File Discovery Issues
```
Could not discover available AESO reports
```
**Solution**: AESO's website structure may have changed. Check http://ets.aeso.ca for new download patterns.

### SPP Certificate Errors
```
SPP LMP download requires digital certificates
```
**Solution**:
1. Contact SPP Customer Relations: (501) 614-3200
2. Alternative: Use `gridstatus` Python library for SPP data

## Advanced Usage

### Download Only Hubs (Not Full Nodal)
Edit the downloader files to specify hub-only locations:
```python
# In download_all_isos.py, modify the download functions
# For example, CAISO hubs only:
await downloader.download_lmp(
    "DAM",
    config.start_date,
    config.end_date,
    locations=["TH_NP15_GEN-APND", "TH_SP15_GEN-APND"]  # Hubs only
)
```

### Customize Date Ranges Per ISO
Edit `download_all_isos.py` to create different configs for each ISO:
```python
# Different date ranges for each ISO
nyiso_config = DownloadConfig(start_date=datetime(2020, 1, 1), ...)
caiso_config = DownloadConfig(start_date=datetime(2019, 1, 1), ...)
```

### Run in Background
```bash
# Run as background process with output logging
nohup python download_all_isos.py \
  --start-date 2019-01-01 \
  --end-date 2025-10-10 \
  > iso_downloads.log 2>&1 &

# Monitor progress
tail -f iso_downloads.log
```

## Storage Requirements

Estimated storage for CSV files (2019-2025):

| ISO | CSV Size (compressed) | Parquet Size | Total |
|-----|----------------------|--------------|--------|
| NYISO | ~50 GB | ~10 GB | ~60 GB |
| CAISO | ~80 GB | ~15 GB | ~95 GB |
| IESO | ~40 GB | ~8 GB | ~48 GB |
| AESO | ~20 GB | ~4 GB | ~24 GB |
| **Total** | **~190 GB** | **~37 GB** | **~227 GB** |

Ensure you have at least **250 GB** available on the target drive.

## Support & Documentation

- **ISO API Reference**: `ISO_API_REFERENCE.md`
- **Project Architecture**: `CLAUDE.md`
- **Data Pipeline**: `ISO_DATA_PIPELINE_STRATEGY.md`
- **Downloader Code**: `downloaders/{iso}/downloader_v2.py`

## Ready to Start!

Start with a test download:
```bash
python download_all_isos.py --start-date 2024-01-01 --end-date 2024-01-07
```

This will download 1 week of data from all ISOs (except SPP) and verify everything works correctly.

Once verified, run the full historical download:
```bash
python download_all_isos.py --start-date 2019-01-01 --end-date 2025-10-10
```

Good luck! ⚡📊

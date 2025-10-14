# ERCOT Web Service API Downloader

Comprehensive Python-based downloader for ERCOT market data using the official Web Service API (available from December 11, 2023 onwards).

## Overview

This system replaces the legacy Selenium-based screen scraping approach with direct API access for:
- ✅ **Real-time updates** - No more manual ZIP file downloads
- ✅ **Automatic gap detection** - Ensures no missing data
- ✅ **State tracking** - Resumes from last successful download
- ✅ **Rate limiting** - Respects ERCOT's 30 requests/minute limit
- ✅ **Error handling** - Automatic retries with exponential backoff
- ✅ **Parallel processing** - Downloads multiple datasets efficiently

## Data Coverage

### Current Parquet File Status
(As of October 8, 2025 - from `ercot_ws_get_last_timestamps.py`)

| Priority | Dataset | Last Data | Days Old | Status |
|----------|---------|-----------|----------|--------|
| 1 | Day-Ahead Market Prices | 2025-08-19 | 50 days | ⚠️ Update needed |
| 2 | Real-Time Market Prices | Invalid timestamp | - | ⚠️ Needs investigation |
| 3 | Ancillary Services Prices | 2025-08-19 | 50 days | ⚠️ Update needed |
| 4 | DAM Gen Resources (60-day) | No timestamp | - | ⚠️ Update needed |
| 5 | SCED Gen Resources (60-day) | 2024-12-31 | 280 days | ⚠️ Update needed |
| 6 | DAM Load Resources (60-day) | No timestamp | - | ⚠️ Update needed |
| 7 | SCED Load Resources (60-day) | 2024-12-31 | 280 days | ⚠️ Update needed |

## Architecture

### Components

```
ercot_ws_downloader/
├── __init__.py              # Package initialization
├── client.py                # Web Service API client with auth & rate limiting
├── state_manager.py         # Download state tracking (JSON-based)
├── base_downloader.py       # Abstract base class for all downloaders
└── downloaders.py           # Specific downloader implementations

Scripts:
├── ercot_ws_download_all.py        # Main CLI for downloading data
├── ercot_ws_get_last_timestamps.py # Check parquet file status
└── ercot_ws_verify_downloads.py    # Verify CSV downloads for gaps
```

### Design Principles

1. **Incremental Downloads**: Never re-download existing data
2. **State Persistence**: Track last successful download in JSON file
3. **Chunked Processing**: Download in manageable chunks (1-30 days)
4. **CSV First, Parquet Later**: Download to CSV, post-process to Parquet
5. **No Parquet Modification**: Never touch existing parquet files until ready

### Download Flow

```
1. Check state file for last successful download
   ↓
2. Calculate date range to download
   ↓
3. Chunk date range (e.g., 7-day chunks)
   ↓
4. For each chunk:
   a. Authenticate with ERCOT
   b. Fetch paginated data from API
   c. Save to CSV file
   d. Update state with last timestamp
   e. Record attempt in download history
   ↓
5. Verify downloads for gaps
   ↓
6. (Future) Post-process CSVs to Parquet
```

## Installation

### Prerequisites

```bash
# Ensure you have Python dependencies
uv sync  # or pip install -r requirements.txt

# Set environment variables in .env file
ERCOT_USERNAME=your_username
ERCOT_PASSWORD=your_password
ERCOT_SUBSCRIPTION_KEY=your_subscription_key
ERCOT_DATA_DIR=/pool/ssd8tb/data/iso/ERCOT
```

### Getting ERCOT API Credentials

1. Go to https://www.ercot.com/
2. Register for an account
3. Navigate to "Market Information" → "Data" → "API Access"
4. Subscribe to the Public API
5. Copy your subscription key

## Usage

### 1. Check Current Data Status

Before downloading, check what data you have and what's missing:

```bash
# Check all parquet files for last timestamps
uv run python ercot_ws_get_last_timestamps.py

# Output shows:
# - Current data coverage
# - Days since last update
# - Data gaps
```

### 2. Download New Data

#### Test Mode (Recommended First Time)

```bash
# Test connection and show what would be downloaded
uv run python ercot_ws_download_all.py --test
```

#### Download All Datasets (Batch Mode)

```bash
# Download all datasets since last successful download
uv run python ercot_ws_download_all.py

# This will:
# - Read state from ercot_download_state.json
# - Download only missing data
# - Save CSVs in appropriate directories
# - Update state file after each successful chunk
```

#### Download Specific Datasets

```bash
# Download only DAM prices
uv run python ercot_ws_download_all.py --datasets DAM_Prices

# Download DAM and RTM prices
uv run python ercot_ws_download_all.py --datasets DAM_Prices RTM_Prices

# Download all 60-day disclosure data
uv run python ercot_ws_download_all.py --datasets \
  60d_DAM_Gen_Resources \
  60d_DAM_Load_Resources \
  60d_SCED_Gen_Resources \
  60d_SCED_Load_Resources
```

#### Continuous Mode (Real-time Updates)

```bash
# Run continuously, updating every hour
uv run python ercot_ws_download_all.py --continuous --interval 3600

# Update every 6 hours
uv run python ercot_ws_download_all.py --continuous --interval 21600
```

### 3. Verify Downloads

After downloading, verify there are no gaps:

```bash
# Verify all datasets
uv run python ercot_ws_verify_downloads.py

# Verify specific dataset
uv run python ercot_ws_verify_downloads.py --dataset DAM_Settlement_Point_Prices

# Save report to specific location
uv run python ercot_ws_verify_downloads.py --output my_report.json
```

## Dataset Details

### 1. Day-Ahead Market (DAM) Prices
- **Endpoint**: `np4-190-cd/dam_stlmnt_pnt_prices`
- **Update frequency**: Hourly (24 hours/day)
- **Chunk size**: 30 days
- **Critical for**: BESS revenue analysis, TB2 models

### 2. Real-Time Market (RTM) Prices
- **Endpoint**: `np6-785-cd/rtm_spp`
- **Update frequency**: 5-minute intervals (288/day)
- **Chunk size**: 7 days
- **Critical for**: Real-time pricing analysis, BESS dispatch

### 3. Regulation Up/Down Prices
- **Endpoint**: `np6-793-cd/reg_up_down_prices`
- **Update frequency**: Hourly
- **Chunk size**: 30 days
- **Critical for**: BESS ancillary service revenue

### 4. Reserve Prices (SPIN, NON-SPIN, RRS, ECRS)
- **Endpoint**: `np6-794-cd/reserve_prices`
- **Update frequency**: Hourly
- **Chunk size**: 30 days
- **Critical for**: BESS ancillary service revenue

### 5. 60-Day DAM Generation Resource Data
- **Endpoint**: `np3-966-cd/60d_dam_gen_res_data`
- **Update frequency**: Daily (with 60-day disclosure lag)
- **Chunk size**: 3 days (very large files)
- **Critical for**: BESS DAM awards, AS awards, revenue calculation

### 6. 60-Day DAM Load Resource Data
- **Endpoint**: `np3-966-cd/60d_dam_load_res_data`
- **Update frequency**: Daily (with 60-day disclosure lag)
- **Chunk size**: 3 days
- **Critical for**: BESS charging awards

### 7. 60-Day SCED Generation Resource Data
- **Endpoint**: `np3-965-cd/60d_sced_gen_res_data`
- **Update frequency**: 5-minute intervals (with 60-day disclosure lag)
- **Chunk size**: 1 day (MASSIVE files)
- **Critical for**: BESS actual dispatch, base points, telemetered output, SOC

### 8. 60-Day SCED Load Resource Data
- **Endpoint**: `np3-965-cd/60d_sced_load_res_data`
- **Update frequency**: 5-minute intervals (with 60-day disclosure lag)
- **Chunk size**: 1 day
- **Critical for**: BESS actual charging data

## State File Format

The `ercot_download_state.json` file tracks download progress:

```json
{
  "version": "1.0",
  "last_updated": "2025-10-08T12:00:00",
  "datasets": {
    "DAM_Prices": {
      "last_timestamp": "2025-10-07T00:00:00",
      "last_download": "2025-10-08T12:00:00",
      "last_records_count": 45000,
      "download_history": [
        {
          "timestamp": "2025-10-08T12:00:00",
          "start_date": "2025-08-20T00:00:00",
          "end_date": "2025-10-07T00:00:00",
          "success": true,
          "records_count": 180000
        }
      ]
    }
  }
}
```

## Error Handling

### Rate Limiting (429 Errors)
- Automatically waits for `Retry-After` header duration
- Default 2.5 second delay between requests
- Configurable in client initialization

### Authentication Failures
- Automatic token refresh when expired
- Tokens last ~1 hour, refreshed at 50 minutes
- Clear error messages for credential issues

### Network Errors
- Exponential backoff: 2s → 4s → 8s → 16s → 32s → 60s (max)
- Max 5 retries per request
- Logs all retry attempts

### Data Gaps
- Detected automatically by verification script
- State file prevents re-downloading existing data
- Manual gap-filling supported

## Performance Tuning

### Chunk Sizes
Default chunk sizes are conservative. Adjust based on your needs:

```python
# In downloaders.py
def get_chunk_size(self) -> int:
    return 30  # Increase for faster downloads
```

### Page Sizes
Default page size is 50,000 records. Adjust if needed:

```python
def get_page_size(self) -> int:
    return 100000  # Larger pages = fewer API calls
```

### Rate Limiting
Adjust delay between requests (default 2.5s for 30 req/min limit):

```python
client = ERCOTWebServiceClient(rate_limit_delay=1.0)  # More aggressive
```

## Troubleshooting

### Issue: "Authentication failed"
- Verify credentials in .env file
- Check subscription is active on ERCOT website
- Ensure no special characters in password

### Issue: "No data returned"
- Check date range is after Dec 11, 2023 (API start date)
- Verify endpoint names are correct
- Check ERCOT API documentation for changes

### Issue: "Rate limit exceeded"
- Increase `rate_limit_delay` in client
- Reduce `page_size` in downloader
- Run during off-peak hours

### Issue: CSV files not appearing
- Check output directory path
- Verify write permissions
- Check disk space

## Next Steps

After downloading CSV files:

1. **Verify completeness**: Run verification script
2. **Process to Parquet**: Use existing Rust processor or create new pipeline
3. **Update existing Parquet**: Append new data to year-based parquet files
4. **Run BESS revenue analysis**: Use updated data for calculations

## API Endpoint Reference

Based on `ercot_webservices/ercot_api_subset.json`:

| Report ID | Endpoint | Description |
|-----------|----------|-------------|
| NP4-190-CD | `/np4-190-cd/dam_stlmnt_pnt_prices` | DAM Settlement Point Prices |
| NP6-785-CD | `/np6-785-cd/rtm_spp` | RTM Settlement Point Prices |
| NP6-788-CD | `/np6-788-cd/dam_shadow_prices` | DAM Shadow Prices |
| NP6-793-CD | `/np6-793-cd/reg_up_down_prices` | Regulation Up/Down Prices |
| NP6-794-CD | `/np6-794-cd/reserve_prices` | Reserve Prices |
| NP3-966-CD | `/np3-966-cd/60d_dam_gen_res_data` | 60-Day DAM Gen Resource Data |
| NP3-966-CD | `/np3-966-cd/60d_dam_load_res_data` | 60-Day DAM Load Resource Data |
| NP3-965-CD | `/np3-965-cd/60d_sced_gen_res_data` | 60-Day SCED Gen Resource Data |
| NP3-965-CD | `/np3-965-cd/60d_sced_load_res_data` | 60-Day SCED Load Resource Data |

## Support

For issues:
1. Check logs in `ercot_ws_download.log`
2. Review state file `ercot_download_state.json`
3. Run verification script to detect gaps
4. Check ERCOT API status: https://www.ercot.com/services/api

## License

Part of the Power Market Pipeline project.

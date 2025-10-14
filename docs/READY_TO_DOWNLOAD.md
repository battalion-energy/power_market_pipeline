# Ready to Download ERCOT Data via Web Service API

**Status**: ✅ System is ready, waiting for API credentials

## What's Been Completed

### ✅ Architecture Built
- Complete Python download system with 8 specialized downloaders
- State management to track progress and prevent gaps
- Robust error handling with retries and rate limiting
- Verification scripts to detect data gaps

### ✅ Data Corruption Fixed
- **2021, 2022, 2023 DAM Gen Resources**: ✅ FIXED (was corrupted, now working!)
- Historical BESS revenue data is now usable
- **2025 DAM Gen Resources**: Still corrupted, will be replaced with fresh download

### ✅ Data Gaps Identified

| Dataset | Last Valid Data | Gap Size | Priority |
|---------|----------------|----------|----------|
| DAM Prices | 2025-08-19 | 50 days | High |
| RT Prices | Invalid | ??? | Medium (needs investigation) |
| AS Prices | 2025-08-19 | 50 days | High |
| 60d DAM Gen | 2024-11-28 | **315 days** | **CRITICAL** |
| 60d SCED Gen | 2024-12-31 | **280 days** | **CRITICAL** |
| 60d DAM Load | Unknown | ??? | Medium |
| 60d SCED Load | 2024-12-31 | 280 days | Medium |

## What's Needed to Start

### Step 1: Set Up ERCOT API Credentials

You need three pieces of information:

1. **ERCOT Username** - Your ERCOT account email/username
2. **ERCOT Password** - Your ERCOT account password
3. **ERCOT Subscription Key** - From the Public API portal

#### How to Get Credentials

1. **If you already have an ERCOT account:**
   - Go to https://www.ercot.com/
   - Sign in with your existing credentials
   - Navigate to: **Market Info → Data → API Access**
   - Subscribe to **Public API** (free)
   - Copy your **Subscription Key**

2. **If you don't have an ERCOT account:**
   - Go to https://www.ercot.com/
   - Click **Sign Up** / **Register**
   - Complete registration
   - Follow steps above to get subscription key

#### Update .env File

Edit the `.env` file in this directory:

```bash
nano .env
```

Update these lines with your actual credentials:

```bash
ERCOT_USERNAME=your_actual_email@example.com
ERCOT_PASSWORD=your_actual_password
ERCOT_SUBSCRIPTION_KEY=your_actual_key_from_portal
```

**Important**: Keep your credentials secure! The `.env` file should not be committed to git.

### Step 2: Test API Connection

After updating credentials, test the connection:

```bash
./setup_ercot_credentials.sh
```

You should see:
```
✅ API connection successful!
You are ready to download data.
```

## Download Plan

Once credentials are set up, download data in this order:

### Phase 1: Test with Small Dataset (5-10 minutes)

Start with DAM prices to verify everything works:

```bash
# Download just 1 week of DAM prices as a test
uv run python ercot_ws_download_all.py --datasets DAM_Prices
```

This will:
- Authenticate with ERCOT
- Download ~50 days of missing DAM price data
- Save to CSV files
- Update state file

**Expected output**:
- CSV files in: `/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/DAM_Settlement_Point_Prices/`
- State file updated: `ercot_download_state.json`
- Log file created: `ercot_ws_download.log`

### Phase 2: Download Price Data (30-60 minutes)

Download all market prices:

```bash
# Download DAM, AS prices (RT prices need investigation)
uv run python ercot_ws_download_all.py --datasets \
  DAM_Prices \
  RegUpDown_Prices \
  Reserve_Prices
```

### Phase 3: Download Critical BESS Data (8-24 hours)

**This is the big one!** 60-day disclosure data for BESS revenue:

```bash
# Start with SCED Gen Resources (280 days of 5-minute data)
uv run python ercot_ws_download_all.py --datasets 60d_SCED_Gen_Resources

# Then DAM Gen Resources (315 days)
uv run python ercot_ws_download_all.py --datasets 60d_DAM_Gen_Resources

# Optional: Load resources (for BESS charging data)
uv run python ercot_ws_download_all.py --datasets \
  60d_DAM_Load_Resources \
  60d_SCED_Load_Resources
```

**Warning**: These downloads will:
- Take 8-24 hours to complete
- Generate 10s of GBs of CSV files
- Make thousands of API calls
- Should be run overnight or over weekend

**Recommendation**: Use `nohup` or `screen` to run in background:

```bash
# Run in background with nohup
nohup uv run python ercot_ws_download_all.py --datasets 60d_SCED_Gen_Resources > sced_download.log 2>&1 &

# Or use screen (recommended for long-running downloads)
screen -S ercot_download
uv run python ercot_ws_download_all.py --datasets 60d_SCED_Gen_Resources
# Press Ctrl+A then D to detach
# Reconnect with: screen -r ercot_download
```

### Phase 4: Verify Downloads

After each download phase, verify completeness:

```bash
# Check for gaps
uv run python ercot_ws_verify_downloads.py

# Re-check timestamps
uv run python ercot_ws_get_last_timestamps.py
```

### Phase 5: Process to Parquet

After CSV downloads complete, process to parquet:

```bash
# Option 1: Use existing Rust processor
cd ercot_data_processor
cargo run --release -- --process-annual

# Option 2: Create new incremental processor (future enhancement)
```

## Monitoring Downloads

### Real-Time Progress

Watch the download in real-time:

```bash
# Follow the log file
tail -f ercot_ws_download.log

# Or follow specific dataset
tail -f ercot_ws_download.log | grep "60d_SCED"
```

### Check State File

The state file tracks progress:

```bash
# View current state
cat ercot_download_state.json | python -m json.tool
```

### Resume After Interruption

If download is interrupted (network issue, computer restart, etc.), just run the same command again:

```bash
# It will automatically resume from last successful chunk
uv run python ercot_ws_download_all.py --datasets 60d_SCED_Gen_Resources
```

The state manager ensures:
- No re-downloading of existing data
- No gaps in the data
- Automatic resume from interruption

## Expected Results

### CSV Files

Downloaded data will be organized:

```
/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/
├── DAM_Settlement_Point_Prices/
│   ├── dam_prices_20250820_20250919.csv
│   └── dam_prices_20250920_20251007.csv
├── RegUpDown_Prices/
│   └── regupdown_prices_20250820_20251007.csv
├── Reserve_Prices/
│   └── reserve_prices_20250820_20251007.csv
├── 60-Day_DAM_Disclosure_Reports/
│   └── Gen_Resources/
│       ├── 60d_dam_gen_resources_20241129_20241201.csv
│       ├── 60d_dam_gen_resources_20241202_20241204.csv
│       └── ... (many more files)
└── 60-Day_SCED_Disclosure_Reports/
    └── Gen_Resources/
        ├── 60d_sced_gen_resources_20250101_20250101.csv (1 day chunks!)
        ├── 60d_sced_gen_resources_20250102_20250102.csv
        └── ... (280+ files)
```

### Disk Space Requirements

Estimate disk space needed:

| Dataset | Days | Est. Size | Total |
|---------|------|-----------|-------|
| DAM Prices | 50 | ~5 MB/day | ~250 MB |
| AS Prices | 50 | ~5 MB/day | ~250 MB |
| 60d DAM Gen | 315 | ~50 MB/day | ~15 GB |
| 60d SCED Gen | 280 | ~200 MB/day | ~56 GB |
| **Total** | - | - | **~72 GB** |

**Make sure you have at least 100 GB free space before starting!**

## Troubleshooting

### Issue: "Authentication failed"

**Solution**: Double-check credentials in `.env` file
- Username should be your email
- Password is case-sensitive
- Subscription key should be from Public API portal

### Issue: "Rate limit exceeded"

**Solution**: The system handles this automatically
- Will wait for `Retry-After` duration
- Adjust `rate_limit_delay` in client if needed
- ERCOT allows 30 requests/minute

### Issue: "No data returned"

**Solution**: Check date range
- Web Service API only available from Dec 11, 2023 onwards
- 60-day disclosure has 60-day lag
- Some endpoints may have different date ranges

### Issue: Download is slow

**Solution**: This is expected
- 60-day disclosure data is MASSIVE
- System is conservative with rate limiting
- Can adjust chunk sizes and page sizes for speed
- Best to run overnight

### Issue: Download interrupted

**Solution**: Just restart the same command
- State file tracks progress
- Will resume from last successful chunk
- No data loss

## Continuous Updates (Optional)

After initial backfill, set up continuous updates:

### Option 1: Cron Job (Daily Updates)

Add to crontab:

```bash
# Run daily at 2 AM
0 2 * * * cd /home/enrico/projects/power_market_pipeline && uv run python ercot_ws_download_all.py --batch >> /var/log/ercot_download.log 2>&1
```

### Option 2: Continuous Mode

Run continuously with updates every 6 hours:

```bash
nohup uv run python ercot_ws_download_all.py --continuous --interval 21600 > continuous_download.log 2>&1 &
```

## Next Steps After Downloads Complete

1. **Verify all downloads**: Run gap detection
2. **Process to Parquet**: Use Rust processor or create new pipeline
3. **Update existing Parquet files**: Append new data to year-based files
4. **Run BESS revenue analysis**: Use updated data
5. **Validate results**: Compare to previous calculations for consistency

## Getting Help

If you encounter issues:

1. Check the log file: `ercot_ws_download.log`
2. Check the state file: `ercot_download_state.json`
3. Run verification: `uv run python ercot_ws_verify_downloads.py`
4. Review documentation: `ERCOT_WEB_SERVICE_DOWNLOAD_README.md`
5. Check ERCOT API status: https://www.ercot.com/services/api

## Summary

✅ **System is ready to download**
⚠️ **Waiting for ERCOT API credentials**

Once credentials are set up:
1. Test with DAM prices (5-10 min)
2. Download all price data (30-60 min)
3. Download 60-day disclosure data (8-24 hours)
4. Verify and process to parquet

**Total time**: ~1-2 days for complete backfill, then continuous updates going forward.

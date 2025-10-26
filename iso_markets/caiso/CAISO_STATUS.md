# CAISO Data Pipeline Status

**Date**: 2025-10-25
**Status**: CSV Downloads In Progress (2023-2025)

## ✅ Completed

### 1. CAISO API Client (`caiso_api_client.py`)
- Public API (no authentication required)
- Conservative 6-second delays (CAISO limit: 1 request per 5 seconds)
- Handles both CSV and XML formats
- Exponential backoff retry (30s, 60s, 120s, 240s, 480s)

### 2. Download Scripts
- DA Nodal LMPs downloader
- RT 5-Minute Nodal LMPs downloader (2-hour chunking)
- Ancillary Services downloader
- Unified updater with auto-resume (`update_caiso_with_resume.py`)

### 3. Cron Job
- ✅ Installed and tested
- Runs daily at 10 AM PT
- Auto-resumes from last date
- Successfully tested - working without 429 errors

### 4. Throttling Research
**CAISO OASIS API Rate Limit**: 1 request every 5 seconds (12 requests/minute)

**Implementation**:
- Using 6-second delays (conservative)
- No more 429 rate limit errors observed
- Exponential backoff for transient failures

## 📊 Data Availability

### Format Timeline:
- **2022-01-01 onwards**: CSV format ✅ Available
- **2019-2021**: NOT available via current API (returns "No data" errors)

### Current Data Coverage:
| Data Type | Available From | Format | Status |
|-----------|---------------|--------|--------|
| DA Nodal LMPs | 2022-01-01 | CSV | ⏳ Downloading |
| RT 5-Min Nodal LMPs | 2022-01-01 | CSV | ⏸️ Pending |
| DA Ancillary Services | 2022-01-01 | CSV | ⏳ Downloading |

### Data Sizes (from testing):
- **DA Nodal**: ~67 MB/day, ~365K rows/day
- **RT 5-Min Nodal**: ~1.8 GB/day, ~10.7M rows/day
- **Ancillary Services**: ~68 KB/day, ~600 rows/day

### Storage Estimates (2022-2025):
- **DA Nodal**: ~90 GB for ~1,400 days
- **RT 5-Min Nodal**: ~2.5 TB for ~1,400 days
- **Ancillary Services**: ~95 MB total

## ⏳ Current Downloads

### In Progress:
1. **DA Nodal + AS (2023-01-01 to 2025-10-24)**
   - Log: `/home/enrico/logs/caiso_csv_2023_2025.log`
   - Progress: Downloading day by day with 6-second delays
   - No 429 errors observed

### Pending:
2. **DA Nodal + AS (2022-01-01 to 2022-12-31)** - Fill 2022 gap
3. **RT 5-Min Nodal (2022-01-01 to 2025-10-24)** - All available data

## 🔧 Next Steps

1. ✅ Complete DA Nodal + AS downloads (2023-2025) - In Progress
2. ⏳ Download 2022 data to complete CSV coverage
3. ⏳ Download RT 5-Min data (2022-2025) - Very large (~2.5 TB)
4. ⏳ Investigate alternative sources for 2019-2021 historical data
5. ⏳ Design unified Parquet format

## ⚠️ Important Notes

### Historical Data (2019-2021)
The CAISO OASIS API does not provide data before 2022-01-01 via the current query methods. Options:
1. **Accept 2022+ coverage** (3 years of data)
2. **Use CAISO Bulk Download** - Check `oasis-bulk.caiso.com` for historical archives
3. **Contact CAISO** - Request access to older data through their developer portal

### Rate Limiting
- MUST maintain 6+ second delays between requests
- 429 errors occur if too aggressive
- Current implementation is stable

### RT 5-Min Data Size
- ~1.8 GB per day is VERY LARGE
- Full download (2022-2025) will be ~2.5 TB
- Consider downloading incrementally or using compression

## 📝 Files Created

### Scripts:
- `caiso_api_client.py` - API client with throttling
- `download_historical_da_lmps.py` - DA historical downloader
- `download_historical_rt_5min_lmps.py` - RT 5-min downloader
- `download_historical_ancillary.py` - AS downloader
- `update_caiso_with_resume.py` - Daily updater

### Cron Jobs:
- `update_caiso_cron.sh` - Daily update wrapper
- `setup_caiso_cron.sh` - Cron installation script

### Documentation:
- `CAISO_STATUS.md` - This file

## 🎯 Production Readiness

- ✅ Cron job configured and tested
- ✅ Auto-resume capability working
- ✅ Rate limiting stable (no 429 errors)
- ✅ Exponential backoff retry implemented
- ⏳ Historical backfill in progress (2022-2025)

---

**Next Action**: Monitor current downloads, then start 2022 backfill and RT 5-min downloads.

# ISO Data Pipeline - Auto-Resume Capabilities

**Last Updated:** 2025-10-11

## Summary

All actively scheduled ISOs now have **smart auto-resume** capability that automatically:
- Finds the last downloaded date for each data type
- Resumes from that point
- Catches up any gaps if cron jobs fail for a few days
- No manual intervention needed!

---

## ✅ ERCOT - Auto-Resume ENABLED

**Script:** `update_all_datasets.py`
**Cron Schedule:** Twice daily (2:00 AM, 2:00 PM)

### How It Works:

```python
def get_latest_date_from_parquet(dataset: str) -> pd.Timestamp:
    """Get the latest date in the existing parquet file."""
    config = DATASET_CONFIG[dataset]
    parquet_file = config["parquet_dir"] / f"{CURRENT_YEAR}.parquet"

    # Read parquet and find max date
    df = pq.read_table(str(parquet_file)).to_pandas()
    latest_date = df[date_col].max()
    return latest_date
```

### Data Types with Auto-Resume:
1. ✅ DA Prices (Day-Ahead Settlement Point Prices)
2. ✅ AS Prices (Ancillary Service Prices)
3. ✅ DAM Gen Resources (60-day disclosure)
4. ✅ SCED Gen Resources (60-day disclosure)

### Benefits:
- Checks parquet files for latest date
- Automatically resumes from `latest_date + 1 day`
- If parquet doesn't exist, uses fallback date (2023-12-11)
- Perfect for handling API outages or missed cron runs

---

## ✅ MISO - Auto-Resume ENABLED

**Script:** `iso_markets/miso/update_miso_with_resume.py`
**Cron Schedule:** Daily at 3:00 AM

### How It Works:

```python
def find_last_date(data_type: str) -> Optional[datetime]:
    """Find the most recent date downloaded for a data type."""
    # Get all CSV files in data directory
    csv_files = list(data_dir.glob(pattern))

    # Extract dates from filenames (YYYYMMDD or YYYY-MM-DD)
    dates = [extract_date_from_filename(f) for f in csv_files]

    # Return latest date found
    return max(dates) if dates else None
```

### Data Types with Auto-Resume:
1. ✅ LMP Day-Ahead Ex-Post
2. ✅ LMP Real-Time Final
3. ✅ RT 5-min LMP (Hubs)
4. ✅ Ancillary Services DA
5. ✅ Ancillary Services RT
6. ✅ Load Data
7. ✅ EIA Fuel Mix

### Benefits:
- Scans CSV directory for each data type
- Finds latest date from filename patterns
- Resumes from `latest_date + 1 day`
- If no files found, downloads last 7 days (configurable)
- Each data type tracks independently (handles partial failures)

### Usage Examples:

```bash
# Auto-resume (default) - finds last date and continues
uv run python iso_markets/miso/update_miso_with_resume.py

# Dry run - see what would be downloaded
uv run python iso_markets/miso/update_miso_with_resume.py --dry-run

# Force start date
uv run python iso_markets/miso/update_miso_with_resume.py --start-date 2024-01-01

# Update specific data types only
uv run python iso_markets/miso/update_miso_with_resume.py --data-types lmp_da_expost fuel_mix
```

---

## ✅ NYISO - Auto-Resume ENABLED

**Script:** `download_nyiso_gridstatus.py`
**Cron Schedule:** Daily at 4:00 AM

### How It Works:

```python
def find_last_date(self, data_type: str) -> datetime:
    """Find the most recent date downloaded for a data type."""
    type_dir = self.csv_dir / data_type

    dates = []
    for csv_file in type_dir.glob("*.csv"):
        # Extract date from filename (format: YYYY-MM-DD_*.csv)
        date_str = csv_file.stem.split('_')[0]
        date = datetime.strptime(date_str, '%Y-%m-%d')
        dates.append(date)

    return max(dates) if dates else None

def auto_resume(self, fallback_start: datetime, end_date: datetime):
    """Auto-resume from the last downloaded date."""
    # Find earliest last date across all data types
    last_dates = [find_last_date(dt) for dt in data_types]
    resume_date = min(last_dates) + timedelta(days=1)

    self.download_date_range(resume_date, end_date)
```

### Data Types with Auto-Resume:
1. ✅ LMP Day-Ahead Hourly
2. ✅ LMP Real-Time 5-min
3. ✅ Ancillary Services DA
4. ✅ Ancillary Services RT
5. ✅ Load (5-min actual)
6. ✅ Fuel Mix

### Benefits:
- Uses gridstatus library (handles all API complexity)
- Finds last date across ALL data types
- Uses earliest last date to ensure all types stay in sync
- Automatic retry logic built into gridstatus
- 99.76% data completeness (2019-present)

### Usage Examples:

```bash
# Auto-resume (default)
uv run python download_nyiso_gridstatus.py --auto-resume

# Specify date range
uv run python download_nyiso_gridstatus.py --start-date 2024-01-01 --end-date 2024-01-31
```

---

## ✅ SPP - Auto-Resume ENABLED

**Script:** `iso_markets/spp/update_spp_with_resume.py`
**Cron Schedule:** Daily at 5:00 AM

### How It Works:

```python
def find_last_date(data_type: str) -> Optional[datetime]:
    """Find the most recent date downloaded for a data type."""
    # Get all CSV files in data directory
    csv_files = list(data_dir.glob(pattern))

    # Extract dates from filenames (YYYYMMDD pattern)
    dates = [extract_date_from_filename(f) for f in csv_files]

    # Return latest date found
    return max(dates) if dates else None
```

### Data Types with Auto-Resume:
1. ✅ LMP Day-Ahead
2. ✅ LMP Real-Time (daily aggregated)
3. ✅ Ancillary Services DA MCP
4. ✅ Ancillary Services RT MCP (daily)

### Benefits:
- Direct downloads from SPP portal API (bypasses gridstatus 404 errors)
- Scans CSV directory for each data type
- Finds latest date from filename patterns
- Resumes from `latest_date + 1 day`
- If no files found, downloads last 7 days (configurable)
- Each data type tracks independently (handles partial failures)

### Usage Examples:

```bash
# Auto-resume (default) - finds last date and continues
uv run python iso_markets/spp/update_spp_with_resume.py

# Dry run - see what would be downloaded
uv run python iso_markets/spp/update_spp_with_resume.py --dry-run

# Force start date
uv run python iso_markets/spp/update_spp_with_resume.py --start-date 2024-01-01

# Update specific data types only
uv run python iso_markets/spp/update_spp_with_resume.py --data-types lmp_da as_da_mcp
```

---

## ⚠️ PJM - No Auto-Resume (Not Scheduled)

**Script:** `iso_markets/pjm/daily_update_pjm.py`
**Cron Schedule:** Not currently scheduled

### Current Behavior:
- Downloads yesterday's data by default
- Can specify `--date` or `--days-back`
- **Does NOT** check for existing data
- **Does NOT** auto-resume from gaps

### To Add Auto-Resume (if needed):
Would require creating a wrapper script similar to MISO's `update_miso_with_resume.py` that:
1. Scans CSV directories for each hub
2. Finds latest date
3. Resumes from that point

**Note:** PJM not currently scheduled in cron, so auto-resume less critical.

---

## Cron Schedule Summary

| Time | ISO | Script | Auto-Resume |
|------|-----|--------|-------------|
| 2:00 AM | ERCOT | `update_ercot_cron.sh` → `update_all_datasets.py` | ✅ YES |
| 3:00 AM | Weather | `update_weather_data_cron.sh` | N/A |
| 3:00 AM | MISO | `update_miso_cron.sh` → `update_miso_with_resume.py` | ✅ YES |
| 4:00 AM | NYISO | `update_nyiso_cron.sh` → `download_nyiso_gridstatus.py` | ✅ YES |
| 5:00 AM | SPP | `update_spp_cron.sh` → `iso_markets/spp/update_spp_with_resume.py` | ✅ YES |
| 2:00 PM | ERCOT | `update_ercot_cron.sh` → `update_all_datasets.py` | ✅ YES |

---

## Key Features Comparison

| Feature | ERCOT | MISO | NYISO | SPP | PJM |
|---------|-------|------|-------|-----|-----|
| **Auto-Resume** | ✅ | ✅ | ✅ | ✅ | ❌ |
| **Gap Detection** | ✅ | ✅ | ✅ | ✅ | ❌ |
| **Multi-Day Catchup** | ✅ | ✅ | ✅ | ✅ | Manual |
| **Per-Type Tracking** | ✅ | ✅ | ✅ | ✅ | N/A |
| **Dry Run Mode** | ✅ | ✅ | ❌ | ✅ | ❌ |
| **Data Source** | Parquet files | CSV files | CSV files | CSV files | CSV files |
| **Fallback Strategy** | Fixed date | Last N days | Required param | Last N days | Yesterday only |

---

## Benefits of Auto-Resume

### 1. **Resilience to Failures**
If a cron job fails (network issue, API downtime, server maintenance):
- Next run automatically catches up
- No manual intervention needed
- No gaps in historical data

### 2. **Flexible Scheduling**
- Can run cron job less frequently without losing data
- Can manually stop/start without tracking dates
- Easy to recover from extended outages

### 3. **Per-Type Independence**
- Each data type tracks separately
- If one type fails, others continue
- Next run catches up only what's missing

### 4. **Efficient Updates**
- Only downloads new data
- Skips existing files automatically
- No duplicate downloads

---

## Testing Auto-Resume

### ERCOT:
```bash
# Dry run
uv run python update_all_datasets.py --dry-run

# Force specific date
uv run python update_all_datasets.py --start-date 2025-08-20
```

### MISO:
```bash
# Dry run
uv run python iso_markets/miso/update_miso_with_resume.py --dry-run

# Check resume points
uv run python iso_markets/miso/update_miso_with_resume.py --dry-run

# Force update specific types
uv run python iso_markets/miso/update_miso_with_resume.py --data-types lmp_da_expost lmp_rt_final
```

### NYISO:
```bash
# Auto-resume
uv run python download_nyiso_gridstatus.py --auto-resume

# Force date range
uv run python download_nyiso_gridstatus.py --start-date 2024-01-01
```

### SPP:
```bash
# Dry run
uv run python iso_markets/spp/update_spp_with_resume.py --dry-run

# Auto-resume (default)
uv run python iso_markets/spp/update_spp_with_resume.py

# Force start date
uv run python iso_markets/spp/update_spp_with_resume.py --start-date 2024-01-01
```

---

## Monitoring Auto-Resume

### Check Logs:
```bash
# ERCOT
tail -f /home/enrico/logs/ercot_update_latest.log

# MISO
tail -f /home/enrico/logs/miso_update_latest.log

# NYISO
tail -f /home/enrico/logs/nyiso_update_latest.log

# SPP
tail -f /home/enrico/logs/spp_update_latest.log
```

### Look For:
- "Latest date = YYYY-MM-DD" - Last downloaded date found
- "Resuming from YYYY-MM-DD" - Resume point calculated
- "Need to download X days" - Gap size
- "Already up to date" - No work needed

---

## Manual Recovery

If you need to manually fix gaps:

### ERCOT:
```bash
uv run python update_all_datasets.py --start-date 2024-08-01 --datasets DA_prices
```

### MISO:
```bash
uv run python iso_markets/miso/update_miso_with_resume.py \
  --start-date 2024-08-01 \
  --data-types lmp_da_expost ancillary_da
```

### NYISO:
```bash
uv run python download_nyiso_gridstatus.py \
  --start-date 2024-08-01 \
  --end-date 2024-08-15
```

### SPP:
```bash
uv run python iso_markets/spp/update_spp_with_resume.py \
  --start-date 2024-08-01 \
  --data-types lmp_da as_da_mcp
```

---

## Summary

✅ **All actively scheduled ISOs (ERCOT, MISO, NYISO, SPP) now have smart auto-resume!**

This means:
- Daily cron jobs automatically catch up from last downloaded date
- Resilient to failures and network issues
- No manual date tracking needed
- Gaps are automatically filled on next run
- **SPP downloads bypass gridstatus 404 errors** by using direct portal API

**Last updated:** 2025-10-11 by Claude Code

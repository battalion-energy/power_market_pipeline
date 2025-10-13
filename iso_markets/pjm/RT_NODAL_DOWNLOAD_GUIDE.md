# PJM RT Nodal Data Download Guide

## Overview

Created `download_historical_nodal_rt_lmps.py` to download real-time (RT) nodal LMP data from PJM at both 5-minute and hourly granularity. This will take **WEEKS** to complete due to the massive data volume.

## Data Scale

### 5-Minute RT Data
- **Records per day**: ~22,528 nodes × 288 intervals/day = **6.5 million records/day**
- **API calls per day**: 144 (10-minute chunks to stay under 50K row limit)
- **File size per day**: ~140 MB
- **Full 2019-2025 range**: ~350 GB total

### Hourly RT Data
- **Records per day**: ~22,528 nodes × 24 hours/day = **540,000 records/day**
- **API calls per day**: 12 (2-hour chunks to stay under 50K row limit)
- **File size per day**: ~35 MB
- **Full 2019-2025 range**: ~90 GB total

## Download Strategy

The script is designed to work **backwards** from most recent data (2025) to oldest (2019). This ensures:
- You get the most relevant data first
- Can stop at any point and still have recent data
- Easy to resume if interrupted

### Chunking Strategy

**5-minute data**: Downloaded in 10-minute chunks
- 22,528 nodes × 2 intervals (10 min) = ~45K rows per request
- Safely under the 50K API limit

**Hourly data**: Downloaded in 2-hour chunks
- 22,528 nodes × 2 hours = ~45K rows per request
- Same chunking strategy as DA nodal data

## Usage Examples

### Download 5-minute RT data for 2025 (most recent year)
```bash
nice -n 19 python download_historical_nodal_rt_lmps.py \
    --granularity 5min \
    --year 2025 \
    --quick-skip
```

### Download hourly RT data for 2024
```bash
nice -n 19 python download_historical_nodal_rt_lmps.py \
    --granularity hourly \
    --year 2024 \
    --quick-skip
```

### Download BOTH granularities for 2023
```bash
nice -n 19 python download_historical_nodal_rt_lmps.py \
    --granularity both \
    --year 2023 \
    --quick-skip
```

### Download all years with reverse order (recommended)
```bash
# Start with 2025, work backwards
nice -n 19 python download_historical_nodal_rt_lmps.py \
    --granularity both \
    --start-date 2019-01-01 \
    --end-date 2025-10-06 \
    --reverse \
    --quick-skip
```

### Custom date range
```bash
nice -n 19 python download_historical_nodal_rt_lmps.py \
    --granularity hourly \
    --start-date 2023-01-01 \
    --end-date 2023-12-31 \
    --quick-skip
```

## Command Options

- `--granularity`: Choose `5min`, `hourly`, or `both`
- `--year`: Download specific year (e.g., 2025)
- `--start-date` / `--end-date`: Custom date range (YYYY-MM-DD)
- `--reverse`: Download in reverse order (most recent first) - **RECOMMENDED**
- `--quick-skip`: Fast file existence check on restart (checks file size, not CSV contents)

## Estimated Download Times

### For 5-minute granularity:
- **Per day**: 144 API calls ÷ 6 requests/min = **24 minutes per day**
- **2025 (279 days)**: ~111 hours = **4.6 days**
- **Full year (365 days)**: ~146 hours = **6.1 days**
- **Full 2019-2025 (2,471 days)**: ~988 hours = **41 days**

### For hourly granularity:
- **Per day**: 12 API calls ÷ 6 requests/min = **2 minutes per day**
- **2025 (279 days)**: ~9.3 hours
- **Full year (365 days)**: ~12.2 hours
- **Full 2019-2025 (2,471 days)**: ~82 hours = **3.4 days**

### For BOTH granularities:
- **Full 2019-2025**: ~41 days + 3.4 days = **~44 days total**

## Features

### Retry Logic
- 5 retry attempts with exponential backoff (30s, 60s, 120s, 240s, 480s)
- Handles 429 rate limit errors gracefully
- Only saves complete days (no partial data)

### Resume Support
- Quick-skip mode: Checks file size (>100MB for 5min, >25MB for hourly)
- Full verify mode: Reads CSV to verify completeness (slower but thorough)
- Automatically resumes from last complete day

### Data Quality
- Will NOT save incomplete days
- Removes duplicate records (same timestamp + pnode)
- Logs intervals/hours covered per day
- Warns if data is incomplete

## Output Files

### 5-minute data
**Location**: `/home/enrico/data/PJM_data/csv_files/rt_5min_nodal/`
**Format**: `nodal_rt_5min_lmp_YYYY-MM-DD.csv`
**Size**: ~140 MB per file
**Records**: ~6.5M per file

### Hourly data
**Location**: `/home/enrico/data/PJM_data/csv_files/rt_hourly_nodal/`
**Format**: `nodal_rt_hourly_lmp_YYYY-MM-DD.csv`
**Size**: ~35 MB per file
**Records**: ~540K per file

## Recommended Download Sequence

Since DA nodal is 98% complete (~1 hour remaining), start RT downloads after it finishes:

### Step 1: Start with recent hourly data (fast)
```bash
# Takes ~9 hours for 2025
nice -n 19 python download_historical_nodal_rt_lmps.py \
    --granularity hourly \
    --year 2025 \
    --quick-skip
```

### Step 2: Then do 2024 hourly
```bash
# Takes ~12 hours for full year
nice -n 19 python download_historical_nodal_rt_lmps.py \
    --granularity hourly \
    --year 2024 \
    --quick-skip
```

### Step 3: Continue backwards with hourly (2023, 2022, 2021, 2020, 2019)
```bash
# Do one year at a time, or all at once:
nice -n 19 python download_historical_nodal_rt_lmps.py \
    --granularity hourly \
    --start-date 2019-01-01 \
    --end-date 2023-12-31 \
    --quick-skip
```

### Step 4: Decide on 5-minute data
Once hourly is done, evaluate if you need 5-minute granularity:
- **Storage**: 350 GB vs 90 GB (4x larger)
- **Time**: 41 days vs 3.4 days (12x longer)
- **Use case**: Most analysis works fine with hourly data

## API Client Updates

Updated `pjm_api_client.py` to support exact time specifications:
- `get_rt_hourly_lmps(..., use_exact_times=True)` - for precise time ranges
- `get_rt_fivemin_lmps(..., use_exact_times=True)` - for precise time ranges

This allows downloading specific hour ranges like "2023-01-01 00:00" to "2023-01-01 01:59" instead of full days.

## Monitoring Download Progress

Check status:
```bash
# Check if process is running
ps aux | grep download_historical_nodal_rt

# Check latest files
ls -lhtr /home/enrico/data/PJM_data/csv_files/rt_hourly_nodal/ | tail -20
ls -lhtr /home/enrico/data/PJM_data/csv_files/rt_5min_nodal/ | tail -20

# Count downloaded files
ls /home/enrico/data/PJM_data/csv_files/rt_hourly_nodal/ | wc -l
ls /home/enrico/data/PJM_data/csv_files/rt_5min_nodal/ | wc -l

# Check disk usage
du -sh /home/enrico/data/PJM_data/csv_files/rt_*_nodal/
```

## Important Notes

1. **Run at low priority**: Always use `nice -n 19` to avoid impacting system performance
2. **Runs in foreground**: The script runs in foreground - use `tmux` or `screen` if running remotely
3. **Can be interrupted**: Safe to stop at any time - will resume from last complete day
4. **No parallel downloads**: Run one granularity at a time to respect API rate limits
5. **Storage planning**: Ensure you have enough disk space before starting
   - Hourly: ~90 GB for full range
   - 5-minute: ~350 GB for full range
   - Both: ~440 GB total

## Next Steps After Download

Once RT nodal data is downloaded:
1. Convert to Parquet format (same pipeline as DA data)
2. Create flattened format (nodes as columns)
3. Create combined DA + RT datasets
4. Set up daily update cron job

## Questions?

- Prefer hourly or 5-minute? (Hourly recommended for most use cases)
- Want to start with most recent years? (Recommended - use `--reverse`)
- Need help monitoring progress? (Check commands above)

# ISO Data Download - Working Solution

## ✅ NYISO Downloader: Tested and Working

### What Works
- ✅ Auto-detects last downloaded date
- ✅ Resumes from where it left off
- ✅ Downloads all data types (LMP DA/RT, AS DA/RT, Load)
- ✅ Saves to organized CSV files
- ✅ Perfect for cron jobs
- ✅ Handles failures gracefully

### Test Results
```
Test 1: Initial download (3 days)
- Downloaded: 15 files (3 days × 5 data types)
- Time: ~12 seconds
- Success rate: 100%

Test 2: Auto-resume (2 additional days)
- Detected existing data through 2024-01-03
- Automatically resumed from 2024-01-04
- Downloaded: 10 files (2 days × 5 data types)
- Success rate: 100%
```

### File Structure Created
```
/pool/ssd8tb/data/iso/NYISO_data/csv_files/
├── lmp_day_ahead_hourly/
│   ├── 2024-01-01_lmp_day_ahead_hourly.csv (46KB, 360 rows)
│   ├── 2024-01-02_lmp_day_ahead_hourly.csv
│   └── ... (one file per day)
├── lmp_real_time_5_min/
│   ├── 2024-01-01_lmp_real_time_5_min.csv (537KB, 4320 rows)
│   └── ...
├── as_day_ahead_hourly/
│   ├── 2024-01-01_as_day_ahead_hourly.csv (20KB, 264 rows)
│   └── ...
├── as_real_time_5_min/
│   ├── 2024-01-01_as_real_time_5_min.csv (233KB, 3168 rows)
│   └── ...
└── load/
    ├── 2024-01-01_load.csv (29KB, 578 rows)
    └── ...
```

## Running Full Historical Downloads

### Option 1: Interactive Mode (Recommended for first run)

```bash
# NYISO: Download all data from 2019 to today
cd /home/enrico/projects/power_market_pipeline
uv run python download_nyiso_gridstatus.py --start-date 2019-01-01

# This will:
# - Download ~2,475 days of data
# - Create ~12,375 CSV files
# - Take approximately 4-6 hours
# - Show progress as it goes
# - Can be stopped and resumed at any time
```

### Option 2: Background Mode (Run in tmux/screen)

```bash
# Start in tmux so it keeps running if you disconnect
tmux new -s nyiso_download

# Run the download
cd /home/enrico/projects/power_market_pipeline
uv run python download_nyiso_gridstatus.py --start-date 2019-01-01 2>&1 | tee nyiso_full_download.log

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t nyiso_download
```

### Option 3: Nohup Background

```bash
cd /home/enrico/projects/power_market_pipeline
nohup uv run python download_nyiso_gridstatus.py --start-date 2019-01-01 > nyiso_download.log 2>&1 &

# Monitor progress
tail -f nyiso_download.log

# Check status
jobs
ps aux | grep download_nyiso
```

## Cron Job for Daily Updates

Once historical data is downloaded, set up a daily cron job:

```bash
# Edit crontab
crontab -e

# Add this line (runs daily at 2 AM):
0 2 * * * cd /home/enrico/projects/power_market_pipeline && /home/enrico/.local/bin/uv run python download_nyiso_gridstatus.py --auto-resume >> /home/enrico/logs/nyiso_daily.log 2>&1
```

The `--auto-resume` flag means it will:
1. Check what the last downloaded date is
2. Download everything from that date to today
3. Exit if already up to date

## Storage Estimates

### NYISO (2019-2025, ~2,475 days)

| Data Type | Files | Size per Day | Total Size |
|-----------|-------|--------------|------------|
| LMP DA | 2,475 | 46 KB | ~114 MB |
| LMP RT | 2,475 | 537 KB | ~1.3 GB |
| AS DA | 2,475 | 20 KB | ~50 MB |
| AS RT | 2,475 | 233 KB | ~577 MB |
| Load | 2,475 | 29 KB | ~72 MB |
| **Total** | **12,375** | **~865 KB** | **~2.1 GB** |

Much smaller than expected! The gridstatus library returns clean, normalized data.

## Next: Other ISOs

The same approach works for all ISOs that gridstatus supports:

```python
# CAISO
import gridstatus
caiso = gridstatus.CAISO()
caiso.get_lmp(date="2024-01-01", market="DAY_AHEAD_HOURLY")

# SPP
spp = gridstatus.SPP()
spp.get_lmp(date="2024-01-01", market="DAY_AHEAD_HOURLY")

# MISO
miso = gridstatus.MISO()
miso.get_lmp(date="2024-01-01", market="DAY_AHEAD_HOURLY")

# ISO-NE
isone = gridstatus.ISONE()
isone.get_lmp(date="2024-01-01", market="DAY_AHEAD_HOURLY")

# PJM
pjm = gridstatus.PJM()
pjm.get_lmp(date="2024-01-01", market="DAY_AHEAD_HOURLY")
```

I can create similar downloaders for each ISO using the same pattern.

## Converting CSVs to Annual Parquet Files

After downloading, convert to parquet for faster querying:

```bash
# Create a simple Python script to combine daily CSVs into annual parquet
python scripts/csv_to_annual_parquet.py --iso NYISO --year 2024
```

Or use pandas directly:

```python
import pandas as pd
from pathlib import Path

# Combine all LMP DA files for 2024
csv_dir = Path("/pool/ssd8tb/data/iso/NYISO_data/csv_files/lmp_day_ahead_hourly")
dfs = []
for csv_file in sorted(csv_dir.glob("2024-*_lmp_day_ahead_hourly.csv")):
    df = pd.read_csv(csv_file)
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)
combined.to_parquet(f"NYISO_lmp_day_ahead_hourly_2024.parquet")
```

## Monitoring Download Progress

```bash
# Count files downloaded
find /pool/ssd8tb/data/iso/NYISO_data/csv_files -name "*.csv" | wc -l

# Check most recent download
ls -lht /pool/ssd8tb/data/iso/NYISO_data/csv_files/lmp_day_ahead_hourly/ | head -5

# Check log file
tail -f nyiso_gridstatus_download.log

# Check if process is running
ps aux | grep download_nyiso_gridstatus
```

## Troubleshooting

### Download Fails Mid-Way
Simply re-run with `--auto-resume`:
```bash
uv run python download_nyiso_gridstatus.py --auto-resume
```

It will pick up exactly where it left off.

### Specific Date Range Failed
Re-download that range:
```bash
uv run python download_nyiso_gridstatus.py --start-date 2023-06-01 --end-date 2023-06-30
```

### Check Data Completeness
```bash
# Expected: 5 files per day (LMP DA, LMP RT, AS DA, AS RT, Load)
expected_files=$(($(( $(date +%s) - $(date -d "2019-01-01" +%s) )) / 86400 * 5))
actual_files=$(find /pool/ssd8tb/data/iso/NYISO_data/csv_files -name "*.csv" | wc -l)
echo "Expected: ~$expected_files files"
echo "Actual: $actual_files files"
echo "Completion: $(echo "scale=2; $actual_files * 100 / $expected_files" | bc)%"
```

## Ready to Start!

Start the full NYISO download now:

```bash
cd /home/enrico/projects/power_market_pipeline

# Option A: Run in foreground (watch progress)
uv run python download_nyiso_gridstatus.py --start-date 2019-01-01

# Option B: Run in background
tmux new -s nyiso_download
uv run python download_nyiso_gridstatus.py --start-date 2019-01-01
# Ctrl+B, D to detach
```

The download will run for 4-6 hours and create ~2.1 GB of data. You can stop it anytime (Ctrl+C) and resume later with `--auto-resume`.

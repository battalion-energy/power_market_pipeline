# ERCOT BESS Data Cron Job Setup

## Overview

Automated system to maintain a gap-free catalog of ERCOT Battery Energy Storage System (BESS) dispatch data.

## Features

- ✅ Downloads latest data every 5 minutes
- ✅ Maintains gap-free historical catalog
- ✅ Automatically deduplicates records
- ✅ Detects and reports data gaps
- ✅ Low priority execution (nice -n 19)
- ✅ Lock file prevents concurrent runs
- ✅ Comprehensive logging
- ✅ Handles API failures gracefully

## Data Source

**API Endpoint**: `https://www.ercot.com/api/1/services/read/dashboards/energy-storage-resources.json`

**Data Available**:
- Previous day + current day (rolling 48-hour window)
- 5-minute interval resolution
- No authentication required (public API)

**Fields**:
- `timestamp`: Datetime with timezone (Central Time)
- `total_charging_mw`: Total MW charging (negative values)
- `total_discharging_mw`: Total MW discharging (positive values)
- `net_output_mw`: Net output MW (discharging - charging)

## Installation

### Option 1: Automated Setup

```bash
bash setup_bess_cron.sh
```

### Option 2: Manual Setup

```bash
# Make script executable
chmod +x ercot_bess_cron_updater.py

# Add to crontab
crontab -e

# Add this line:
*/5 * * * * nice -n 19 python3 /home/enrico/projects/power_market_pipeline/ercot_bess_cron_updater.py >> /home/enrico/projects/power_market_pipeline/ercot_battery_storage_data/bess_updater.log 2>&1
```

## File Locations

| File | Path | Purpose |
|------|------|---------|
| **Script** | `ercot_bess_cron_updater.py` | Main updater script |
| **Data Catalog** | `ercot_battery_storage_data/bess_catalog.csv` | Gap-free data catalog |
| **Log File** | `ercot_battery_storage_data/bess_updater.log` | Execution logs |
| **Lock File** | `ercot_battery_storage_data/.bess_updater.lock` | Prevents concurrent runs |

## Monitoring

### View Logs

```bash
# Real-time log monitoring
tail -f ercot_battery_storage_data/bess_updater.log

# Last 100 lines
tail -100 ercot_battery_storage_data/bess_updater.log

# Search for errors
grep ERROR ercot_battery_storage_data/bess_updater.log
```

### Check Data Status

```bash
# View latest records
tail ercot_battery_storage_data/bess_catalog.csv

# Count total records
wc -l ercot_battery_storage_data/bess_catalog.csv

# Quick stats with Python
python3 -c "
import pandas as pd
df = pd.read_csv('ercot_battery_storage_data/bess_catalog.csv', parse_dates=['timestamp'])
print(f'Total records: {len(df)}')
print(f'Date range: {df[\"timestamp\"].min()} to {df[\"timestamp\"].max()}')
print(f'Latest: {df[\"timestamp\"].iloc[-1]}')
"
```

### Verify Cron Job

```bash
# List all cron jobs
crontab -l

# Check if BESS cron is running
crontab -l | grep bess

# View cron execution logs
grep CRON /var/log/syslog | grep bess_cron_updater
```

## How It Works

### 1. Acquisition Lock

Prevents multiple simultaneous executions:
- Creates `.bess_updater.lock` with process ID
- Checks for stale locks (>10 minutes old)
- Automatically removes stale locks

### 2. API Data Fetch

Downloads latest 48-hour window:
- Fetches from ERCOT API (30-second timeout)
- Extracts `previousDay` and `currentDay` data
- Standardizes column names
- Validates data structure

### 3. Data Merge

Intelligently merges new with existing:
- Loads existing catalog from CSV
- Concatenates old and new data
- Removes duplicates (keeps latest)
- Sorts by timestamp

### 4. Gap Detection

Identifies missing time intervals:
- Expected interval: 5 minutes
- Tolerance: ±30 seconds
- Reports gap count and duration
- Logs gap details

### 5. Data Persistence

Saves updated catalog:
- Overwrites CSV file atomically
- Maintains timezone information
- Preserves data integrity

### 6. Statistics & Logging

Reports comprehensive metrics:
- Total records
- Date range
- Gap count and duration
- New records added
- Execution status

## Data Format

### CSV Structure

```csv
timestamp,total_charging_mw,total_discharging_mw,net_output_mw
2025-10-10 00:00:00-05:00,-691.654,23.456,-668.197
2025-10-10 00:05:00-05:00,-720.415,120.01,-600.405
...
```

### Field Descriptions

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `timestamp` | datetime | Measurement time (CT) | `2025-10-10 00:00:00-05:00` |
| `total_charging_mw` | float | Total charging power (negative) | `-691.654` |
| `total_discharging_mw` | float | Total discharging power (positive) | `23.456` |
| `net_output_mw` | float | Net output (discharge - charge) | `-668.197` |

## Typical Operation Patterns

### Normal Operation

```
2025-10-11 11:26:18,283 - INFO - ERCOT BESS Data Updater Starting
2025-10-11 11:26:18,398 - INFO - Fetched 425 records from API
2025-10-11 11:26:18,399 - INFO - Merging data...
2025-10-11 11:26:18,441 - INFO - Total records: 425
2025-10-11 11:26:18,441 - INFO - Gaps: 0 (0 minutes total)
2025-10-11 11:26:18,441 - INFO - Added 12 new records
2025-10-11 11:26:18,441 - INFO - Update completed successfully
```

### Concurrent Run Blocked

```
2025-10-11 11:31:00,123 - WARNING - Another instance is running (lock file exists)
2025-10-11 11:31:00,123 - ERROR - Could not acquire lock, exiting
```

### Gap Detected

```
2025-10-11 11:36:22,456 - WARNING - Found 2 gaps in data:
2025-10-11 11:36:22,456 - WARNING -   Gap: 2025-10-11 08:15:00-05:00 to 2025-10-11 08:45:00-05:00 (30 minutes)
2025-10-11 11:36:22,456 - WARNING -   Gap: 2025-10-11 09:20:00-05:00 to 2025-10-11 09:25:00-05:00 (5 minutes)
```

## Performance

- **Execution time**: ~0.5 seconds per run
- **CPU usage**: Minimal (nice -n 19 priority)
- **Memory usage**: <50 MB
- **Network**: 1 API call per run (~20 KB download)
- **Disk I/O**: Minimal (append to CSV)

## Troubleshooting

### Script Not Running

```bash
# Check if cron daemon is running
systemctl status cron

# Check cron logs
grep CRON /var/log/syslog | tail -20

# Manually test script
python3 ercot_bess_cron_updater.py
```

### API Errors

```bash
# Test API manually
curl -s "https://www.ercot.com/api/1/services/read/dashboards/energy-storage-resources.json" | jq .

# Check network connectivity
ping -c 3 www.ercot.com
```

### Lock File Issues

```bash
# Remove stuck lock file
rm ercot_battery_storage_data/.bess_updater.lock

# Check lock file age
ls -lh ercot_battery_storage_data/.bess_updater.lock
```

### Permission Issues

```bash
# Ensure script is executable
chmod +x ercot_bess_cron_updater.py

# Check file permissions
ls -l ercot_battery_storage_data/bess_catalog.csv
```

## Maintenance

### Rotate Logs

Add to crontab to rotate logs monthly:

```bash
0 0 1 * * mv /home/enrico/projects/power_market_pipeline/ercot_battery_storage_data/bess_updater.log /home/enrico/projects/power_market_pipeline/ercot_battery_storage_data/bess_updater_$(date +\%Y\%m).log && touch /home/enrico/projects/power_market_pipeline/ercot_battery_storage_data/bess_updater.log
```

### Backup Data

```bash
# Create backup
cp ercot_battery_storage_data/bess_catalog.csv ercot_battery_storage_data/bess_catalog_$(date +%Y%m%d).csv.bak
```

### Remove Cron Job

```bash
# List current cron jobs
crontab -l

# Remove BESS cron job
crontab -l | grep -v bess_cron_updater | crontab -
```

## Integration

### Python

```python
import pandas as pd

# Load data
df = pd.read_csv('ercot_battery_storage_data/bess_catalog.csv',
                 parse_dates=['timestamp'])

# Analysis
print(f"Total records: {len(df)}")
print(f"Average net output: {df['net_output_mw'].mean():.2f} MW")
print(f"Peak discharge: {df['total_discharging_mw'].max():.2f} MW")
print(f"Peak charge: {df['total_charging_mw'].min():.2f} MW")

# Plot
import matplotlib.pyplot as plt
df.plot(x='timestamp', y='net_output_mw', figsize=(15, 6))
plt.title('ERCOT BESS Net Output')
plt.ylabel('Net Output (MW)')
plt.show()
```

### Database Import

```sql
-- PostgreSQL example
COPY bess_data (timestamp, total_charging_mw, total_discharging_mw, net_output_mw)
FROM '/path/to/bess_catalog.csv'
WITH (FORMAT CSV, HEADER TRUE);
```

## FAQ

**Q: Why every 5 minutes?**
A: The API provides 5-minute interval data. Running more frequently provides no additional data.

**Q: What happens if I miss several hours?**
A: The API provides a 48-hour rolling window, so data is automatically backfilled.

**Q: Does it use a lot of bandwidth?**
A: No, each API call is ~20 KB. Running 24/7 uses ~6 MB/day.

**Q: What if ERCOT API is down?**
A: The script logs the error and exits gracefully. Next run will catch up.

**Q: Can I run it manually?**
A: Yes: `python3 ercot_bess_cron_updater.py`

**Q: How do I pause updates?**
A: Comment out the cron job: `crontab -e` and add `#` before the line.

## Support

For issues or questions:
1. Check logs: `tail -f ercot_battery_storage_data/bess_updater.log`
2. Test manually: `python3 ercot_bess_cron_updater.py`
3. Review this documentation
4. Check ERCOT API status

---

**Created**: October 11, 2025
**Author**: Claude Code
**Project**: Power Market Pipeline

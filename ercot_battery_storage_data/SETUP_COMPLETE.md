# ERCOT ESR Daily Data Collection - Setup Complete ✅

## System Overview

The ERCOT Energy Storage Resources (ESR) automated data collection system has been successfully installed and configured.

## What Was Installed

### 1. Download Script
**Location**: `ercot_battery_storage_data/download_esr_daily.py`

Features:
- Downloads ESR data from ERCOT API
- Saves both "previousDay" and "currentDay" data
- Creates timestamped archives
- Maintains daily files by date
- Auto-cleanup of old full archives (7 days retention)

### 2. Cron Job Setup
**Schedule**: Daily at 12:05 AM Central Time

Files:
- `/cronjobs/download_esr_daily.sh` - Cron wrapper with logging
- `/cronjobs/setup_esr_cron.sh` - Setup script

Current cron entry:
```
5 0 * * * /home/enrico/projects/power_market_pipeline/cronjobs/download_esr_daily.sh
```

### 3. Data Viewer Utility
**Location**: `ercot_battery_storage_data/view_esr_data.py`

View collected data:
```bash
python3 ercot_battery_storage_data/view_esr_data.py          # List available dates
python3 ercot_battery_storage_data/view_esr_data.py 20251108 # View daily summary
python3 ercot_battery_storage_data/view_esr_data.py 20251108 --hourly # Hourly breakdown
```

### 4. Documentation
- `ESR_DAILY_DOWNLOAD_README.md` - Complete system documentation
- `SETUP_COMPLETE.md` - This file

## Data Archive Location

```
ercot_battery_storage_data/esr_archive/
├── esr_20251108.json              # Daily file (kept forever)
├── esr_20251109.json              # Daily file (kept forever)
└── esr_full_YYYYMMDD_HHMMSS.json  # Full archives (kept 7 days)
```

## Verification

The system has been tested and verified:

✅ Download script runs successfully
✅ Data is being saved to archive
✅ Cron job added to crontab
✅ Logs are being written to `/pool/ssd8tb/logs/`
✅ Data viewer utility works correctly

Current status:
```
$ crontab -l | grep esr
5 0 * * * /home/enrico/projects/power_market_pipeline/cronjobs/download_esr_daily.sh
```

## Initial Data Collection

Data collected during setup:
- **2025-11-08**: 288 data points (complete day)
- **2025-11-09**: 187 data points (partial day, still in progress)

Example output from 2025-11-08:
```
Peak Events:
  Max Charging:    -4,893.73 MW at 2025-11-08 08:45:00-0600
  Max Discharging: 5,046.59 MW at 2025-11-08 17:25:00-0600
```

## Quick Reference

### Run Manual Download
```bash
/home/enrico/projects/power_market_pipeline/cronjobs/download_esr_daily.sh
```

### View Latest Log
```bash
tail -f /pool/ssd8tb/logs/esr_download_latest.log
```

### View Cron Status
```bash
crontab -l | grep esr
```

### List Archived Data
```bash
ls -lh ercot_battery_storage_data/esr_archive/
python3 ercot_battery_storage_data/view_esr_data.py
```

### Analyze Data
```bash
# View daily summary
python3 ercot_battery_storage_data/view_esr_data.py 20251108

# View hourly breakdown
python3 ercot_battery_storage_data/view_esr_data.py 20251108 --hourly
```

## Data Format

Each daily file contains:
```json
{
  "date": "2025-11-08 03:00:00-0600",
  "downloaded_at": "2025-11-09 15:26:01-0600",
  "download_timestamp": "20251109_153233",
  "data_points": 288,
  "data": [
    {
      "timestamp": "2025-11-08 00:00:00-0600",
      "totalCharging": -1048.823,      // MW (negative = consuming)
      "totalDischarging": 13.575,      // MW (positive = injecting)
      "netOutput": -1035.249,          // MW (sum of above)
      "epoch": 1762581600000           // Unix timestamp (ms)
    },
    // ... 287 more 5-minute intervals
  ]
}
```

## Why This Matters

ERCOT's API only provides 2 days of ESR data (yesterday and today). There is no historical archive available through their API. **This automated daily collection is the ONLY way to build a historical ESR dataset.**

## Monitoring

Logs are retained for 30 days:
- `/pool/ssd8tb/logs/esr_download_*.log`
- `/pool/ssd8tb/logs/esr_download_latest.log` (symlink)

The cron job logs:
- Start/end timestamps
- Number of data points retrieved
- File sizes
- Success/failure status
- Any errors or warnings

## Next Steps

The system will automatically:
1. Run daily at 12:05 AM to capture complete previous day data
2. Download both "previousDay" and "currentDay" datasets
3. Save to individual daily files
4. Keep full archives for 7 days
5. Keep daily files indefinitely
6. Log all operations

You now have a growing historical archive of ERCOT ESR data at 5-minute resolution!

## Integration

Use this data for:
- Battery storage revenue analysis
- Charging/discharging pattern studies
- Price correlation analysis
- Machine learning model training
- Market forecasting

Example Python integration:
```python
import json

# Load a day's data
with open('ercot_battery_storage_data/esr_archive/esr_20251108.json') as f:
    data = json.load(f)

# Process time series
for point in data['data']:
    print(f"{point['timestamp']}: Net Output = {point['netOutput']} MW")
```

## Support

For issues or questions:
- Check logs in `/pool/ssd8tb/logs/esr_download_latest.log`
- Review `ESR_DAILY_DOWNLOAD_README.md` for detailed documentation
- Run manual download to test: `/home/enrico/projects/power_market_pipeline/cronjobs/download_esr_daily.sh`

---

**Setup completed**: 2025-11-09 15:33 CST
**First data collected**: 2025-11-08 (288 points), 2025-11-09 (187 points)
**Status**: ✅ Operational

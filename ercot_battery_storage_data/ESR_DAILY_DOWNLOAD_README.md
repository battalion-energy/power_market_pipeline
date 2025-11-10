# ERCOT Energy Storage Resources (ESR) Daily Data Collection

## Overview

This system automatically downloads and archives Energy Storage Resources (ESR) data from ERCOT's dashboard API on a daily basis. The data provides 5-minute resolution metrics for battery storage operations across the ERCOT grid.

## Data Source

- **API Endpoint**: `https://www.ercot.com/api/1/services/read/dashboards/energy-storage-resources.json`
- **Dashboard**: `https://www.ercot.com/gridmktinfo/dashboards/energystorageresources`
- **Update Frequency**: Every 59 seconds (ERCOT's refresh rate)
- **Data Resolution**: 5-minute intervals (288 data points per day)

## Data Fields

Each 5-minute data point contains:

- **totalCharging** (MW): Negative values representing total power consumed from the grid by ESRs
- **totalDischarging** (MW): Positive values representing total power injected to the grid by ESRs
- **netOutput** (MW): Sum of charging and discharging (negative = net charging, positive = net discharging)
- **timestamp**: Timestamp in Central Time with DST flag
- **epoch**: Unix timestamp in milliseconds

## Files and Scripts

### Main Script
- **`download_esr_daily.py`**: Python script that downloads and archives ESR data
  - Downloads both "previousDay" and "currentDay" data
  - Saves timestamped full archives
  - Creates individual daily files by date
  - Automatically cleans up old full archives (keeps 7 days)

### Cron Job
- **`/cronjobs/download_esr_daily.sh`**: Wrapper script for cron execution
- **`/cronjobs/setup_esr_cron.sh`**: Setup script to add cron job
- **Schedule**: Daily at 12:05 AM Central Time

## Data Archive Structure

```
ercot_battery_storage_data/esr_archive/
├── esr_20251108.json              # Daily file for 2025-11-08
├── esr_20251109.json              # Daily file for 2025-11-09
├── esr_full_20251109_153233.json  # Full archive with timestamp (kept 7 days)
└── ...
```

### Daily Files (esr_YYYYMMDD.json)

These are kept indefinitely and contain:
```json
{
  "date": "2025-11-08 03:00:00-0600",
  "downloaded_at": "2025-11-09 15:26:01-0600",
  "download_timestamp": "20251109_153233",
  "data_points": 288,
  "data": [
    {
      "tagCLastTime": "2025-11-08 00:00:00",
      "dstFlag": "N",
      "totalCharging": -1048.823,
      "totalDischarging": 13.575,
      "netOutput": -1035.249,
      "timestamp": "2025-11-08 00:00:00-0600",
      "epoch": 1762581600000
    },
    ...
  ]
}
```

### Full Archive Files (esr_full_YYYYMMDD_HHMMSS.json)

These contain both previousDay and currentDay data as received from the API:
```json
{
  "lastUpdated": "2025-11-09 15:26:01-0600",
  "previousDay": {
    "dayDate": "2025-11-08 03:00:00-0600",
    "data": [...]
  },
  "currentDay": {
    "dayDate": "2025-11-09 03:00:00-0600",
    "data": [...]
  }
}
```

## Usage

### Manual Download
```bash
# Run the download script directly
python3 ercot_battery_storage_data/download_esr_daily.py

# Or use the cron wrapper
/home/enrico/projects/power_market_pipeline/cronjobs/download_esr_daily.sh
```

### Setup Cron Job
```bash
# Add to crontab (runs daily at 12:05 AM)
/home/enrico/projects/power_market_pipeline/cronjobs/setup_esr_cron.sh

# Verify cron job
crontab -l | grep esr
```

### View Logs
```bash
# View latest log
tail -f /pool/ssd8tb/logs/esr_download_latest.log

# List all ESR download logs
ls -lh /pool/ssd8tb/logs/esr_download_*.log
```

## Data Limitations

### ERCOT API Limitations
- **Only 2 days available**: The API only provides "previousDay" and "currentDay" data
- **No historical archive**: ERCOT does not provide an API endpoint for dates beyond yesterday
- **No date parameters**: The API endpoint does not accept date parameters

This is why **daily automated collection is critical** - it's the only way to build a historical archive.

### Dashboard Limitations
- The web dashboard only has two buttons: "Previous Day" and "Current Day"
- No calendar picker or date range selection
- CSV download only exports the currently displayed day

## Alternative Data Sources

For historical ESR data beyond what this script has collected:

1. **ERCOT Data Portal** (`data.ercot.com`)
   - Requires registration
   - May have historical ESR datasets

2. **60-Day SCED Reports**
   - Contains resource-level dispatch data
   - Includes ESR/battery information
   - Available through ERCOT's document system

3. **Data Aggregation Reports**
   - Available at `www.ercot.com/mktinfo/data_agg`
   - May contain aggregated ESR statistics

## Monitoring

The script logs all activities including:
- Download success/failure
- Number of data points retrieved
- File sizes
- Any errors or warnings

Logs are kept for 30 days and can be found at:
- `/pool/ssd8tb/logs/esr_download_*.log`
- `/pool/ssd8tb/logs/esr_download_latest.log` (symlink to most recent)

## Troubleshooting

### Script Fails to Download
```bash
# Check internet connectivity
curl -I https://www.ercot.com/

# Check API endpoint
curl -s 'https://www.ercot.com/api/1/services/read/dashboards/energy-storage-resources.json' | python3 -m json.tool | head
```

### Cron Job Not Running
```bash
# Check cron is running
systemctl status cron

# Check crontab entry
crontab -l | grep esr

# Check system logs for cron
journalctl -u cron | grep esr
```

### Missing Data Points
The currentDay data may have fewer than 288 points if:
- Downloaded before the day is complete
- ERCOT is experiencing data issues
- Network interruption during the day

The script will update the daily file if re-run with more complete data.

## Data Quality

- **Completeness**: Each complete day should have exactly 288 data points (24 hours × 12 five-minute intervals)
- **Updates**: The script intelligently overwrites daily files only if the new download has more data points
- **Deduplication**: Daily files prevent duplicate storage of the same date

## Integration

The archived ESR data can be used for:
- Time series analysis of battery charging/discharging patterns
- Revenue optimization for battery storage assets
- Market analysis and price correlation studies
- Machine learning model training
- Forecasting ESR behavior

Example Python code to load the data:
```python
import json
from pathlib import Path

# Load a specific day
with open('esr_archive/esr_20251108.json', 'r') as f:
    data = json.load(f)

print(f"Date: {data['date']}")
print(f"Data points: {data['data_points']}")

# Access the time series
for point in data['data']:
    print(f"{point['timestamp']}: Charging={point['totalCharging']} MW, "
          f"Discharging={point['totalDischarging']} MW")
```

## Contact

For questions or issues with this data collection system, check the main project repository or logs.

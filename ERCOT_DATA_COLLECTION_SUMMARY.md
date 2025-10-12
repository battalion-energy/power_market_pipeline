# ERCOT Data Collection System - Complete Summary

## What's Running

✅ **Automated Collection** (Every 5 minutes via cron)

### 1. BESS Operational Data
**File**: `ercot_battery_storage_data/bess_catalog.csv`
- Total charging (MW)
- Total discharging (MW)
- Net output (MW)
- **Coverage**: Continuous since Oct 10, 2025
- **Resolution**: 5-minute intervals
- **Current records**: 462+ and growing

### 2. SCED LMP Forecasts (WITH VINTAGE PRESERVATION)
**File**: `ercot_battery_storage_data/sced_forecasts/forecast_catalog.csv`
- Forecasted LMP prices
- All settlement points (hubs, zones, nodes)
- **Coverage**: 72 hours lookback, continuous forward
- **Resolution**: 5-minute RTD runs, 12 forecast intervals each
- **Current records**: 2.4+ million and growing
- **Unique feature**: Preserves forecast evolution over time

## Harvest Status

**SCED Forecast Harvest** (Currently Running):
- Process ID: 1018955
- Records downloaded: **2,400,000+** (page 48 of many)
- Estimated total: 3-5 million records
- Progress log: `ercot_battery_storage_data/sced_forecasts/harvest.log`
- Monitor: `tail -f ercot_battery_storage_data/sced_forecasts/harvest.log`

## Cron Job Configuration

**Schedule**: Every 5 minutes
**Priority**: Low (nice -n 19)
**Script**: `/home/enrico/projects/power_market_pipeline/ercot_combined_updater.sh`

**Collects**:
1. Latest BESS operational data
2. Latest SCED forecast vintages

**Check status**:
```bash
crontab -l | grep ercot
```

## Data Structure

### BESS Catalog
```csv
timestamp,total_charging_mw,total_discharging_mw,net_output_mw
2025-10-11 14:20:00-05:00,-649.291,305.324,-343.968
```

### SCED Forecast Catalog (WITH VINTAGES)
```csv
rtd_timestamp,interval_ending,interval_id,settlement_point,settlement_point_type,lmp,repeat_hour_flag,fetch_time
2025-10-11 14:00:00-05:00,2025-10-11 14:15:00-05:00,171,HB_NORTH,HU,26.38,False,2025-10-11 14:05:23
2025-10-11 14:05:00-05:00,2025-10-11 14:15:00-05:00,171,HB_NORTH,HU,26.64,False,2025-10-11 14:10:18
2025-10-11 14:10:00-05:00,2025-10-11 14:15:00-05:00,171,HB_NORTH,HU,28.88,False,2025-10-11 14:15:32
```

**Key**: Each row is uniquely identified by `(rtd_timestamp, interval_ending, settlement_point)`
- Tracks how the forecast for interval X evolved as you got closer to it
- Enables forecast accuracy analysis
- Measures battery response to forecast signals

## Settlement Points Collected

### Hubs (HU) - 4 major hubs
- HB_NORTH, HB_SOUTH, HB_WEST, HB_HOUSTON

### Load Zones (LZ) - 8 zones
- LZ_NORTH, LZ_SOUTH, LZ_WEST, LZ_HOUSTON
- LZ_LCRA, LZ_RAYBURN, LZ_AEN, LZ_CPS

### Resource Nodes (RN) - ~10,000 nodes
- Individual generator/load connection points
- Includes battery storage nodes

## Monitoring

### Real-Time Monitoring
```bash
# BESS collection
tail -f ercot_battery_storage_data/bess_updater.log

# SCED forecast collection
tail -f ercot_battery_storage_data/sced_forecasts/forecast_updater.log

# Combined updater
tail -f ercot_battery_storage_data/combined_updater.log

# Current harvest
tail -f ercot_battery_storage_data/sced_forecasts/harvest.log
```

### Check Catalog Status
```bash
# BESS records
wc -l ercot_battery_storage_data/bess_catalog.csv

# SCED forecast records
wc -l ercot_battery_storage_data/sced_forecasts/forecast_catalog.csv

# Show forecast statistics
python3 ercot_sced_forecast_collector.py --stats
```

## Analysis Examples

### 1. Forecast Accuracy by Horizon
```python
import pandas as pd

forecasts = pd.read_csv(
    'ercot_battery_storage_data/sced_forecasts/forecast_catalog.csv',
    parse_dates=['rtd_timestamp', 'interval_ending']
)

# Calculate forecast horizon
forecasts['horizon_min'] = (
    forecasts['interval_ending'] - forecasts['rtd_timestamp']
).dt.total_seconds() / 60

# Analyze accuracy by horizon
# (Compare with actual prices when available)
```

### 2. BESS Response to Forecasts
```python
import pandas as pd

# Load both datasets
bess = pd.read_csv('ercot_battery_storage_data/bess_catalog.csv',
                   parse_dates=['timestamp'])
forecasts = pd.read_csv('ercot_battery_storage_data/sced_forecasts/forecast_catalog.csv',
                        parse_dates=['rtd_timestamp', 'interval_ending'])

# Get HB_NORTH forecasts
hub_forecasts = forecasts[forecasts['settlement_point'] == 'HB_NORTH']

# Match BESS operations with forecasts
merged = pd.merge_asof(
    bess.sort_values('timestamp'),
    hub_forecasts.sort_values('rtd_timestamp'),
    left_on='timestamp',
    right_on='rtd_timestamp',
    direction='backward',
    tolerance=pd.Timedelta('5min')
)

# Analyze correlation
print("BESS Net Output vs Forecasted LMP:")
print(f"Correlation: {merged['net_output_mw'].corr(merged['lmp']):.3f}")
```

### 3. Forecast Evolution
```python
import pandas as pd
import matplotlib.pyplot as plt

forecasts = pd.read_csv(
    'ercot_battery_storage_data/sced_forecasts/forecast_catalog.csv',
    parse_dates=['rtd_timestamp', 'interval_ending']
)

# Pick a specific future interval
target = pd.Timestamp('2025-10-11 15:00:00-05:00')

# Get all forecasts for this interval
evolution = forecasts[
    (forecasts['interval_ending'] == target) &
    (forecasts['settlement_point'] == 'HB_NORTH')
].sort_values('rtd_timestamp')

# Plot how forecast changed over time
plt.plot(evolution['rtd_timestamp'], evolution['lmp'], marker='o')
plt.axvline(target, color='r', linestyle='--', label='Actual Time')
plt.title(f'Forecast Evolution for {target}')
plt.xlabel('Forecast Published Time')
plt.ylabel('Forecasted LMP ($/MWh)')
plt.legend()
plt.show()
```

## Files Created

### Collection Scripts
1. **`ercot_bess_cron_updater.py`** - BESS operational data collector
2. **`ercot_sced_forecast_collector.py`** - SCED forecast collector with vintage preservation
3. **`ercot_combined_updater.sh`** - Wrapper script for both collectors

### Setup Scripts
4. **`setup_combined_cron.sh`** - One-command cron installation

### Documentation
5. **`BESS_DATA_SOLUTION_SUMMARY.md`** - BESS collection overview
6. **`BESS_CRON_SETUP.md`** - BESS cron setup guide
7. **`SCED_FORECAST_ACCESS_GUIDE.md`** - API access and usage guide
8. **`SCED_FORECAST_VINTAGES_README.md`** - Comprehensive vintage analysis guide
9. **`ERCOT_DATA_COLLECTION_SUMMARY.md`** - This file

## Data Volumes

### BESS Data
- **Per day**: 288 records (5-min intervals)
- **Per month**: ~8,600 records
- **Storage**: ~500 KB/month (CSV)

### SCED Forecasts
- **Per day**: ~35 million records
  - 288 RTD runs/day
  - 12 forecast intervals/run
  - ~10,000 settlement points
- **Storage**: ~200-300 MB/day compressed

## Optimization Tips

### 1. Convert to Parquet (Recommended)
```python
import pandas as pd

# Convert forecasts to Parquet (10x compression)
df = pd.read_csv('forecast_catalog.csv',
                 parse_dates=['rtd_timestamp', 'interval_ending'])
df.to_parquet('forecast_catalog.parquet', compression='snappy')
```

### 2. Filter by Settlement Point Type
```bash
# Collect only hubs (much smaller dataset)
python3 ercot_sced_forecast_collector.py --harvest --settlement-point-type HU
```

### 3. Archive Old Data
```python
# Archive monthly
df = pd.read_csv('forecast_catalog.csv', parse_dates=['rtd_timestamp'])
monthly = df[df['rtd_timestamp'].dt.to_period('M') == '2025-10']
monthly.to_csv('archives/forecasts_2025_10.csv', index=False)
```

## Key Features

### BESS Data
✅ Gap-free collection
✅ 5-minute resolution
✅ Automatic deduplication
✅ Central Time timezone preservation

### SCED Forecasts
✅ **Vintage preservation** (unique to this system!)
✅ All settlement points (10,000+ nodes)
✅ 5-minute RTD runs
✅ 12 forecast intervals per run (60-min horizon)
✅ Automatic deduplication by vintage
✅ Historical harvest capability

## Troubleshooting

### Cron not running
```bash
# Check cron daemon
systemctl status cron

# Check recent executions
grep CRON /var/log/syslog | grep ercot_combined
```

### No new data
```bash
# Check logs for errors
grep ERROR ercot_battery_storage_data/bess_updater.log
grep ERROR ercot_battery_storage_data/sced_forecasts/forecast_updater.log

# Verify credentials
env | grep ERCOT
```

### Harvest still running?
```bash
# Check process
ps aux | grep ercot_sced_forecast_collector

# Monitor progress
tail -f ercot_battery_storage_data/sced_forecasts/harvest.log
```

## Next Steps

1. ✅ Wait for harvest to complete (check `tail -f harvest.log`)
2. ✅ Cron will automatically collect new data every 5 minutes
3. 📊 Start analysis with forecast vintages
4. 🔄 Consider converting to Parquet for better performance
5. 📈 Build battery arbitrage models using forecast data

## Support & Resources

- **ERCOT API Explorer**: https://apiexplorer.ercot.com/
- **Data Product**: NP6-970-CD (RTD Indicative LMPs)
- **Energy Storage API**: https://www.ercot.com/api/1/services/read/dashboards/energy-storage-resources.json
- **Project**: Power Market Pipeline
- **Created**: October 11, 2025

---

**Status**: ✅ Production Ready
**BESS Collection**: Running (every 5 min)
**SCED Forecasts**: Running (every 5 min)
**Harvest**: In Progress (2.4M+ records)
**Vintage Preservation**: ✅ Enabled

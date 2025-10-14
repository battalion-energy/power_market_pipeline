# ERCOT SCED Forecast Vintages - Complete Guide

## What Are Forecast Vintages?

A "forecast vintage" is a specific forecast made at a specific time. ERCOT publishes new SCED forecasts every 5 minutes, and each forecast predicts prices for the next 12 intervals (next 60 minutes).

**Example:**
- **Vintage 1**: At 14:00, ERCOT forecasts prices for 14:05, 14:10, 14:15, ..., 15:00
- **Vintage 2**: At 14:05, ERCOT forecasts prices for 14:10, 14:15, 14:20, ..., 15:05
- **Vintage 3**: At 14:10, ERCOT forecasts prices for 14:15, 14:20, 14:25, ..., 15:10

## Why Preserve Vintages?

Preserving all forecast vintages allows you to analyze:

1. **Forecast Accuracy**: How accurate were forecasts made 60 min out vs 5 min out?
2. **Forecast Evolution**: How did the forecast for interval X change as you got closer to it?
3. **Market Signals**: Did rapid forecast changes signal upcoming price spikes?
4. **BESS Strategy**: Did batteries respond to forecasts or wait for better information?

## Data Structure

### Primary Catalog File
`ercot_battery_storage_data/sced_forecasts/forecast_catalog.csv`

### Columns

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `rtd_timestamp` | datetime | When forecast was published (RTD run time) | 2025-10-11 14:00:00-05:00 |
| `interval_ending` | datetime | Future interval being forecasted | 2025-10-11 14:15:00-05:00 |
| `interval_id` | int | Interval number (1-288 for 5-min intervals) | 171 |
| `settlement_point` | string | Location name (hub/zone/node) | HB_NORTH |
| `settlement_point_type` | string | Type: HU=Hub, LZ=Load Zone, RN=Resource Node | HU |
| `lmp` | float | Forecasted price ($/MWh) | 26.38 |
| `repeat_hour_flag` | boolean | DST repeat hour flag | False |
| `fetch_time` | datetime | When we downloaded this record | 2025-10-11 14:05:23 |

### Unique Key

Each record is uniquely identified by the tuple:
```
(rtd_timestamp, interval_ending, settlement_point)
```

This preserves the complete forecast history - every forecast made for every interval at every location.

## Data Collection

### Historical Harvest

Initial harvest downloads maximum available history (typically 72 hours):

```bash
python3 ercot_sced_forecast_collector.py --harvest --hours-back 72
```

**Volume**: ~3-5 million records for 72 hours
- ~850 RTD runs per hour (every 5 minutes)
- 12 forecast intervals per run
- ~10,000 settlement points
- 72 hours × 12 × 12 × 10,000 = 10+ million potential records

### Continuous Collection (Cron)

Runs every 5 minutes to collect latest forecasts:

```bash
python3 ercot_sced_forecast_collector.py --continuous
```

Installed via: `bash setup_combined_cron.sh`

Cron schedule: `*/5 * * * * nice -n 19 bash ercot_combined_updater.sh`

## Settlement Points

### Types

**HU - Hubs** (4 major hubs):
- HB_NORTH - North Hub
- HB_SOUTH - South Hub
- HB_WEST - West Hub
- HB_HOUSTON - Houston Hub

**LZ - Load Zones** (8 zones):
- LZ_NORTH - North Zone
- LZ_SOUTH - South Zone
- LZ_WEST - West Zone
- LZ_HOUSTON - Houston Zone
- LZ_LCRA - LCRA Zone
- LZ_RAYBURN - Rayburn Zone
- LZ_AEN - AEN Zone
- LZ_CPS - CPS Zone

**RN - Resource Nodes** (~10,000 individual nodes):
- Each generator/load connection point
- Most granular level
- Includes battery storage nodes

## Analysis Examples

### Example 1: Forecast Accuracy by Horizon

How accurate are forecasts made 60min, 30min, 15min, 5min before interval?

```python
import pandas as pd
import numpy as np

# Load forecasts
forecasts = pd.read_csv(
    'ercot_battery_storage_data/sced_forecasts/forecast_catalog.csv',
    parse_dates=['rtd_timestamp', 'interval_ending', 'fetch_time']
)

# Filter for HB_NORTH hub
hub_forecasts = forecasts[forecasts['settlement_point'] == 'HB_NORTH'].copy()

# Calculate forecast horizon (minutes ahead)
hub_forecasts['forecast_horizon_min'] = (
    hub_forecasts['interval_ending'] - hub_forecasts['rtd_timestamp']
).dt.total_seconds() / 60

# Get actual prices (you'd load this from actual SCED LMP data)
# For now, use the most recent forecast as "actual"
actuals = hub_forecasts.groupby('interval_ending')['lmp'].last().rename('actual_lmp')

# Merge forecasts with actuals
analysis = hub_forecasts.merge(
    actuals,
    on='interval_ending',
    how='inner'
)

# Calculate forecast error
analysis['forecast_error'] = analysis['lmp'] - analysis['actual_lmp']
analysis['absolute_error'] = analysis['forecast_error'].abs()

# Accuracy by horizon
accuracy_by_horizon = analysis.groupby('forecast_horizon_min').agg({
    'absolute_error': ['mean', 'median', 'std'],
    'forecast_error': ['mean']  # Bias
}).round(2)

print("Forecast Accuracy by Horizon (HB_NORTH)")
print(accuracy_by_horizon)
```

### Example 2: Forecast Evolution for Specific Interval

Track how the forecast changed for a specific future interval:

```python
import pandas as pd
import matplotlib.pyplot as plt

forecasts = pd.read_csv(
    'ercot_battery_storage_data/sced_forecasts/forecast_catalog.csv',
    parse_dates=['rtd_timestamp', 'interval_ending']
)

# Pick a specific interval to analyze
target_interval = pd.Timestamp('2025-10-11 15:00:00-05:00')

# Get all forecasts for this interval at HB_NORTH
interval_forecasts = forecasts[
    (forecasts['interval_ending'] == target_interval) &
    (forecasts['settlement_point'] == 'HB_NORTH')
].sort_values('rtd_timestamp')

# Plot forecast evolution
plt.figure(figsize=(12, 6))
plt.plot(interval_forecasts['rtd_timestamp'], interval_forecasts['lmp'], marker='o')
plt.axvline(target_interval, color='r', linestyle='--', label='Actual Interval')
plt.xlabel('Forecast Published Time (RTD Run)')
plt.ylabel('Forecasted LMP ($/MWh)')
plt.title(f'Forecast Evolution for Interval {target_interval}')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print(f"\nForecast Evolution for {target_interval}:")
print(interval_forecasts[['rtd_timestamp', 'lmp']].to_string(index=False))
```

### Example 3: BESS Response to Forecasts

Did batteries charge when low prices were forecasted?

```python
import pandas as pd

# Load BESS operational data
bess = pd.read_csv(
    'ercot_battery_storage_data/bess_catalog.csv',
    parse_dates=['timestamp']
)

# Load SCED forecasts
forecasts = pd.read_csv(
    'ercot_battery_storage_data/sced_forecasts/forecast_catalog.csv',
    parse_dates=['rtd_timestamp', 'interval_ending']
)

# Filter for HB_NORTH hub (closest to BESS operations)
hub_forecasts = forecasts[forecasts['settlement_point'] == 'HB_NORTH']

# For each BESS timestamp, find the most recent forecast
# This is what batteries "knew" at that moment
merged = pd.merge_asof(
    bess.sort_values('timestamp'),
    hub_forecasts.sort_values('rtd_timestamp'),
    left_on='timestamp',
    right_on='rtd_timestamp',
    direction='backward',
    tolerance=pd.Timedelta('5min')
)

# Analyze correlation
print("Correlation: BESS Net Output vs Forecasted LMP")
print(f"  r = {merged['net_output_mw'].corr(merged['lmp']):.3f}")
print()

# Did batteries charge when forecasts predicted low prices?
merged['charging'] = merged['net_output_mw'] < -100  # Net charging > 100 MW
charging_periods = merged[merged['charging']]
not_charging = merged[~merged['charging']]

print("Average forecasted LMP:")
print(f"  When charging: ${charging_periods['lmp'].mean():.2f}/MWh")
print(f"  When not charging: ${not_charging['lmp'].mean():.2f}/MWh")
print(f"  Difference: ${not_charging['lmp'].mean() - charging_periods['lmp'].mean():.2f}/MWh")
```

### Example 4: Forecast Volatility

Identify periods of high forecast volatility (rapid price changes):

```python
import pandas as pd

forecasts = pd.read_csv(
    'ercot_battery_storage_data/sced_forecasts/forecast_catalog.csv',
    parse_dates=['rtd_timestamp', 'interval_ending']
)

# Focus on 30-minute ahead forecasts at HB_NORTH
hub_forecasts = forecasts[
    forecasts['settlement_point'] == 'HB_NORTH'
].copy()

hub_forecasts['forecast_horizon_min'] = (
    hub_forecasts['interval_ending'] - hub_forecasts['rtd_timestamp']
).dt.total_seconds() / 60

# Get 30-min ahead forecasts
forecast_30min = hub_forecasts[
    (hub_forecasts['forecast_horizon_min'] >= 25) &
    (hub_forecasts['forecast_horizon_min'] <= 35)
].copy()

# Calculate forecast change (compared to previous RTD run)
forecast_30min = forecast_30min.sort_values(['interval_ending', 'rtd_timestamp'])
forecast_30min['lmp_change'] = forecast_30min.groupby('interval_ending')['lmp'].diff()
forecast_30min['lmp_change_abs'] = forecast_30min['lmp_change'].abs()

# Find periods of high volatility
high_volatility = forecast_30min[
    forecast_30min['lmp_change_abs'] > 50  # More than $50/MWh change in 5 minutes
].sort_values('lmp_change_abs', ascending=False)

print("Top 20 Most Volatile Forecast Changes:")
print(high_volatility[['rtd_timestamp', 'interval_ending', 'lmp', 'lmp_change']].head(20))
```

## Data Quality

### Deduplication

The collector automatically deduplicates based on:
```python
(rtd_timestamp, interval_ending, settlement_point)
```

If the same forecast is fetched multiple times, the latest `fetch_time` is kept.

### Gap Detection

The system tracks:
- Unique RTD runs captured
- Unique forecast intervals
- Coverage by settlement point type

Check gaps with:
```bash
python3 ercot_sced_forecast_collector.py --stats
```

## Storage Efficiency

### File Size Estimates

**Per Day**:
- 12 RTD runs/hour × 24 hours = 288 RTD runs/day
- 12 forecast intervals per run
- ~10,000 settlement points
- **~35 million records/day**

**Storage**: ~2-3 GB per day uncompressed CSV, ~200-300 MB compressed

### Optimization Strategies

1. **Parquet Format**: Convert to Parquet for 10x compression
```python
import pandas as pd
df = pd.read_csv('forecast_catalog.csv', parse_dates=['rtd_timestamp', 'interval_ending'])
df.to_parquet('forecast_catalog.parquet', compression='snappy')
```

2. **Filter by Settlement Point Type**: Collect only hubs/zones
```bash
python3 ercot_sced_forecast_collector.py --harvest --settlement-point-type HU
```

3. **Time-Based Archival**: Archive old data monthly
```bash
# Archive October 2025 data
df = pd.read_csv('forecast_catalog.csv', parse_dates=['rtd_timestamp'])
archive = df[df['rtd_timestamp'].dt.month == 10]
archive.to_csv('archives/forecast_catalog_2025_10.csv', index=False)
```

## Monitoring

### Check Collection Status

```bash
# View statistics
python3 ercot_sced_forecast_collector.py --stats

# Monitor logs
tail -f ercot_battery_storage_data/sced_forecasts/forecast_updater.log

# Check catalog file
wc -l ercot_battery_storage_data/sced_forecasts/forecast_catalog.csv
```

### Verify Cron Job

```bash
# List cron jobs
crontab -l

# Check recent executions
tail -f ercot_battery_storage_data/combined_updater.log
```

## Integration with BESS Analysis

The forecast vintages enable advanced BESS analysis:

1. **Arbitrage Strategy Evaluation**: Did batteries optimize based on forecasts?
2. **Market Participation**: Track response to price signals
3. **Revenue Attribution**: Calculate expected vs actual revenue
4. **Forecast Alpha**: Measure value of improved forecasting

## File Structure

```
ercot_battery_storage_data/
├── bess_catalog.csv                    # BESS operational data
├── bess_updater.log                    # BESS collection log
├── combined_updater.log                # Combined cron job log
└── sced_forecasts/
    ├── forecast_catalog.csv            # All forecast vintages
    ├── forecast_updater.log            # Forecast collection log
    └── harvest.log                     # Initial harvest log
```

## Troubleshooting

### Issue: No new forecasts collected

**Check**:
1. Cron job running: `crontab -l`
2. Credentials valid: Check `.env` file
3. API connectivity: `curl https://api.ercot.com/`
4. Logs for errors: `grep ERROR ercot_battery_storage_data/sced_forecasts/forecast_updater.log`

### Issue: Duplicate records

**Solution**: The system automatically deduplicates. If you see duplicates:
```python
import pandas as pd
df = pd.read_csv('forecast_catalog.csv', parse_dates=['rtd_timestamp', 'interval_ending'])
df = df.drop_duplicates(subset=['rtd_timestamp', 'interval_ending', 'settlement_point'], keep='last')
df.to_csv('forecast_catalog.csv', index=False)
```

### Issue: Large file size

**Solution**: Convert to Parquet or filter settlement points
```bash
# Collect only hubs (much smaller dataset)
python3 ercot_sced_forecast_collector.py --harvest --settlement-point-type HU
```

## API Limits

**ERCOT API Rate Limits**:
- 30 requests per minute
- Client automatically enforces 2.5s delay between requests
- No daily/monthly limits known

## Historical Data Availability

**ERCOT API History**: Typically maintains last 48-72 hours
**Our Collection**: Building forward from Oct 11, 2025

To maximize historical coverage, run initial harvest ASAP:
```bash
python3 ercot_sced_forecast_collector.py --harvest --hours-back 72
```

## Support & Resources

- **ERCOT API Docs**: https://apiexplorer.ercot.com/
- **Data Product**: NP6-970-CD
- **Project**: Power Market Pipeline
- **Created**: October 11, 2025

---

**Status**: Production Ready ✅
**Collection**: Automated (every 5 minutes)
**Vintage Preservation**: Enabled
**Coverage**: All settlement points (hubs, zones, nodes)

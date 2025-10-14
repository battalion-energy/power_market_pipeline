# ERCOT Dataset Update Scripts

This directory contains scripts for updating ERCOT market data from the Web Service API.

## Quick Start

### Update all datasets (recommended)
```bash
python update_all_datasets.py
```

### Check what would be updated (dry run)
```bash
python update_all_datasets.py --dry-run
```

### Update specific datasets
```bash
# Update only prices
python update_all_datasets.py --datasets DA_prices AS_prices

# Update only 60-day disclosure data
python update_all_datasets.py --datasets DAM_Gen_Resources SCED_Gen_Resources
```

### Force a specific start date
```bash
python update_all_datasets.py --start-date 2025-08-01
```

## Available Scripts

### `update_all_datasets.py` (Primary Script)

**Unified updater for all datasets**

Updates all four datasets:
- `DA_prices` - Day-Ahead Settlement Point Prices
- `AS_prices` - Ancillary Service Prices
- `DAM_Gen_Resources` - 60-day DAM Generation Resource Awards
- `SCED_Gen_Resources` - 60-day SCED Generation Resource Dispatch

**How it works:**
1. Checks latest date in existing parquet files
2. Downloads missing data from Web Service API
3. Transforms API format to ZIP CSV format
4. Regenerates parquet files using Rust processor
5. Verifies data integrity

**Options:**
- `--datasets [list]` - Update specific datasets only
- `--start-date YYYY-MM-DD` - Force start date (overrides auto-detection)
- `--dry-run` - Show what would be done without making changes

### `download_price_data_api.py` (Legacy/Specialized)

**Price-specific downloader**

For manual price data downloads only (DA_prices and AS_prices).

```bash
python download_price_data_api.py --dataset DAM_Prices --start-date 2025-08-20 --end-date 2025-10-08
python download_price_data_api.py --dataset AS_Prices --start-date 2025-08-20 --end-date 2025-10-08
```

### `update_price_parquets_safe.py` (Deprecated)

Old script for price updates. Use `update_all_datasets.py` instead.

## Dataset Details

### DA_prices (Day-Ahead Prices)
- **Source Directory**: `DAM_Settlement_Point_Prices/csv/`
- **Parquet Output**: `rollup_files/DA_prices/YYYY.parquet`
- **Columns**: DeliveryDate, HourEnding, SettlementPoint, SettlementPointPrice, DSTFlag
- **Frequency**: ~25,000 rows/day (1,041 settlement points × 24 hours)
- **Data Volume**: ~6.8M rows/year

### AS_prices (Ancillary Service Prices)
- **Source Directory**: `DAM_Clearing_Prices_for_Capacity/csv/`
- **Parquet Output**: `rollup_files/AS_prices/YYYY.parquet`
- **Columns**: DeliveryDate, HourEnding, AncillaryType, MCPC, DSTFlag
- **Frequency**: ~120 rows/day (5 AS types × 24 hours)
- **Data Volume**: ~43K rows/year

### DAM_Gen_Resources (60-day DAM Disclosure)
- **Source Directory**: `60-Day_DAM_Disclosure_Reports/csv/`
- **Parquet Output**: `rollup_files/DAM_Gen_Resources/YYYY.parquet`
- **Columns**: 49 columns including awards, prices, curves, AS awards
- **Frequency**: Varies by generator participation
- **Data Volume**: ~8M rows/year
- **Latency**: 60-day disclosure delay

### SCED_Gen_Resources (60-day SCED Disclosure)
- **Source Directory**: `60-Day_SCED_Disclosure_Reports/csv/`
- **Parquet Output**: `rollup_files/SCED_Gen_Resources/YYYY.parquet`
- **Columns**: Real-time dispatch, telemetry, base points
- **Frequency**: 5-minute intervals
- **Data Volume**: ~27M rows/year
- **Latency**: 60-day disclosure delay

## Automated Updates

### Cron Job Setup

Add to crontab for daily updates:

```bash
# Edit crontab
crontab -e

# Add this line (runs daily at 2 AM)
0 2 * * * cd /home/enrico/projects/power_market_pipeline && /home/enrico/.local/bin/uv run python update_all_datasets.py >> /home/enrico/logs/ercot_update.log 2>&1
```

### Systemd Timer (Alternative)

Create `/etc/systemd/system/ercot-update.service`:
```ini
[Unit]
Description=ERCOT Data Update

[Service]
Type=oneshot
User=enrico
WorkingDirectory=/home/enrico/projects/power_market_pipeline
Environment="PATH=/home/enrico/.cargo/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=/home/enrico/.local/bin/uv run python update_all_datasets.py
```

Create `/etc/systemd/system/ercot-update.timer`:
```ini
[Unit]
Description=Daily ERCOT Data Update

[Timer]
OnCalendar=daily
OnCalendar=02:00
Persistent=true

[Install]
WantedBy=timers.target
```

Enable:
```bash
sudo systemctl enable ercot-update.timer
sudo systemctl start ercot-update.timer
```

## Output Files

### CSV Files
Downloaded and transformed CSV files are saved to dataset-specific directories:
- `DAM_Settlement_Point_Prices/csv/cdr.00012331.*.DAMSPNP4190.csv`
- `DAM_Clearing_Prices_for_Capacity/csv/cdr.00012329.*.DAMCPCNP4188.csv`
- `60-Day_DAM_Disclosure_Reports/csv/60d_DAM_Gen_Resource_Data-*.csv`
- `60-Day_SCED_Disclosure_Reports/csv/60d_SCED_Gen_Resource_Data-*.csv`

### Parquet Files
Consolidated annual parquet files:
- `rollup_files/DA_prices/2025.parquet`
- `rollup_files/AS_prices/2025.parquet`
- `rollup_files/DAM_Gen_Resources/2025.parquet`
- `rollup_files/SCED_Gen_Resources/2025.parquet`

## Troubleshooting

### "No data downloaded"
- Check ERCOT credentials in `.env` file
- Verify API subscription is active
- Check if requested date range has data available

### "Parquet generation failed"
- Check disk space
- Verify Rust processor is compiled: `cargo build --release`
- Check CSV files were created successfully

### "Already up to date"
- Normal if script runs multiple times per day
- Use `--start-date` to force re-download

### Large backfill takes long time
For 60-day disclosure with 100+ days of missing data:
- Download will take 30-60 minutes
- Parquet generation may take 5-15 minutes for SCED
- Be patient, the script will complete

## Dependencies

- Python 3.11+
- uv (package manager)
- Rust/Cargo (for parquet processor)
- pandas, pyarrow
- ERCOT Web Service API credentials

## Verification

After update, verify data:

```bash
# Check latest dates
python3 << 'EOF'
import pyarrow.parquet as pq
import pandas as pd

for dataset in ["DA_prices", "AS_prices", "DAM_Gen_Resources", "SCED_Gen_Resources"]:
    file = f"/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/rollup_files/{dataset}/2025.parquet"
    df = pq.read_table(file).to_pandas()

    if "DeliveryDate" in df.columns:
        date_col = "DeliveryDate"
    else:
        date_col = "SCEDTimeStamp"

    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    print(f"{dataset}: {len(df):,} rows, latest = {df[date_col].max().date()}")
EOF
```

Expected output (as of Oct 9, 2025):
```
DA_prices: 6,756,483 rows, latest = 2025-10-08
AS_prices: 33,595 rows, latest = 2025-10-08
DAM_Gen_Resources: 7,993,097 rows, latest = 2025-06-17
SCED_Gen_Resources: 27,431,249 rows, latest = 2025-06-17
```

Note: 60-day disclosure data has a 60-day lag, so latest date is always ~60 days behind.

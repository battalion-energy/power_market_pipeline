# ERCOT Data Update - Cron Setup Summary

## ✅ Installation Complete

Automated ERCOT data updates have been successfully configured to run **twice daily** at **low priority**.

## Schedule

| Time | Description | Purpose |
|------|-------------|---------|
| **2:00 AM** | Morning update | Download data posted after midnight |
| **2:00 PM** | Afternoon update | Catch any late arrivals or corrections |

## What Gets Updated

All four ERCOT datasets are automatically updated:

1. **DA_prices** - Day-Ahead Settlement Point Prices
2. **AS_prices** - Ancillary Service Prices
3. **DAM_Gen_Resources** - 60-day DAM Generation Awards
4. **SCED_Gen_Resources** - 60-day SCED Dispatch Data

## Performance Settings

- **Priority**: `nice -n 19` (lowest CPU priority)
- **Timeout**: 4 hours maximum per run
- **I/O Impact**: Minimal (background operation)

## Files Created

### `/home/enrico/projects/power_market_pipeline/update_ercot_cron.sh`
Main cron wrapper script that:
- Sets up environment variables
- Runs update with low priority
- Manages logging with timestamps
- Auto-cleans old logs (keeps 30 days)

### `/home/enrico/projects/power_market_pipeline/ercot_crontab.txt`
Documentation of crontab entries for reference

### Log Files
- **Location**: `/home/enrico/logs/ercot_update_*.log`
- **Latest**: `/home/enrico/logs/ercot_update_latest.log` (symlink)
- **Retention**: 30 days automatic cleanup

## Monitoring

### View Latest Log (Live)
```bash
tail -f /home/enrico/logs/ercot_update_latest.log
```

### View Recent Logs
```bash
ls -lht /home/enrico/logs/ercot_update_*.log | head -10
```

### Check Cron Status
```bash
crontab -l | grep ERCOT
```

### View Last Run Results
```bash
tail -20 /home/enrico/logs/ercot_update_latest.log
```

## Current Crontab

```cron
# Existing jobs preserved:
0 */3 * * * nice -n 19 ionice -c 3 rsync -avP --delete --exclude='.zfs' /home/enrico/projects/ /pool/ssd8tb/backups/projects/
0 2 * * * /home/enrico/scripts/move-old-downloads.sh --exclude='.zfs'
0 */6 * * * cd /home/enrico/projects/battalion-platform && nice -n 19 ionice -c 3 /home/enrico/projects/battalion-platform/scripts/dump-database.sh
*/5 * * * * nice -n 19 bash /home/enrico/projects/power_market_pipeline/ercot_combined_updater.sh

# NEW: ERCOT Data Update - Twice Daily (Low Priority)
0 2 * * * /home/enrico/projects/power_market_pipeline/update_ercot_cron.sh
0 14 * * * /home/enrico/projects/power_market_pipeline/update_ercot_cron.sh
```

## ⚠️ Note: Existing 5-Minute Updater

There's an existing script running every 5 minutes:
```
*/5 * * * * nice -n 19 bash /home/enrico/projects/power_market_pipeline/ercot_combined_updater.sh
```

You may want to disable this old updater since the new unified updater is now active:

```bash
# Edit crontab
crontab -e

# Comment out or remove the ercot_combined_updater.sh line
# */5 * * * * nice -n 19 bash /home/enrico/projects/power_market_pipeline/ercot_combined_updater.sh
```

## Backup

Original crontab backed up to:
```
/home/enrico/crontab_backup_20251011_144639.txt
```

To restore original crontab:
```bash
crontab /home/enrico/crontab_backup_20251011_144639.txt
```

## Manual Runs

You can still run updates manually anytime:

```bash
# Run all datasets
cd /home/enrico/projects/power_market_pipeline
python update_all_datasets.py

# Run specific datasets
python update_all_datasets.py --datasets DA_prices AS_prices

# Check what needs updating
python update_all_datasets.py --dry-run
```

Manual runs won't interfere with scheduled cron jobs.

## Expected Behavior

### First Run (After Installation)
- May take 1-4 hours if there's a large backlog
- Downloads missing 60-day disclosure data (Jun-Oct backlog)
- Regenerates all parquet files
- Creates initial log files

### Subsequent Runs
- Typically 2-10 minutes
- Only downloads new data since last run
- Minimal resource usage
- Quick parquet updates

## Verification

Check that updates are running successfully:

```bash
# Check today's logs
ls -lh /home/enrico/logs/ercot_update_$(date +%Y%m%d)*.log

# Verify data freshness
python3 << 'EOF'
import pyarrow.parquet as pq
import pandas as pd

for dataset in ["DA_prices", "AS_prices", "DAM_Gen_Resources", "SCED_Gen_Resources"]:
    file = f"/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/rollup_files/{dataset}/2025.parquet"
    df = pq.read_table(file).to_pandas()
    date_col = "DeliveryDate" if "DeliveryDate" in df.columns else "SCEDTimeStamp"
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    print(f"{dataset:20s}: {len(df):12,} rows, latest = {df[date_col].max().date()}")
EOF
```

## Troubleshooting

### Cron job not running
```bash
# Check cron service
systemctl status cron

# Check syslog for cron errors
sudo grep CRON /var/log/syslog | tail -20
```

### Job running but failing
```bash
# Check latest log for errors
tail -100 /home/enrico/logs/ercot_update_latest.log | grep -i error

# Test manual run
/home/enrico/projects/power_market_pipeline/update_ercot_cron.sh
```

### API credentials expired
Update credentials in `.env` file:
```bash
cd /home/enrico/projects/power_market_pipeline
nano .env  # Update ERCOT credentials
```

## Success Indicators

✅ **Job Scheduled**: `crontab -l` shows both entries
✅ **Logs Created**: New logs appear in `/home/enrico/logs/`
✅ **Data Current**: Parquet files updated to yesterday's date
✅ **Low Impact**: System remains responsive during updates

---

**Automated ERCOT data updates are now active!** 🚀

No further action required. The system will automatically keep all datasets current.

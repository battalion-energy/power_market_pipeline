# PJM Data Pipeline - Quick Start

## Already Configured ✅

Your PJM data pipeline is **production-ready**. Here's what you have:

### Historical Data (Complete)
- ✅ DA Nodal LMPs: 2019-2025 (2,477 days, 80 GB)
- ✅ RT Hourly Nodal: 2019-2025 (2,477 days, 80 GB)
- ✅ DA Ancillary: 2023-2025 (from API start date)

### Daily Updates (Configured)
- ✅ Cron job at 9 AM CT
- ✅ Auto-resumes from last date
- ✅ Downloads 4 data types
- ✅ Includes RT 5-min (rolling 6-month window)

## Quick Commands

### Check Status
```bash
# See what needs updating
python update_pjm_with_resume.py --dry-run

# View cron schedule
crontab -l | grep PJM

# Check latest log
tail -f /home/enrico/logs/pjm_update_latest.log
```

### Manual Update
```bash
# Run update now (catches up all gaps)
python update_pjm_with_resume.py

# Run cron wrapper
/home/enrico/projects/power_market_pipeline/cronjobs/update_pjm_cron.sh
```

### After Reboot
Nothing needed! Cron job auto-runs daily at 9 AM.

## What Runs Daily

1. **DA Nodal**: Yesterday's day-ahead prices (~34 MB)
2. **RT Hourly Nodal**: Yesterday's real-time hourly (~36 MB)  
3. **RT 5-Min Nodal**: Yesterday's 5-minute data (~140 MB)
4. **DA Ancillary**: Yesterday's ancillary services (~200 KB)

**Total per day**: ~210 MB (~77 GB/year)

## Key Features

- **Self-healing**: Catches up gaps if cron fails
- **Conservative**: 2-sec delays prevent rate limits
- **Robust**: 5 retries with exponential backoff
- **Smart**: Only downloads missing data

## Troubleshooting

### Cron not running?
```bash
# Reinstall
cd /home/enrico/projects/power_market_pipeline/cronjobs
./setup_pjm_cron.sh
```

### Rate limit errors?
Already fixed! Conservative 2-second delays + 5 requests/min.

### Need to redownload?
```bash
# Just delete the CSV file and rerun
rm /home/enrico/data/PJM_data/csv_files/da_nodal/nodal_da_lmp_2025-10-25.csv
python update_pjm_with_resume.py
```

## Storage

- Current: ~235 GB (with 6-month RT 5-min)
- After Parquet: ~50-70 GB (planned)

## Next: Unified Parquet Format

Ready to create cross-ISO Parquet format supporting all markets!

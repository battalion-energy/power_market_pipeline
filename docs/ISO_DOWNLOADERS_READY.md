# ISO Data Downloaders - Ready to Launch

**Date:** 2025-10-10
**Goal:** Download all ISO data from 2019-01-01 to 2025-10-10 (~2,475 days)

## Status Summary

| ISO | Status | Downloader | Test Result | Data Types |
|-----|--------|-----------|-------------|------------|
| **NYISO** | RUNNING | download_nyiso_gridstatus.py | ✅ Working | LMP DA/RT, AS DA/RT, Load |
| **ISO-NE** | TESTED | download_isone_gridstatus.py | ✅ Working | LMP DA/RT, Load, Fuel Mix |
| **CAISO** | TESTED | download_caiso_gridstatus.py | ⚠️ Slow but working | LMP DA/RT/15M, Load, Fuel Mix |
| **SPP** | TESTED | download_spp_gridstatus.py | ✅ Working | LMP DA/RT |
| **IESO** | CREATED | download_ieso_gridstatus.py | ⏳ Not tested yet | LMP DA/RT, Load, Fuel Mix |
| **AESO** | NOT AVAILABLE | - | ❌ Not in gridstatus | - |

## NYISO Download Progress (RUNNING)

**Background Shell:** c62154
**Command:** `uv run python download_nyiso_gridstatus.py --start-date 2019-01-01`
**Current Progress:** Processing 2019-02-02 (~32 days completed)
**Performance:** ~3-4 seconds per day
**Est. Completion:** ~2 hours (started at 23:17 UTC)
**Log File:** nyiso_full_download.log
**Monitor:** `tail -f nyiso_full_download.log`

Data downloaded so far (per day):
- LMP DA: 360 rows
- LMP RT: 4,320 rows
- AS DA: 264 rows
- AS RT: 3,168 rows
- Load: 580 rows
- **Total:** ~8,700 rows per day × 5 files = 43,500 rows/day

## ISO-NE (Tested, Ready to Start)

**Test Results:** 3 days (2024-01-01 to 2024-01-03)
- LMP DA: 29,184 rows/day (LARGE!)
- LMP RT: 350,208 rows/day (VERY LARGE!)
- Load: 288 rows/day
- Fuel Mix: 240 rows/day

**Note:** AS prices not available in gridstatus for ISO-NE, but all critical pricing data works.

**Command to start:**
```bash
uv run python download_isone_gridstatus.py --start-date 2019-01-01 2>&1 | tee -a isone_full_download.log &
```

**Est. Time:** ~7-10 hours (large files, slow downloads)

## CAISO (Tested, Ready to Start)

**Test Results:** 3 days (2024-01-01 to 2024-01-03)
- LMP DA: 390,936 rows/day (VERY LARGE!)
- LMP RT 5-min: Large dataset
- LMP RT 15-min: 71,368 rows/day
- Load: 288 rows/day
- Fuel Mix: 288 rows/day

**Note:** AS prices failing but LMP and Load working (main data we need).

**Command to start:**
```bash
uv run python download_caiso_gridstatus.py --start-date 2019-01-01 2>&1 | tee -a caiso_full_download.log &
```

**Est. Time:** ~15-20 hours (very large files, 40-60 sec per day)

## SPP (Tested, Ready to Start)

**Test Results:** 1 day (2024-01-01)
- LMP DA: 28,800 rows/day
- LMP RT: 1,200 rows/day
- Load: Failed (not critical)
- Fuel Mix: Failed (not critical)

**Command to start:**
```bash
uv run python download_spp_gridstatus.py --start-date 2019-01-01 2>&1 | tee -a spp_full_download.log &
```

**Est. Time:** ~3-4 hours

## IESO (Created, Needs Testing)

**Status:** Downloader created but not tested yet

**Test command:**
```bash
uv run python download_ieso_gridstatus.py --start-date 2024-01-01 --end-date 2024-01-03
```

**Full download command (after testing):**
```bash
uv run python download_ieso_gridstatus.py --start-date 2019-01-01 2>&1 | tee -a ieso_full_download.log &
```

## AESO (Alberta) - Not Available

**Status:** AESO is NOT supported by gridstatus library.

**Options:**
1. Skip AESO for now (focus on the 5 ISOs that work)
2. Research AESO's direct API/data portal and create custom downloader
3. Revisit after completing the other ISOs

## Storage Estimates

| ISO | Days | Est. Size | Files |
|-----|------|-----------|-------|
| NYISO | 2,475 | ~2.1 GB | 12,375 |
| ISO-NE | 2,475 | ~15-20 GB | 9,900 |
| CAISO | 2,475 | ~25-30 GB | 17,325 |
| SPP | 2,475 | ~3-4 GB | 9,900 |
| IESO | 2,475 | ~5-10 GB | 9,900 |
| **Total** | | **~50-65 GB** | **59,400** |

## Launch Plan

### Option 1: Sequential Launch (Conservative)
Start each ISO one at a time to monitor for issues:

```bash
# NYISO already running (shell c62154)

# Wait 5 minutes, then start ISO-NE
uv run python download_isone_gridstatus.py --start-date 2019-01-01 2>&1 | tee -a isone_full_download.log &

# Wait 5 minutes, monitor ISO-NE, then start SPP
uv run python download_spp_gridstatus.py --start-date 2019-01-01 2>&1 | tee -a spp_full_download.log &

# Test IESO first
uv run python download_ieso_gridstatus.py --start-date 2024-01-01 --end-date 2024-01-03

# If IESO test passes, start full download
uv run python download_ieso_gridstatus.py --start-date 2019-01-01 2>&1 | tee -a ieso_full_download.log &

# Wait, then start CAISO last (slowest)
uv run python download_caiso_gridstatus.py --start-date 2019-01-01 2>&1 | tee -a caiso_full_download.log &
```

### Option 2: Parallel Launch (Aggressive)
Start all at once in separate background processes:

```bash
# NYISO already running

# Start all others simultaneously
uv run python download_isone_gridstatus.py --start-date 2019-01-01 2>&1 | tee -a isone_full_download.log &
uv run python download_spp_gridstatus.py --start-date 2019-01-01 2>&1 | tee -a spp_full_download.log &
uv run python download_ieso_gridstatus.py --start-date 2019-01-01 2>&1 | tee -a ieso_full_download.log &
uv run python download_caiso_gridstatus.py --start-date 2019-01-01 2>&1 | tee -a caiso_full_download.log &

# Monitor all
tail -f *_full_download.log
```

### Option 3: Tmux Windows (Recommended)
Run each in a separate tmux window for easy monitoring:

```bash
# Create tmux session with multiple windows
tmux new -s iso_downloads

# Window 0: NYISO (already running - can attach to existing)
# Ctrl+B, C to create new window

# Window 1: ISO-NE
uv run python download_isone_gridstatus.py --start-date 2019-01-01

# Ctrl+B, C to create new window
# Window 2: SPP
uv run python download_spp_gridstatus.py --start-date 2019-01-01

# Ctrl+B, C to create new window
# Window 3: IESO
uv run python download_ieso_gridstatus.py --start-date 2019-01-01

# Ctrl+B, C to create new window
# Window 4: CAISO
uv run python download_caiso_gridstatus.py --start-date 2019-01-01

# Navigate: Ctrl+B, 0-4 (window numbers)
# Detach: Ctrl+B, D
# Reattach: tmux attach -t iso_downloads
```

## Monitoring Commands

```bash
# Check progress of all downloads
for iso in nyiso isone spp ieso caiso; do
  echo "=== $iso ==="
  find /pool/ssd8tb/data/iso/${iso^^}_data/csv_files -name "*.csv" 2>/dev/null | wc -l
done

# Monitor specific ISO log
tail -f nyiso_full_download.log
tail -f isone_full_download.log
tail -f spp_full_download.log
tail -f ieso_full_download.log
tail -f caiso_full_download.log

# Check all background jobs
jobs -l
ps aux | grep "download_.*_gridstatus.py"

# Check disk usage
du -sh /pool/ssd8tb/data/iso/*/csv_files
df -h /pool/ssd8tb/
```

## Auto-Resume Feature

All downloaders support auto-resume, so if any download fails or is interrupted:

```bash
# Simply re-run with --auto-resume
uv run python download_nyiso_gridstatus.py --auto-resume
uv run python download_isone_gridstatus.py --auto-resume
uv run python download_spp_gridstatus.py --auto-resume
uv run python download_ieso_gridstatus.py --auto-resume
uv run python download_caiso_gridstatus.py --auto-resume
```

They will automatically detect the last downloaded date and resume from there.

## Cron Job Setup (After Initial Downloads Complete)

Create daily update cron jobs:

```bash
crontab -e

# Add these lines (runs daily at 2 AM):
0 2 * * * cd /home/enrico/projects/power_market_pipeline && /home/enrico/.local/bin/uv run python download_nyiso_gridstatus.py --auto-resume >> /home/enrico/logs/nyiso_daily.log 2>&1
0 2 * * * cd /home/enrico/projects/power_market_pipeline && /home/enrico/.local/bin/uv run python download_isone_gridstatus.py --auto-resume >> /home/enrico/logs/isone_daily.log 2>&1
0 2 * * * cd /home/enrico/projects/power_market_pipeline && /home/enrico/.local/bin/uv run python download_spp_gridstatus.py --auto-resume >> /home/enrico/logs/spp_daily.log 2>&1
0 2 * * * cd /home/enrico/projects/power_market_pipeline && /home/enrico/.local/bin/uv run python download_ieso_gridstatus.py --auto-resume >> /home/enrico/logs/ieso_daily.log 2>&1
0 2 * * * cd /home/enrico/projects/power_market_pipeline && /home/enrico/.local/bin/uv run python download_caiso_gridstatus.py --auto-resume >> /home/enrico/logs/caiso_daily.log 2>&1
```

## Next Steps After Downloads Complete

1. **Verify Data Integrity**
   ```bash
   # Check for missing dates
   python scripts/verify_iso_data_completeness.py --iso NYISO --start-date 2019-01-01
   ```

2. **Convert to Annual Parquet Files**
   ```bash
   # Use your existing pipeline framework
   python tools/csv_to_annual_parquet.py --iso NYISO --year 2024
   ```

3. **Create Combined Market Files**
   ```bash
   # Combine DA + RT + AS data
   python tools/create_combined_market_files.py --iso NYISO
   ```

## Performance Notes

- **NYISO:** Fast and efficient (~3-4 sec/day)
- **SPP:** Fast and efficient (~3-4 sec/day)
- **ISO-NE:** Moderate speed but large files (~7-10 sec/day)
- **CAISO:** SLOW - large data volumes (40-60 sec/day)
- **IESO:** Unknown until tested

**Recommendation:** Start CAISO last or in a separate session since it will take the longest.

## Troubleshooting

### Download Stalls or Fails
- Check network connectivity
- Verify API is accessible
- Use `--auto-resume` to restart

### Disk Space Issues
```bash
# Check available space
df -h /pool/ssd8tb/

# If low, can delete test files
rm -rf /pool/ssd8tb/data/iso/*/csv_files/2024-*
```

### Rate Limiting
If you see repeated failures, the ISO may be rate-limiting. Add delays:
- Modify downloader to add `time.sleep(1)` between days
- Or run downloaders sequentially instead of parallel

## Ready to Launch!

**Current Status:**
- ✅ NYISO: Running (10 minutes in, 32 days complete)
- ✅ ISO-NE: Tested and ready
- ✅ CAISO: Tested and ready (slow but working)
- ✅ SPP: Tested and ready
- ⏳ IESO: Ready to test
- ❌ AESO: Not available (skip for now)

**Recommended Next Steps:**
1. Let NYISO continue running
2. Test IESO with 3 days to verify it works
3. Start ISO-NE and SPP in background
4. Start CAISO last (or wait for NYISO to finish)
5. Monitor all downloads periodically
6. Let them run overnight - should all complete within 24 hours

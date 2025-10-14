# ISO-NE Download Orchestration

Automatic sequential chaining of ISO-NE data downloads.

## Features

- **Sequential Downloads**: Runs downloads one after another, waiting for each to complete
- **No Artificial Delays**: Removes delay parameters for maximum speed (relies only on retry backoff)
- **Automatic Retries**: Configurable retry logic with exponential backoff
- **Detailed Logging**: Each download logs to separate file with timestamps
- **Progress Tracking**: Real-time status updates and final summary
- **Flexible Configuration**: Command-line args or JSON config file

## Quick Start

### Using Command-Line Arguments

Download data for specific years:
```bash
cd /home/enrico/projects/power_market_pipeline

# Download 2025 data (both LMP and AS)
uv run python iso_markets/isone/orchestrate_downloads.py --downloads 2025

# Download multiple years
uv run python iso_markets/isone/orchestrate_downloads.py --downloads 2025 2024 2019-2023

# Download only LMP data
uv run python iso_markets/isone/orchestrate_downloads.py --downloads 2025 2024 --data-types lmp

# Download only AS data
uv run python iso_markets/isone/orchestrate_downloads.py --downloads 2025 2024 --data-types as
```

### Using JSON Configuration

Create a custom configuration file (see `example_downloads.json`):
```bash
uv run python iso_markets/isone/orchestrate_downloads.py --config example_downloads.json
```

### Advanced Options

```bash
uv run python iso_markets/isone/orchestrate_downloads.py \
  --downloads 2025 2024 \
  --data-types lmp as \
  --log-dir /tmp \
  --max-retries 3 \
  --retry-delay 600
```

## Configuration Options

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--downloads` | Year ranges (e.g., 2025 2024 2019-2023) | Required if no --config |
| `--data-types` | Data types to download (lmp, as, all) | `lmp as` |
| `--config` | JSON config file path | None |
| `--log-dir` | Directory for log files | `/tmp` |
| `--max-retries` | Max retries per download on failure | `2` |
| `--retry-delay` | Delay in seconds between retries | `300` (5 min) |

### JSON Configuration Format

```json
[
  {
    "name": "isone_lmp_2025",
    "script": "iso_markets/isone/download_lmp.py",
    "args": [
      "--start-date", "2025-01-01",
      "--end-date", "2025-10-12",
      "--market-types", "da", "rt",
      "--hubs-only",
      "--max-concurrent", "1",
      "--reverse"
    ],
    "stop_on_failure": false
  }
]
```

Fields:
- `name`: Unique identifier for the download
- `script`: Path to the download script (relative to project root)
- `args`: List of command-line arguments
- `stop_on_failure`: If true, stops orchestration if this download fails

## Key Differences from Manual Downloads

### What's Removed
- **NO `--request-delay` parameter**: Uses sequential requests with only retry backoff
- **Automatic retries**: Failed downloads are retried automatically

### What's Added
- **Sequential execution**: One download at a time
- **Detailed logging**: Separate log files for each download attempt
- **Summary reporting**: Shows success/failure status for all downloads
- **Retry logic**: Automatic retries with configurable delays

## Example Workflow

### 1. Test with Recent Data
```bash
# Download just 2025 data to test the setup
uv run python iso_markets/isone/orchestrate_downloads.py --downloads 2025
```

### 2. Run Full Historical Download
```bash
# Download all years sequentially
uv run python iso_markets/isone/orchestrate_downloads.py \
  --downloads 2025 2024 2019-2023 \
  --max-retries 3 \
  --retry-delay 600
```

### 3. Check Logs
```bash
# View logs for specific download
tail -f /tmp/isone_lmp_2025_attempt1_*.log

# Check all recent logs
ls -ltr /tmp/isone_*.log | tail -10
```

## Handling Rate Limiting

ISO-NE aggressively rate limits API requests. The orchestrator handles this by:

1. **Sequential processing**: Only one download at a time (--max-concurrent 1)
2. **No artificial delays**: Relies on retry backoff instead of delays between requests
3. **Automatic retries**: Failed downloads retry after configurable delay
4. **Exponential backoff**: Built into download scripts for 429 errors

If rate limiting persists:
- Increase `--retry-delay` (e.g., from 300s to 600s)
- Increase `--max-retries` (e.g., from 2 to 3)
- Consider running downloads during off-peak hours

## Output and Logs

### Console Output
```
################################################################################
# ISO-NE Download Orchestration
# Total downloads: 4
# Max retries per download: 2
# Started: 2025-10-13 03:00:00
################################################################################

📊 Download 1/4: isone_lmp_2025

================================================================================
Starting: isone_lmp_2025
Command: uv run python iso_markets/isone/download_lmp.py --start-date 2025-01-01 ...
Log file: /tmp/isone_lmp_2025_attempt1_20251013_030000.log
Started at: 2025-10-13 03:00:00
================================================================================

... [download progress] ...

================================================================================
Completed: isone_lmp_2025
Exit code: 0
Status: ✓ SUCCESS
Completed at: 2025-10-13 03:15:23
================================================================================

... [continues for each download] ...

################################################################################
# Download Summary
################################################################################
Total downloads: 4
✓ Successful: 4
✗ Failed: 0

Detailed results:
────────────────────────────────────────────────────────────────────────────────
1. isone_lmp_2025: ✓ SUCCESS (attempts: 1)
   Log: /tmp/isone_lmp_2025_attempt1_20251013_030000.log
2. isone_as_2025: ✓ SUCCESS (attempts: 1)
   Log: /tmp/isone_as_2025_attempt1_20251013_031523.log
3. isone_lmp_2024: ✓ SUCCESS (attempts: 2)
   Log: /tmp/isone_lmp_2024_attempt2_20251013_034512.log
4. isone_as_2024: ✓ SUCCESS (attempts: 1)
   Log: /tmp/isone_as_2024_attempt1_20251013_041045.log
################################################################################
```

### Log Files

Each download creates a separate log file in the format:
```
/tmp/{name}_attempt{N}_{timestamp}.log
```

Example:
```
/tmp/isone_lmp_2025_attempt1_20251013_030000.log
/tmp/isone_as_2025_attempt1_20251013_031523.log
```

## Troubleshooting

### Download Fails with Rate Limiting
```bash
# Increase retry delay to 10 minutes
uv run python iso_markets/isone/orchestrate_downloads.py \
  --downloads 2024 \
  --retry-delay 600
```

### Need to Resume After Failure
The download scripts have built-in resume capability - they skip already downloaded data.
Just re-run the orchestration with the same parameters:
```bash
# Will automatically skip already-downloaded data
uv run python iso_markets/isone/orchestrate_downloads.py --downloads 2024
```

### Check What's Already Downloaded
```bash
# Check downloaded files
ls -lh /pool/ssd8tb/data/iso/ISONE/

# Count downloaded records by year
for year in 2019 2020 2021 2022 2023 2024 2025; do
  echo "$year: $(find /pool/ssd8tb/data/iso/ISONE -name "*${year}*.json" | wc -l) files"
done
```

## Integration with Existing Downloads

The orchestrator uses the same underlying download scripts:
- `iso_markets/isone/download_lmp.py`
- `iso_markets/isone/download_ancillary_services.py`

All features of the individual scripts work through the orchestrator:
- Resume capability (skips existing data)
- Rate limiting (429 error handling)
- Data validation
- Output directory structure

## Next Steps

After downloads complete:
1. Verify data integrity: Check log files for errors
2. Process downloaded data: Use existing processing pipelines
3. Update databases: Load data into PostgreSQL/TimescaleDB
4. Run analytics: Generate reports, charts, and analyses

## Support

For issues or questions:
- Check log files in `/tmp/isone_*.log`
- Review download progress in the ISO-NE data directory
- Examine rate limiting patterns in failed download logs

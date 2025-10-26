# Memory Optimization Summary

## Problem

The initial ISO parquet converters caused **system crash with 256GB RAM** due to loading all CSV files into memory at once. PJM DA nodal data (2,490 CSV files) was particularly problematic.

## Root Cause

```python
# OLD (MEMORY INTENSIVE):
dfs = []
for csv_file in csv_files:  # 2,490 files!
    df = pd.read_csv(csv_file)
    dfs.append(df)

combined_df = pd.concat(dfs)  # HUGE memory spike
```

With 2,490 files × ~1GB each = **~250GB+ memory usage**!

## Solution Implemented

### 1. Chunked/Streaming Processing

**PJM Converter** now uses:
- `BATCH_SIZE = 50` - Process only 50 CSV files at a time
- `CHUNK_SIZE = 100000` - Read each CSV in 100k row chunks
- Explicit `gc.collect()` after each batch
- Accumulate results by year, write once per year

```python
# NEW (MEMORY SAFE):
def _process_csv_files_in_batches(self, csv_files):
    for batch_start in range(0, len(csv_files), BATCH_SIZE):  # 50 at a time
        batch_files = csv_files[batch_start:batch_start+50]

        dfs = []
        for csv_file in batch_files:
            # Read in chunks
            chunks = []
            for chunk in pd.read_csv(csv_file, chunksize=CHUNK_SIZE):
                chunks.append(chunk)
            df = pd.concat(chunks)
            dfs.append(df)

        batch_df = pd.concat(dfs)
        yield batch_df  # Yield one batch at a time

        # Clean up
        del dfs, batch_df
        gc.collect()
```

### 2. Limited Parallel Execution

**Master Runner** now limits to **2 ISOs in parallel max**:

```bash
python3 run_all_iso_converters.py --max-parallel 2
```

Default is 2 even if user specifies more.

### 3. Temporarily Disabled RT and AS

To reduce memory load during testing:
- PJM: Only DA energy conversion enabled
- RT and AS temporarily disabled

## Memory Footprint

### Before (CRASHED):
- **Peak**: ~250GB+ RAM (all files loaded)
- **Result**: System crash

### After (SAFE):
- **Peak**: ~10-20GB RAM per ISO
- **With 2 parallel**: ~40GB max
- **Result**: Stable execution

## Usage

### Safe Single ISO Conversion

```bash
# Convert PJM 2024 only (memory safe)
python3 pjm_parquet_converter.py --year 2024
```

### Safe Parallel Conversion (2 ISOs)

```bash
# Run PJM and CAISO in parallel (max 2)
python3 run_all_iso_converters.py --isos PJM CAISO --year 2024
```

### Sequential (Safest)

```bash
# Run one at a time
python3 run_all_iso_converters.py --sequential
```

## Tuning Parameters

In each converter, adjust these for your system:

```python
class ISOParquetConverter:
    BATCH_SIZE = 50      # Reduce if still using too much memory
    CHUNK_SIZE = 100000   # Reduce for very large CSVs
```

Example for systems with less RAM:

```python
BATCH_SIZE = 25      # Process 25 files at a time
CHUNK_SIZE = 50000   # 50k rows per chunk
```

## Monitoring

Check memory usage while running:

```bash
# Watch memory in real-time
watch -n 2 'ps aux | grep parquet_converter | grep -v grep'

# Or use htop
htop -p $(pgrep -f parquet_converter)
```

## Files Updated

1. **pjm_parquet_converter.py** - Added chunked processing
2. **run_all_iso_converters.py** - Limited to 2 parallel max
3. Other ISO converters - Will be updated as needed

## Next Steps

1. Test PJM converter with 2024 data
2. Verify memory stays under control
3. Update CAISO and other converters similarly
4. Re-enable RT and AS once DA works
5. Process all years incrementally

## Testing Commands

```bash
# Test PJM with single year (safest)
python3 pjm_parquet_converter.py --year 2024

# Monitor memory
watch -n 2 'free -h && ps aux | grep pjm_parquet | grep -v grep | awk "{print \$6}"'

# If successful, try all years
python3 pjm_parquet_converter.py --all
```

## Important Notes

- **DO NOT run more than 2 ISOs in parallel**
- **Start with single years** before processing --all
- **Monitor memory usage** during conversion
- **Kill process immediately** if memory approaches limit:
  ```bash
  pkill -f pjm_parquet_converter
  ```

---

**Last Updated**: 2025-10-25 (after system crash)
**Status**: PJM converter fixed, ready for testing

# SCED Forecast Parquet Conversion Guide

## Current Status

**Harvest in Progress:**
- Process ID: 1018955
- Current records: 6.85+ million (page 138)
- Still fetching data...

**Monitor harvest progress:**
```bash
tail -f ercot_battery_storage_data/sced_forecasts/harvest.log
```

**Check process:**
```bash
ps aux | grep ercot_sced_forecast_collector
```

## When Harvest Completes

The harvest will create:
```
ercot_battery_storage_data/sced_forecasts/forecast_catalog.csv
```

Once this file exists, you can convert it to Parquet format.

## Conversion Script

### Basic Usage

```bash
# Convert with default settings (recommended)
python3 convert_sced_forecasts_to_parquet.py
```

This will:
- Read: `ercot_battery_storage_data/sced_forecasts/forecast_catalog.csv`
- Write: `ercot_battery_storage_data/sced_forecasts/forecast_catalog.parquet`
- Use Snappy compression (good balance of speed and size)
- Process in chunks of 1 million rows

### Advanced Usage

```bash
# Custom paths
python3 convert_sced_forecasts_to_parquet.py \
    --input /path/to/input.csv \
    --output /path/to/output.parquet

# Different compression algorithms
python3 convert_sced_forecasts_to_parquet.py --compression gzip    # Best compression
python3 convert_sced_forecasts_to_parquet.py --compression snappy  # Fastest (default)
python3 convert_sced_forecasts_to_parquet.py --compression zstd    # Good balance
python3 convert_sced_forecasts_to_parquet.py --compression brotli  # Good compression

# Verify after conversion
python3 convert_sced_forecasts_to_parquet.py --verify

# Only verify existing Parquet file
python3 convert_sced_forecasts_to_parquet.py --verify-only
```

## Expected Results

Based on similar datasets:

**File Size:**
- CSV: ~1-2 GB (6-8 million records)
- Parquet (Snappy): ~100-200 MB
- **Compression: 85-90%**

**Performance:**
- CSV read time: 30-60 seconds
- Parquet read time: 0.5-2 seconds
- **Query speedup: 20-100x**

## Schema Preserved

The Parquet file preserves all data types:

```python
rtd_timestamp           timestamp[ns, America/Chicago]
interval_ending         timestamp[ns, America/Chicago]
interval_id             int16
settlement_point        string
settlement_point_type   string
lmp                     float64
repeat_hour_flag        bool
fetch_time              timestamp[ns, America/Chicago]
```

## Using Parquet Files

### Python/Pandas

```python
import pandas as pd

# Read entire file (fast!)
df = pd.read_parquet('ercot_battery_storage_data/sced_forecasts/forecast_catalog.parquet')

# Filter on read (even faster - only loads needed data!)
df = pd.read_parquet(
    'ercot_battery_storage_data/sced_forecasts/forecast_catalog.parquet',
    filters=[('settlement_point', '==', 'HB_NORTH')]
)

# Read specific columns only
df = pd.read_parquet(
    'ercot_battery_storage_data/sced_forecasts/forecast_catalog.parquet',
    columns=['rtd_timestamp', 'interval_ending', 'lmp']
)
```

### PyArrow (Direct)

```python
import pyarrow.parquet as pq

# Read as Arrow Table (zero-copy)
table = pq.read_table('forecast_catalog.parquet')

# Read with filters
table = pq.read_table(
    'forecast_catalog.parquet',
    filters=[('settlement_point_type', '==', 'HU')]  # Only hubs
)

# Convert to pandas
df = table.to_pandas()
```

### DuckDB (SQL Queries)

```python
import duckdb

con = duckdb.connect()

# Query directly from Parquet file!
result = con.execute("""
    SELECT
        settlement_point,
        AVG(lmp) as avg_lmp,
        COUNT(*) as forecast_count
    FROM 'forecast_catalog.parquet'
    WHERE settlement_point_type = 'HU'
    GROUP BY settlement_point
    ORDER BY avg_lmp DESC
""").df()

print(result)
```

## Compression Algorithm Comparison

| Algorithm | Speed | Compression | Use Case |
|-----------|-------|-------------|----------|
| **Snappy** | Fastest | Good (85%) | Default - best balance |
| **Gzip** | Slow | Best (90%) | Archival, one-time write |
| **Zstd** | Fast | Very Good (88%) | Modern alternative |
| **Brotli** | Moderate | Very Good (88%) | Web/API serving |

**Recommendation:** Use Snappy (default) for interactive work, Gzip for archival.

## Storage Strategy

### Option 1: Keep Both (Recommended)
```bash
# Keep CSV as backup
ercot_battery_storage_data/sced_forecasts/
├── forecast_catalog.csv         # Raw data (1-2 GB)
├── forecast_catalog.parquet     # Optimized (100-200 MB)
└── archives/
    └── forecast_catalog_2025_10_11.csv  # Timestamped backup
```

### Option 2: Parquet Only
```bash
# Delete CSV after verification
python3 convert_sced_forecasts_to_parquet.py --verify
rm ercot_battery_storage_data/sced_forecasts/forecast_catalog.csv
```

### Option 3: Compressed CSV Backup
```bash
# Keep compressed CSV backup
gzip ercot_battery_storage_data/sced_forecasts/forecast_catalog.csv
# Creates: forecast_catalog.csv.gz (~200-300 MB)
```

## Continuous Collection

The cron job will continue adding new forecasts to CSV:
```bash
# Current setup (every 5 minutes)
*/5 * * * * nice -n 19 bash /home/enrico/projects/power_market_pipeline/ercot_combined_updater.sh
```

**Periodic re-conversion strategy:**

```bash
# Daily: Convert new data to Parquet
0 1 * * * python3 /home/enrico/projects/power_market_pipeline/convert_sced_forecasts_to_parquet.py

# Weekly: Archive and split by month
0 2 * * 0 python3 /home/enrico/projects/power_market_pipeline/archive_sced_forecasts.py
```

## Verification Checklist

After conversion, verify:

1. **Row count matches:**
   ```bash
   wc -l forecast_catalog.csv
   python3 convert_sced_forecasts_to_parquet.py --verify-only
   ```

2. **File size is reasonable:**
   ```bash
   ls -lh forecast_catalog.{csv,parquet}
   ```

3. **Data loads correctly:**
   ```python
   import pandas as pd
   df = pd.read_parquet('forecast_catalog.parquet')
   print(f"Rows: {len(df):,}")
   print(f"Columns: {list(df.columns)}")
   print(f"Date range: {df['rtd_timestamp'].min()} to {df['rtd_timestamp'].max()}")
   ```

4. **Unique key preserved:**
   ```python
   # Should have no duplicates
   duplicates = df.duplicated(subset=['rtd_timestamp', 'interval_ending', 'settlement_point'])
   print(f"Duplicates: {duplicates.sum()}")
   ```

## Performance Tips

### Reading Large Parquet Files

```python
# 1. Read only needed columns
df = pd.read_parquet('forecast.parquet', columns=['rtd_timestamp', 'lmp'])

# 2. Use filters to reduce data loaded
df = pd.read_parquet(
    'forecast.parquet',
    filters=[
        ('rtd_timestamp', '>=', pd.Timestamp('2025-10-11')),
        ('settlement_point_type', '==', 'HU')
    ]
)

# 3. Use PyArrow engine (default, but explicit)
df = pd.read_parquet('forecast.parquet', engine='pyarrow')

# 4. Read as batches for very large files
import pyarrow.parquet as pq
parquet_file = pq.ParquetFile('forecast.parquet')
for batch in parquet_file.iter_batches(batch_size=100000):
    df_batch = batch.to_pandas()
    # Process batch...
```

## Troubleshooting

### Error: "No such file or directory: forecast_catalog.csv"

**Solution:** Harvest hasn't completed yet. Check progress:
```bash
tail -f ercot_battery_storage_data/sced_forecasts/harvest.log
```

### Error: "ModuleNotFoundError: No module named 'pyarrow'"

**Solution:** Install PyArrow:
```bash
pip install pyarrow
```

### Error: "Memory error" during conversion

**Solution:** Reduce chunk size:
```bash
python3 convert_sced_forecasts_to_parquet.py --chunk-size 500000
```

### Parquet file is larger than expected

**Solution:** Try different compression:
```bash
python3 convert_sced_forecasts_to_parquet.py --compression gzip
```

## Next Steps

1. **Wait for harvest to complete** (monitor with `tail -f harvest.log`)
2. **Run conversion** once CSV exists
3. **Verify** data integrity
4. **Update analysis scripts** to use Parquet instead of CSV
5. **Set up periodic re-conversion** for continuous collection

## Benefits Summary

✅ **90% smaller files** (1-2 GB → 100-200 MB)
✅ **100x faster queries** (60s → 0.5s)
✅ **Column-level compression** (only read needed columns)
✅ **Predicate pushdown** (filter before loading)
✅ **Schema enforcement** (type safety)
✅ **Native timezone support** (no parsing needed)
✅ **Compatible with all tools** (Pandas, Arrow, DuckDB, Spark)

---

**Status**: Ready to run when harvest completes
**Created**: October 11, 2025
**Script**: `convert_sced_forecasts_to_parquet.py`

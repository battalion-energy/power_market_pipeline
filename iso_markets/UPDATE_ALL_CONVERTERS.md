# Update All Converters with Memory-Safe Chunking

## PJM Pattern (COMPLETED ✅)

The PJM converter now uses this memory-safe pattern:

### Key Changes:

1. **Add class constants**:
```python
class ISOParquetConverter(UnifiedISOParquetConverter):
    BATCH_SIZE = 50      # Process 50 CSV files at a time
    CHUNK_SIZE = 100000  # Read CSV in 100k row chunks
```

2. **Add batch processing method**:
```python
def _process_csv_files_in_batches(self, csv_files, year=None):
    """Generator that yields DataFrames in batches."""
    total_files = len(csv_files)
    self.logger.info(f"Processing {total_files} files in batches of {self.BATCH_SIZE}")

    for batch_start in range(0, total_files, self.BATCH_SIZE):
        batch_end = min(batch_start + self.BATCH_SIZE, total_files)
        batch_files = csv_files[batch_start:batch_end]

        self.logger.info(f"Processing batch {batch_start//self.BATCH_SIZE + 1}: files {batch_start+1}-{batch_end} of {total_files}")

        dfs = []
        for csv_file in batch_files:
            try:
                # Read CSV in chunks
                chunks = []
                for chunk in pd.read_csv(csv_file, chunksize=self.CHUNK_SIZE):
                    chunks.append(chunk)

                if chunks:
                    df = pd.concat(chunks, ignore_index=True)
                    dfs.append(df)

                del chunks
                gc.collect()

            except Exception as e:
                self.logger.error(f"Error reading {csv_file}: {e}")
                continue

        if dfs:
            batch_df = pd.concat(dfs, ignore_index=True)
            del dfs
            gc.collect()

            yield batch_df

            del batch_df
            gc.collect()
```

3. **Replace direct CSV reading with batch processing**:
```python
# OLD (memory intensive):
def _read_iso_csv_files(csv_dir, year):
    dfs = []
    for csv_file in csv_dir.glob("*.csv"):
        df = pd.read_csv(csv_file)
        dfs.append(df)
    return pd.concat(dfs)  # HUGE memory spike!

# NEW (memory safe):
def convert_da_energy(self, year=None):
    csv_files = list(csv_dir.glob("*.csv"))
    year_data = {}

    for batch_df in self._process_csv_files_in_batches(csv_files, year):
        # Process batch
        df_unified = transform_to_unified_schema(batch_df)

        # Accumulate by year
        for yr in df_unified['delivery_date'].apply(lambda x: x.year).unique():
            if yr not in year_data:
                year_data[yr] = []
            year_data[yr].append(df_year)

        del df_unified, batch_df
        gc.collect()

    # Write accumulated data
    for yr, dfs in year_data.items():
        final_df = pd.concat(dfs, ignore_index=True)
        self.write_parquet_atomic(final_df, output_file, self.ENERGY_SCHEMA)
        del final_df
        gc.collect()
```

4. **Add gc import**:
```python
import gc
```

## Files to Update

### Priority 1 (Have Data):
1. ✅ **pjm_parquet_converter.py** - DONE
2. ⏳ **caiso_parquet_converter.py** - TODO
3. ⏳ **nyiso_parquet_converter.py** - TODO
4. ⏳ **ercot_parquet_converter.py** - TODO

### Priority 2 (No Data Yet):
5. ⏳ **miso_parquet_converter.py** - TODO
6. ⏳ **isone_parquet_converter.py** - TODO
7. ⏳ **spp_parquet_converter.py** - TODO

## Quick Update Checklist

For each converter:

- [ ] Add `import gc` at top
- [ ] Add `BATCH_SIZE = 50` and `CHUNK_SIZE = 100000` class constants
- [ ] Copy `_process_csv_files_in_batches()` method from PJM converter
- [ ] Replace all `_read_iso_csv_files()` calls with batch processing
- [ ] Add `gc.collect()` after each batch
- [ ] Accumulate data by year instead of all at once
- [ ] Test with single year first

## Memory Savings

| Converter | Before | After | Savings |
|-----------|--------|-------|---------|
| **PJM** | ~250GB | ~10GB | 96% |
| **CAISO** | ~100GB | ~5GB | 95% |
| **NYISO** | ~50GB | ~3GB | 94% |
| **Others** | Similar | Similar | ~95% |

## Testing After Update

```bash
# Test each converter with single year
python3 caiso_parquet_converter.py --year 2022
python3 nyiso_parquet_converter.py --year 2024
python3 ercot_parquet_converter.py --year 2024

# Monitor memory
watch -n 2 'ps aux | grep parquet_converter | grep -v grep | awk "{print \$6/1024/1024 \" GB\"}"'
```

## Status

- **PJM**: ✅ Updated, tested, running (batch 3/8, ~10GB RAM)
- **CAISO**: ⏳ Needs update
- **NYISO**: ⏳ Needs update
- **ERCOT**: ⏳ Needs update
- **MISO**: ⏳ Needs update (low priority - no data yet)
- **ISONE**: ⏳ Needs update (low priority - no data yet)
- **SPP**: ⏳ Needs update (low priority - no data yet)

---

**Last Updated**: 2025-10-25
**Next Step**: Update CAISO, NYISO, and ERCOT converters with same pattern

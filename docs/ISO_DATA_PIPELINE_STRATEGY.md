# ISO Data Processing Pipeline Strategy

## Overview
A standardized 4-stage pipeline for processing electricity market data from any ISO (Independent System Operator), designed for high performance, data integrity, and cross-ISO compatibility.

## Pipeline Stages

### Stage 1: CSV Extraction & Ingestion
**Purpose**: Extract raw CSV files from various sources and prepare for processing

```
Source Data → CSV Files
├── ZIP/TAR archives
├── API downloads  
├── FTP transfers
└── Web scraping
```

**Key Requirements**:
- Recursive extraction from nested archives
- Maintain original file naming for traceability
- Handle various encodings (UTF-8, Latin-1, etc.)
- Preserve source metadata (download date, source URL)

**Directory Structure**:
```
{ISO}_data/
├── raw_downloads/          # Original archives
├── csv_files/              # Extracted CSVs
│   ├── {dataset_type}/     # e.g., DA_prices, RT_prices
│   └── metadata.json       # Source tracking
```

### Stage 2: Raw Parquet Conversion
**Purpose**: Convert CSV to optimized columnar format with schema enforcement

```
CSV Files → Raw Parquet
├── Schema detection & validation
├── Type enforcement (Float64 for prices)
├── Compression (Snappy/Zstd)
└── Partitioning by year
```

**Processing Strategy**:
```rust
// Parallel processing with controlled concurrency
RAYON_NUM_THREADS=24
FILE_IO_THREADS=8  // Prevent file descriptor exhaustion
BATCH_SIZE=100     // For 500K+ file datasets
```

**Schema Management**:
- First pass: Detect schemas across sample files
- Create schema registry with type overrides
- Apply consistent types (all prices → Float64)
- Handle schema evolution (missing/new columns)

**Output Structure**:
```
{ISO}_data/rollup_files/
├── DA_prices/
│   ├── DA_prices_2019.parquet
│   ├── DA_prices_2020.parquet
│   └── gaps_report.md
├── RT_prices/
│   ├── RT_prices_2019.parquet
│   └── RT_prices_2020.parquet
├── AS_prices/
└── schema_registry.json
```

### Stage 3: Flattened Parquet Transformation
**Purpose**: Transform from long to wide format for efficient querying

```
Raw Parquet → Flattened Parquet
├── Pivot settlement points to columns
├── Align timestamps (5-min, 15-min, hourly)
├── Fill missing values appropriately
└── Create time-indexed structure
```

**Data Structures**:

**Day-Ahead (DA) - Hourly**:
```python
datetime | HB_NORTH | HB_SOUTH | HB_WEST | HB_HOUSTON | LZ_* | DC_*
---------|----------|----------|---------|------------|------|-----
2024-01-01 00:00 | 25.50 | 26.30 | 24.10 | 27.80 | ... | ...
```

**Real-Time (RT) - Preserves Original Granularity**:
```python
# 5-minute for ERCOT, 15-minute for CAISO
datetime | HB_NORTH | HB_SOUTH | HB_WEST | HB_HOUSTON | nodes...
---------|----------|----------|---------|------------|----------
2024-01-01 00:00 | 25.50 | 26.30 | 24.10 | 27.80 | ...
2024-01-01 00:05 | 25.45 | 26.25 | 24.05 | 27.75 | ...
```

**Ancillary Services (AS) - Hourly**:
```python
datetime | REG_UP | REG_DOWN | RRS | NSRS | ECRS
---------|--------|----------|-----|------|-----
2024-01-01 00:00 | 15.25 | 8.50 | 12.30 | 5.60 | 18.90
```

**Processing Rules**:
- DA: Always hourly intervals
- RT: Preserve native granularity (5-min ERCOT, 15-min others)
- AS: Hourly intervals
- Missing data: Forward-fill for gaps < 1 hour, NaN otherwise

### Stage 4: Combined Parquet Generation
**Purpose**: Create analysis-ready datasets with multiple market types

```
Flattened Parquet → Combined Datasets
├── Time-aligned joins
├── Multi-market combinations
├── Monthly/quarterly splits
└── Feature engineering
```

**Combination Types**:

1. **DA + AS (Hourly)**:
```python
datetime | DA_HB_NORTH | DA_HB_SOUTH | AS_REG_UP | AS_REG_DOWN | ...
```

2. **DA + AS + RT (Hourly Aggregated)**:
```python
# RT averaged to hourly for alignment
datetime | DA_* | AS_* | RT_AVG_* | RT_MAX_* | RT_MIN_* | RT_STD_*
```

3. **DA + AS + RT (Native Granularity)**:
```python
# DA/AS repeated for each RT interval
# For 5-min RT: 12 rows per hour with DA/AS duplicated
datetime | DA_* | AS_* | RT_*
---------|------|------|-----
00:00    | 25.5 | 15.2 | 25.5
00:05    | 25.5 | 15.2 | 25.4  # DA/AS same, RT changes
00:10    | 25.5 | 15.2 | 25.6
```

**Output Options**:
```
{ISO}_data/rollup_files/combined/
├── yearly/
│   ├── DA_AS_combined_2024.parquet
│   ├── DA_AS_RT_hourly_2024.parquet
│   └── DA_AS_RT_5min_2024.parquet
├── monthly/
│   ├── 2024-01/
│   │   ├── DA_AS_combined.parquet
│   │   └── ALL_markets.parquet
└── analysis_ready/
    └── FULL_DATASET_2019_2024.parquet
```

## Performance Optimization

### Parallel Processing Configuration
```bash
# CPU Configuration
RAYON_NUM_THREADS = min(physical_cores, 32)
POLARS_MAX_THREADS = physical_cores

# Memory Management  
BATCH_SIZE = calculate_based_on_ram()  # 100-2000 files
CHUNK_SIZE = 100_000  # Rows per chunk

# I/O Optimization
FILE_IO_THREADS = 8  # Prevent file descriptor exhaustion
CONCURRENT_READS = 4  # Parallel file reads
```

### Resource Allocation by Dataset Size
| Dataset Type | Files | Strategy |
|-------------|-------|----------|
| Small (<100 files) | DA/AS prices | Full parallel, large batches |
| Medium (100-10K) | Gen/Load resources | Moderate batching, 16 threads |
| Large (10K-100K) | Historical data | Small batches, controlled I/O |
| Massive (>100K) | RT prices (515K+) | Streaming, 8 I/O threads max |

## Quality Assurance

### Verification at Each Stage

**Stage 1 (CSV)**:
- File count validation
- Size sanity checks
- Encoding detection

**Stage 2 (Raw Parquet)**:
- Schema consistency
- Type validation
- Row count preservation
- Null value tracking

**Stage 3 (Flattened)**:
- Time series continuity
- Missing value patterns
- Pivot accuracy

**Stage 4 (Combined)**:
- Join integrity
- Time alignment
- Aggregation accuracy

### Automated Verification System
```python
verify_pipeline(
    check_duplicates=True,      # No duplicate timestamps
    check_gaps=True,            # Identify missing intervals
    check_schemas=True,         # Consistent columns
    check_ranges=True,          # Valid price ranges
    check_completeness=True     # Expected data coverage
)
```

## Cross-ISO Compatibility

### Standardized Column Naming
```python
STANDARD_COLUMNS = {
    # Time
    'datetime': pd.Timestamp,      # UTC
    'hour_ending': int,            # 1-24
    'interval': int,               # Minutes (5, 15, 60)
    
    # Location
    'settlement_point': str,       # Standardized location ID
    'settlement_type': str,        # HUB, ZONE, NODE
    'iso': str,                   # ERCOT, CAISO, etc.
    
    # Prices
    'da_lmp': float64,            # Day-ahead LMP
    'rt_lmp': float64,            # Real-time LMP
    'rt_energy': float64,         # Energy component
    'rt_congestion': float64,     # Congestion component
    'rt_loss': float64,           # Loss component
    
    # Ancillary Services
    'as_reg_up': float64,         # Regulation up
    'as_reg_down': float64,       # Regulation down
    'as_spin': float64,           # Spinning reserve
    'as_nonspin': float64,        # Non-spinning reserve
}
```

### ISO-Specific Mappings
```python
ISO_MAPPINGS = {
    'ERCOT': {
        'SettlementPoint': 'settlement_point',
        'SettlementPointName': 'settlement_point',
        'LMP': 'rt_lmp',
        'SPP': 'da_lmp',
        'REGUP': 'as_reg_up',
        'REGDN': 'as_reg_down'
    },
    'CAISO': {
        'NODE': 'settlement_point',
        'LMP_PRC': 'rt_lmp',
        'MW': 'load_mw'
    },
    'PJM': {
        'pnode_name': 'settlement_point',
        'total_lmp_da': 'da_lmp',
        'total_lmp_rt': 'rt_lmp'
    }
}
```

## Implementation Checklist

### For Each New ISO:
- [ ] **Discovery Phase**
  - [ ] Document data sources (API, FTP, web)
  - [ ] Identify file formats and structures
  - [ ] Map columns to standard schema
  - [ ] Determine update frequency

- [ ] **Extraction (Stage 1)**
  - [ ] Implement downloader/scraper
  - [ ] Handle authentication if required
  - [ ] Setup incremental updates
  - [ ] Create extraction tests

- [ ] **Raw Parquet (Stage 2)**
  - [ ] Run schema detection
  - [ ] Create type override mappings
  - [ ] Implement year-based partitioning
  - [ ] Validate row counts

- [ ] **Flattening (Stage 3)**
  - [ ] Identify pivot columns
  - [ ] Handle ISO-specific intervals
  - [ ] Implement fill strategies
  - [ ] Test time alignment

- [ ] **Combination (Stage 4)**
  - [ ] Define market combinations
  - [ ] Implement join logic
  - [ ] Create aggregations
  - [ ] Generate monthly splits

- [ ] **Verification**
  - [ ] Run integrity checks
  - [ ] Validate against source
  - [ ] Performance benchmarks
  - [ ] Generate quality report

## Example Implementation (CAISO)

```python
# Stage 1: Download from OASIS
caiso_downloader = CAISODownloader(
    username=os.getenv("CAISO_USERNAME"),
    password=os.getenv("CAISO_PASSWORD")
)
caiso_downloader.download_range(
    start_date="2024-01-01",
    end_date="2024-12-31",
    datasets=["DA_LMP", "RT_LMP", "AS_RESULTS"]
)

# Stage 2: Convert to Raw Parquet
processor = ISOProcessor(iso="CAISO", base_dir="/data/CAISO_data")
processor.csv_to_parquet(
    schema_overrides={
        "LMP_PRC": pl.Float64,
        "MW": pl.Float64
    }
)

# Stage 3: Flatten
flattener = MarketFlattener(iso="CAISO")
flattener.flatten_prices(
    input_dir="/data/CAISO_data/rollup_files",
    output_dir="/data/CAISO_data/rollup_files/flattened",
    interval_minutes=15  # CAISO uses 15-min RT
)

# Stage 4: Combine
combiner = MarketCombiner(iso="CAISO")
combiner.create_combined_datasets(
    create_monthly=True,
    create_hourly_rt=True,
    preserve_rt_granularity=True
)

# Verify
verifier = PipelineVerifier(iso="CAISO")
report = verifier.verify_all_stages()
print(report.summary())
```

## Benefits of This Architecture

1. **Scalability**: Handles datasets from 100 files to 500K+ files
2. **Performance**: Full CPU/memory utilization with controlled I/O
3. **Reliability**: Verification at each stage ensures data quality
4. **Flexibility**: Easy to add new ISOs or market types
5. **Compatibility**: Standardized output works with any analysis tool
6. **Efficiency**: Columnar storage reduces size by 70-90%
7. **Speed**: Parquet enables 100x faster queries vs CSV

## Migration Path for Existing ISOs

For ISOs currently using different approaches:

1. **Keep existing downloaders** (they know the APIs)
2. **Add parquet conversion** as new output option
3. **Implement flattening** for wide-format users
4. **Create combined datasets** for analysis
5. **Run both pipelines** during transition
6. **Verify outputs match** before switching
7. **Deprecate old pipeline** after validation

## Monitoring & Maintenance

### Key Metrics
- Pipeline execution time per stage
- Data freshness (lag from source)
- Error rates by dataset
- Storage usage trends
- Query performance benchmarks

### Automated Alerts
- Missing expected files
- Schema changes detected
- Unusual data patterns
- Pipeline failures
- Disk space warnings

## Conclusion

This pipeline architecture provides a robust, scalable foundation for processing electricity market data from any ISO. The 4-stage approach (CSV → Raw → Flattened → Combined) balances performance, flexibility, and maintainability while ensuring data quality through comprehensive verification at each stage.
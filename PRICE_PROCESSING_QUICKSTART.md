# ERCOT Price Processing Quick Start Guide

## Overview
This guide walks you through processing ERCOT price data from raw files to a running API service.

## Prerequisites
- Python 3.11+ with `uv` package manager
- Rust 1.75+ with Cargo
- ~50GB disk space for data
- PostgreSQL (optional, for full pipeline)

## Step 1: Process Raw Data into Annual Rollups

First, ensure you have the raw ERCOT data processed into annual rollup files:

```bash
# Process Day-Ahead prices, Real-Time prices, and Ancillary Services
make rollup-da-prices
make rollup-rt-prices  # Warning: This takes a while (500K+ files)
make rollup-as-prices
```

This creates annual parquet files in:
- `/home/enrico/data/ERCOT_data/rollup_files/DA_prices/`
- `/home/enrico/data/ERCOT_data/rollup_files/RT_prices/`
- `/home/enrico/data/ERCOT_data/rollup_files/AS_prices/`

## Step 2: Flatten Prices to Wide Format

Convert the data from long format to wide format where each row is an hour and each column is a settlement point:

```bash
make flatten-prices
```

This runs `flatten_ercot_prices.py` which creates:
- `flattened/DA_prices_YYYY.parquet` - 21 columns (datetime + 20 settlement points)
- `flattened/RT_prices_YYYY.parquet` - Same columns, hourly aggregated from 5-min
- `flattened/AS_prices_YYYY.parquet` - 6 columns (datetime + 5 AS types)

Output location: `/home/enrico/data/ERCOT_data/rollup_files/flattened/`

## Step 3: Combine and Split into Monthly Files

Merge different price types and create monthly breakdowns:

```bash
make combine-prices
```

This runs `combine_ercot_prices.py` which creates:
- `combined/DA_AS_combined_YYYY.parquet` - DA + AS prices
- `combined/DA_AS_RT_combined_YYYY.parquet` - All price types with prefixes
- `combined/monthly/*/` - Monthly breakdown files

## Step 4: Run the Complete Pipeline

To run all price processing steps at once:

```bash
make process-prices
```

This executes:
1. `flatten-prices` - Flatten to wide format
2. `combine-prices` - Combine and create monthly files

## Step 5: Start the Price Service

Build and run the Rust-based API service:

```bash
# Build and run the service
make run-price-service
```

The service will be available at:
- **JSON API**: http://localhost:8080
- **Arrow Flight**: grpc://localhost:50051

### Test the Service

```bash
# Check health
curl http://localhost:8080/api/health

# Get available hubs
curl http://localhost:8080/api/available_hubs

# Query prices
curl "http://localhost:8080/api/prices?start_date=2023-01-01T00:00:00Z&end_date=2023-01-07T23:59:59Z&hubs=HB_HOUSTON,HB_NORTH&price_type=day_ahead"
```

## Step 6: Docker Deployment (Optional)

For production deployment:

```bash
# Build Docker image
make price-service-docker

# Run with Docker Compose
make price-service-compose
```

## File Locations Summary

```
/home/enrico/data/ERCOT_data/rollup_files/
├── DA_prices/          # Original DA rollups (input)
├── RT_prices/          # Original RT rollups (input)
├── AS_prices/          # Original AS rollups (input)
├── flattened/          # Wide-format files (output of step 2)
│   ├── DA_prices_2023.parquet
│   ├── RT_prices_2023.parquet
│   └── AS_prices_2023.parquet
├── combined/           # Combined files (output of step 3)
│   ├── DA_AS_combined_2023.parquet
│   ├── DA_AS_RT_combined_2023.parquet
│   └── monthly/
│       ├── DA_AS_combined/
│       └── DA_AS_RT_combined/
└── specs/              # Documentation
```

## Python Scripts

The main Python scripts used are:

1. **`flatten_ercot_prices.py`** - Converts long to wide format
   - Input: `rollup_files/{DA,RT,AS}_prices/*.parquet`
   - Output: `rollup_files/flattened/*.parquet`

2. **`combine_ercot_prices.py`** - Combines price types and creates monthly files
   - Input: `rollup_files/flattened/*.parquet`
   - Output: `rollup_files/combined/*.parquet`

## Troubleshooting

### Missing Data
```bash
# Check if rollup files exist
ls -la /home/enrico/data/ERCOT_data/rollup_files/DA_prices/
```

### Service Won't Start
```bash
# Check data directory
ls -la /home/enrico/data/ERCOT_data/rollup_files/flattened/

# Run with debug logging
RUST_LOG=debug make run-price-service
```

### Python Module Errors
```bash
# Ensure dependencies are installed
uv sync
```

## Next Steps

- Integrate with Next.js frontend
- Connect Apache Superset to Arrow Flight endpoint
- Set up automatic daily processing with cron
- Configure monitoring and alerts

## Documentation

- API Specification: `ercot_price_service/specs/API_SPECIFICATION.md`
- Architecture: `ercot_price_service/specs/ARCHITECTURE.md`
- Data Specifications: `rollup_files/specs/`
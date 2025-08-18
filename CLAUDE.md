# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Build and Run Commands
```bash
# Python development (using uv package manager)
uv sync                                    # Install dependencies
uv run pmp --help                         # Show CLI help
uv run pytest                             # Run Python tests
uv run ruff format                        # Format Python code
uv run ruff check                         # Lint Python code
uv run mypy .                             # Type check Python code

# Rust development (in ercot_data_processor directory)
cd ercot_data_processor
cargo build --release                     # Build optimized binary
cargo run --release -- <args>             # Run Rust processor
cargo test                                # Run Rust tests
cargo clippy                              # Lint Rust code
cargo fmt                                 # Format Rust code

# Extract and process ERCOT data (uses ERCOT_DATA_DIR from .env)
cargo run --release -- --extract-all-ercot  # Uses ERCOT_DATA_DIR env var
SKIP_CSV=1 cargo run --release -- --process-annual  # Skip CSV, only Parquet

# Database commands
createdb power_market                     # Create database
uv run pmp init                          # Initialize database schema
uv run pmp-db                            # Run database setup script

# Data collection commands
uv run pmp download --iso ERCOT --days 7              # Download recent data
uv run pmp realtime --iso ERCOT                       # Start real-time updater
uv run pmp backfill --iso ERCOT --start 2019-01-01   # Historical backfill
uv run pmp catalog                                     # View data catalog
```

### Testing Individual Components
```bash
# Test specific modules
uv run python test_ercot_simple.py       # Test ERCOT downloader
uv run python test_mock_data.py          # Test with mock data
uv run python test_realtime_updater.py   # Test real-time updates
uv run python test_incremental_processor.py # Test incremental processing

# Run specific pytest tests
uv run pytest power_market_pipeline/tests/test_downloaders.py -v
uv run pytest -k "test_ercot" -v         # Run only ERCOT tests
```

## Architecture Overview

### Project Structure
```
power_market_pipeline/
├── downloaders/           # ISO-specific data downloaders
│   ├── base_v2.py        # Abstract base class defining downloader interface
│   ├── ercot/            # ERCOT: Selenium + WebService API
│   ├── caiso/            # CAISO: OASIS API
│   ├── isone/            # ISO-NE: Web Services API
│   └── nyiso/            # NYISO: Public CSV downloads
│
├── database/             # Database layer
│   ├── models_v2.py      # SQLAlchemy models (standardized schema)
│   ├── schema_v2.sql     # Core SQL schema
│   ├── migrations/       # Schema migrations
│   └── seeds/            # Reference data (ISOs, locations)
│
├── processors/           # Data transformation layer
│   └── *.py             # ISO-specific processors
│
├── services/            # Business logic layer
│   ├── data_fetcher.py  # Orchestrates downloads across ISOs
│   ├── realtime_updater.py # Real-time data collection scheduler
│   └── dataset_registry.py  # Dataset metadata management
│
├── power_market_pipeline/  # CLI package
│   └── cli.py             # Click-based command interface
│
└── ercot_data_processor/     # High-performance Rust data processor
    └── src/              # Various specialized processors
```

### Data Flow Architecture
1. **Downloaders** fetch raw data from ISO websites/APIs
2. **Processors** transform and standardize the data
3. **Database models** define the storage schema
4. **Services** orchestrate the pipeline operations
5. **CLI** provides user interface to all functionality

### Key Design Patterns
- **Standardized Schema**: All ISOs use same column names (iso, location, market, interval_start/end)
- **Interval-Based Time**: Every table uses interval_start/end for consistent temporal queries
- **Bulk Operations**: Data is processed in batches for performance
- **Async I/O**: Downloaders use async/await for concurrent operations
- **Retry Logic**: Built-in retry mechanisms for network failures

### ISO-Specific Implementation Details

#### ERCOT
- Historical data (before Dec 2023): Selenium web scraping
- Recent data: WebService API with subscription key
- Markets: DAM (hourly), RT5M (5-minute real-time)
- Special handling for Texas-specific terminology

#### CAISO
- OASIS API with authentication
- Markets: DAM, RT5M, RT15M, HASP
- Node-based pricing system

#### ISO-NE
- Web Services API
- Markets: DAM, RT5M
- Zone and node pricing

#### NYISO
- Public CSV downloads
- Markets: DAM, RT5M
- Zone-based pricing with LBMP terminology

### Database Schema Philosophy
- **TimescaleDB**: Optimized for time-series data with automatic partitioning
- **Hypertables**: Infinite scalability for high-frequency data
- **Compression**: Automatic compression of older data (90%+ savings)
- **ISO Views**: User-friendly views with domain-specific terminology
- **Standardized Columns**: Enables meta-programming across all tables

### Environment Variables Required
```bash
DATABASE_URL          # PostgreSQL connection string
ERCOT_DATA_DIR       # Path to ERCOT data directory (e.g. /home/enrico/data/ERCOT_data)
ERCOT_USERNAME       # ERCOT WebService credentials
ERCOT_PASSWORD
ERCOT_SUBSCRIPTION_KEY
CAISO_USERNAME       # CAISO OASIS credentials  
CAISO_PASSWORD
ISONE_USERNAME       # ISO-NE credentials
ISONE_PASSWORD
NYISO_USERNAME       # NYISO credentials (if needed)
NYISO_PASSWORD
```

### Common Development Tasks

#### Adding New Data Types
1. Define model in `database/models_v2.py`
2. Create SQL migration in `database/migrations/`
3. Add download method to base downloader
4. Implement in each ISO-specific downloader
5. Update data fetcher service
6. Add CLI command if needed

#### Debugging Download Issues
- Check credentials in .env file
- Enable SQL echo: `export SQL_ECHO=true`
- Use debug flag: `uv run pmp download --iso ERCOT --days 1 --debug`
- Check logs for specific error messages
- Verify network connectivity to ISO websites

#### Performance Optimization
- Use bulk inserts (already implemented)
- Enable TimescaleDB compression for historical data
- Create appropriate indexes for query patterns
- Use materialized views for aggregations
- Consider partitioning large tables by ISO

### Testing Strategy
- Unit tests for individual components
- Integration tests with mock data
- End-to-end tests with small date ranges
- Performance tests for bulk operations
- Always test with multiple ISOs to ensure standardization

### Rust Processor Usage
The `ercot_data_processor` is used for high-performance data processing:
- Handles large CSV/Excel files efficiently
- Converts between formats (CSV, Parquet, Arrow)
- Performs BESS (Battery Energy Storage) **HISTORICAL REVENUE ANALYSIS**
- Generates market reports and visualizations
- Automatic schema evolution handling (e.g., 2011 DSTFlag column addition)
- Forces all price columns to Float64 to prevent type mismatches
- Processes millions of records per year in seconds

### IMPORTANT: BESS Revenue Analysis Clarification
**We are doing HISTORICAL ANALYSIS and REVENUE ACCOUNTING, not optimization!**

These batteries already operated in ERCOT. They already made decisions, already got paid. Our job is to:
- **RECONSTRUCT what actually happened** from 60-day disclosure data
- **CALCULATE actual revenues earned** from DAM awards and RT operations
- **TRACK actual state of charge** from telemetered output
- **This is forensic accounting, NOT optimization**

We DO NOT need to:
- ❌ Optimize battery dispatch schedules
- ❌ Predict optimal arbitrage points
- ❌ Solve Linear Programming problems
- ❌ Make operational decisions

We DO need to:
- ✅ Read actual awards from `60d_DAM_Gen_Resource_Data-*.csv`
- ✅ Read actual dispatch from `60d_SCED_Gen_Resource_Data-*.csv`
- ✅ Calculate revenues: Awards × Prices = Revenue
- ✅ Track SOC from actual operations

### Recent Fixes and Improvements (August 2024)
1. **Float64 Type Enforcement**: All price columns now forced to Float64 at CSV reading stage
2. **Schema Evolution**: Handles mid-year column additions gracefully (normalize_dataframe)
3. **Disk Space Optimization**: SKIP_CSV=1 environment variable to output only Parquet files
4. **SPP File Separation**: DAM_Hourly_LMPs directory properly separates LMP and Settlement Point Price files
5. **Reduced Logging**: Only logs every 10th batch to reduce output verbosity
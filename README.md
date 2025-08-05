# Power Market Pipeline

A world-class data pipeline for downloading and storing historical and real-time energy market data from major US ISOs (Independent System Operators).

## Features

- **Multi-ISO Support**: ERCOT, CAISO, ISO-NE, NYISO (with extensible architecture for more)
- **Standardized Schema**: Consistent data model across all ISOs using interval-based timestamps
- **Real-Time Updates**: Smart polling system with 5-minute interval updates
- **Historical Backfill**: Download data back to January 1, 2019
- **TimescaleDB Ready**: Optimized for time-series data with compression support
- **High-Performance Rust Processor**: Process millions of records in seconds
- **Data Types Supported**:
  - Locational Marginal Prices (LMP) - Day-ahead and real-time
  - Settlement Point Prices (ERCOT)
  - Ancillary Services pricing
  - Load data and forecasts
  - Generation by fuel type (planned)
  - Transmission constraints (planned)
  - Weather data (planned)

## Quick Start

```bash
# Clone and install
git clone https://github.com/battalion-energy/power_market_pipeline.git
cd power_market_pipeline
pip install uv
uv sync

# Configure
cp .env.example .env
# Edit .env with your database URL and ISO credentials

# Initialize database
createdb power_market
uv run pmp init

# Download recent data
uv run pmp download --iso ERCOT --days 3

# Start real-time updates
uv run pmp realtime --iso ERCOT
```

## Installation

### Using uv (Recommended)

```bash
pip install uv
uv sync
```

### Using pip

```bash
pip install -e .
```

## Configuration

1. Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

2. Edit `.env` with your configuration:
```env
# Database (required)
DATABASE_URL=postgresql://localhost:5432/power_market

# ERCOT (optional - needed for WebService API after Dec 2023)
ERCOT_USERNAME=your_username
ERCOT_PASSWORD=your_password
ERCOT_SUBSCRIPTION_KEY=your_key

# Other ISOs (optional)
CAISO_USERNAME=your_username
CAISO_PASSWORD=your_password
```

## Database Setup

1. Create PostgreSQL database:
```bash
createdb power_market
```

2. Initialize schema:
```bash
uv run pmp init
```

This creates all tables and seeds initial data (ISOs, dataset categories).

## Usage

### Command Line Interface

The `pmp` (Power Market Pipeline) CLI provides all functionality:

```bash
# Show help
uv run pmp --help

# Initialize database
uv run pmp init

# Download data
uv run pmp download --iso ERCOT --days 7
uv run pmp download --iso CAISO --start 2024-01-01 --end 2024-01-31

# Start real-time updates
uv run pmp realtime --iso ERCOT
uv run pmp realtime --iso ERCOT --iso CAISO --data-types lmp,load

# Run historical backfill
uv run pmp backfill --iso ERCOT --start 2019-01-01

# View data catalog
uv run pmp catalog
uv run pmp catalog --iso ERCOT
```

### High-Performance Rust Processor

For processing large ERCOT datasets, use the Rust processor:

```bash
cd rt_rust_processor

# Build the processor
cargo build --release

# Extract all CSV files from ERCOT ZIP archives
cargo run --release -- --extract-all-ercot /path/to/ERCOT_data

# Process extracted data into annual Parquet files
SKIP_CSV=1 cargo run --release -- --process-annual

# Other commands
cargo run --release -- --dam              # Process DAM data
cargo run --release -- --ancillary        # Process ancillary services
cargo run --release -- --lmp              # Process LMP data
cargo run --release -- --bess             # Analyze BESS resources
```

Features:
- Processes millions of records in seconds
- Automatic schema evolution handling (e.g., 2011 DSTFlag addition)
- Forces all price columns to Float64 to prevent type mismatches
- Outputs compressed Parquet files (95%+ compression ratio)
- Handles nested ZIP extraction efficiently

### Real-Time Updates

The real-time updater runs continuously, fetching new data at 5-minute intervals:

```bash
# Update single ISO
uv run pmp realtime --iso ERCOT

# Update multiple ISOs
uv run pmp realtime --iso ERCOT --iso CAISO

# Use specialized ERCOT updater (polls every 5 seconds)
uv run pmp realtime --ercot-only
```

Features:
- Triggers exactly at 5-minute marks (00:00, 00:05, 00:10, etc.)
- Polls aggressively for new data when triggered
- Handles connection failures gracefully
- Comprehensive logging for monitoring

### Historical Backfill

Download historical data in chunks:

```bash
# Last 30 days
uv run pmp download --iso ERCOT --days 30

# Specific date range
uv run pmp download --iso ERCOT --start 2024-01-01 --end 2024-03-31

# Full historical (since Jan 1, 2019)
uv run pmp backfill --iso ERCOT --start 2019-01-01
```

Note: Full historical backfill requires:
- Valid API credentials for each ISO
- Significant storage space (~1GB compressed for all ISOs)
- Several hours to complete

## Data Schema

### Standardized Tables

**LMP (Locational Marginal Prices)**
```sql
CREATE TABLE lmp (
    interval_start TIMESTAMPTZ NOT NULL,
    interval_end TIMESTAMPTZ NOT NULL,
    iso VARCHAR(10) NOT NULL,
    location VARCHAR(100) NOT NULL,
    location_type VARCHAR(50),
    market VARCHAR(10) NOT NULL,
    lmp DECIMAL(10, 2),
    energy DECIMAL(10, 2),
    congestion DECIMAL(10, 2),
    loss DECIMAL(10, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Ancillary Services**
```sql
CREATE TABLE ancillary_services (
    interval_start TIMESTAMPTZ NOT NULL,
    interval_end TIMESTAMPTZ NOT NULL,
    iso VARCHAR(10) NOT NULL,
    region VARCHAR(100) NOT NULL,
    market VARCHAR(10) NOT NULL,
    product VARCHAR(50) NOT NULL,
    clearing_price DECIMAL(10, 2),
    clearing_quantity DECIMAL(10, 2),
    requirement DECIMAL(10, 2)
);
```

### Market Types
- `DAM`: Day-Ahead Market (hourly)
- `RT5M`: Real-Time 5-minute
- `RT15M`: Real-Time 15-minute
- `HASP`: Hour-Ahead Scheduling Process (CAISO)

### Location Types
- `hub`: Trading hubs
- `zone`: Load zones
- `node`: Individual nodes/buses

## Testing

### Run Tests with Mock Data

```bash
# Test basic functionality
uv run python test_mock_data.py

# Test historical periods (3 days, 1 month, historical)
uv run python test_with_mock_historical.py

# Test real-time updater
uv run python test_realtime_updater.py
```

### Test Results

The mock tests demonstrate:
- Database connectivity and schema
- Bulk insert performance (~5,000 records/second)
- Data quality validation
- Real-time update scheduling
- Historical data volume estimates

## Architecture

For detailed information about the database design, schema, and management practices, see [Database Design Documentation](database/docs/DATABASE_DESIGN.md).

For deployment instructions on AWS EC2, see [Deployment Guide](deployment/README.md).

```
power_market_pipeline/
├── downloaders/              # ISO-specific downloaders
│   ├── base_v2.py           # Base class defining interface
│   ├── ercot/
│   │   ├── downloader_v2.py # ERCOT implementation
│   │   ├── selenium_client.py
│   │   └── webservice_client.py
│   ├── caiso/
│   │   └── downloader_v2.py # CAISO implementation
│   ├── isone/
│   │   └── downloader_v2.py # ISO-NE implementation
│   └── nyiso/
│       └── downloader_v2.py # NYISO implementation
├── database/
│   ├── docs/                # Database documentation
│   │   └── DATABASE_DESIGN.md # Schema design philosophy
│   ├── migrations/          # Versioned schema changes
│   ├── seeds/               # Reference data
│   ├── scripts/             # Database management scripts
│   ├── utils/               # Migration and seed utilities
│   ├── models_v2.py         # SQLAlchemy models
│   ├── connection.py        # Database connection
│   └── schema_v2.sql        # SQL schema definition
├── services/
│   ├── data_fetcher.py      # Orchestrates downloads
│   ├── realtime_updater.py  # Real-time scheduler
│   └── dataset_registry.py  # Dataset metadata
├── processors/              # Data transformation (Python)
├── rt_rust_processor/       # High-performance Rust processor
│   ├── src/
│   │   ├── main.rs         # CLI entry point
│   │   ├── annual_processor.rs # Annual data aggregation
│   │   ├── csv_extractor.rs # ZIP/CSV extraction
│   │   └── ...             # Other processors
│   └── annual_output/      # Processed Parquet files
└── power_market_pipeline/
    └── cli.py              # Command-line interface
```

## Development

### Adding a New ISO

1. Create downloader in `downloaders/<iso>/downloader_v2.py`
2. Inherit from `BaseDownloaderV2`
3. Implement required methods:
   ```python
   async def download_lmp(self, market, start_date, end_date, locations=None)
   async def download_ancillary_services(self, product, market, start_date, end_date)
   async def download_load(self, forecast_type, start_date, end_date)
   async def get_available_locations(self)
   ```
4. Add ISO to seed data
5. Update tests

### Adding New Data Types

1. Create table in `database/models_v2.py`
2. Add to SQL schema
3. Create download method in base class
4. Implement in each ISO downloader
5. Update `DataFetcher` service

### Code Style

```bash
# Format code
uv run ruff format

# Lint
uv run ruff check

# Type check
uv run mypy .
```

## Deployment

### Systemd Service (Linux)

Create `/etc/systemd/system/power-market-realtime.service`:

```ini
[Unit]
Description=Power Market Pipeline Real-time Updater
After=network.target postgresql.service

[Service]
Type=simple
User=pmp
WorkingDirectory=/opt/power_market_pipeline
Environment="PATH=/opt/power_market_pipeline/.venv/bin"
ExecStart=/opt/power_market_pipeline/.venv/bin/python -m power_market_pipeline.cli realtime
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Docker (Coming Soon)

```bash
docker-compose up -d
```

### AWS EC2

1. Launch t3.large or larger
2. Install PostgreSQL 14+ with TimescaleDB
3. Clone repository
4. Configure environment
5. Set up systemd service
6. Configure CloudWatch/Dash0 monitoring

## Monitoring

### OpenTelemetry Support

Configure in `.env`:
```env
OTEL_EXPORTER_OTLP_ENDPOINT=https://your-endpoint
OTEL_EXPORTER_OTLP_HEADERS=Authorization=Bearer your-token
```

### Metrics Tracked
- Download success/failure rates
- Records processed per second
- Data quality metrics
- API response times
- Database performance

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure you're using `uv run` or have activated the virtual environment
2. **Database connection**: Check DATABASE_URL in .env
3. **No data downloading**: Most ISOs require API credentials
4. **Selenium errors**: Install Chrome/ChromeDriver for ERCOT historical data

### Debug Mode

```bash
# Enable SQL echo
export SQL_ECHO=true

# Run with debug logging
uv run pmp download --iso ERCOT --days 1 --debug
```

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Make changes with tests
4. Run linting (`uv run ruff check`)
5. Commit (`git commit -m 'Add amazing feature'`)
6. Push (`git push origin feature/amazing-feature`)
7. Open Pull Request

## License

Proprietary - Battalion Energy

## Support

- GitHub Issues: https://github.com/battalion-energy/power_market_pipeline/issues
- Documentation: See `/docs` folder
- Email: [contact email]
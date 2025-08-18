# ERCOT Price Service

High-performance Rust service for serving ERCOT electricity price data via Apache Arrow Flight and JSON REST API.

## Features

- **Apache Arrow Flight**: High-performance binary protocol for large datasets
- **JSON REST API**: Easy-to-use HTTP API for web applications
- **Data Caching**: In-memory caching for frequently accessed data
- **Multiple Price Types**: Day-Ahead, Real-Time, Ancillary Services, Combined
- **Hub Support**: All ERCOT hubs (HB_*), load zones (LZ_*), and DC ties (DC_*)
- **Time-based Queries**: Filter data by date range
- **Docker Support**: Containerized deployment

## Installation

### Requirements

- Rust 1.75+
- Cargo

### Build from Source

```bash
# Clone the repository
cd /home/enrico/projects/power_market_pipeline/ercot_price_service

# Build release version
./build.sh

# Or using Make
make release
```

### Docker Build

```bash
# Build Docker image
make docker

# Or using docker-compose
docker-compose build
```

## Usage

### Starting the Service

```bash
# Run directly
./target/release/ercot-price-server \
  --data-dir /home/enrico/data/ERCOT_data/rollup_files \
  --json-addr 0.0.0.0:8080 \
  --flight-addr 0.0.0.0:50051

# Or using Make
make run-release

# Or using Docker
docker-compose up
```

### API Endpoints

#### JSON REST API (Port 8080)

**Get Prices**
```bash
GET /api/prices?start_date=2023-01-01T00:00:00Z&end_date=2023-01-31T23:59:59Z&hubs=HB_HOUSTON,HB_NORTH&price_type=day_ahead
```

Parameters:
- `start_date`: ISO 8601 datetime
- `end_date`: ISO 8601 datetime
- `hubs`: Comma-separated list of hub codes
- `price_type`: One of `day_ahead`, `real_time`, `ancillary_services`, `combined`

**Health Check**
```bash
GET /api/health
```

**Available Hubs**
```bash
GET /api/available_hubs
```

#### Apache Arrow Flight (Port 50051)

Connect using any Arrow Flight client:

```python
import pyarrow.flight as flight

client = flight.connect("grpc://localhost:50051")

# List available flights
for flight_info in client.list_flights():
    print(flight_info)

# Get data
ticket = flight.Ticket(json.dumps({
    "start_date": "2023-01-01T00:00:00Z",
    "end_date": "2023-01-31T23:59:59Z",
    "hubs": ["HB_HOUSTON", "HB_NORTH"],
    "price_type": "day_ahead"
}).encode())

reader = client.do_get(ticket)
table = reader.read_all()
```

## Data Structure

The service expects data in the following directory structure:

```
/data/
├── flattened/
│   ├── DA_prices_2023.parquet
│   ├── RT_prices_2023.parquet
│   └── AS_prices_2023.parquet
├── combined/
│   └── DA_AS_RT_combined_2023.parquet
└── monthly/
    ├── DA_AS_combined/
    │   └── DA_AS_combined_2023_01.parquet
    └── DA_AS_RT_combined/
        └── DA_AS_RT_combined_2023_01.parquet
```

## Development

```bash
# Format code
make fmt

# Run linter
make lint

# Run tests
make test

# Watch mode (auto-rebuild on changes)
make watch

# Generate documentation
make docs
```

## Performance

- Handles millions of data points per second
- Sub-millisecond response times for cached data
- Efficient memory usage with Arrow columnar format
- Automatic data compression

## Integration

### Next.js Example

```typescript
async function fetchPrices(
  startDate: Date,
  endDate: Date,
  hubs: string[],
  priceType: string
) {
  const params = new URLSearchParams({
    start_date: startDate.toISOString(),
    end_date: endDate.toISOString(),
    hubs: hubs.join(','),
    price_type: priceType
  });

  const response = await fetch(`http://localhost:8080/api/prices?${params}`);
  return response.json();
}
```

### Apache Superset

Configure a new database connection:
- Type: Apache Arrow Flight
- Host: localhost
- Port: 50051

## License

MIT
# ERCOT Price Service Architecture

## System Overview

The ERCOT Price Service is a high-performance Rust-based data service that provides access to ERCOT electricity market prices through multiple protocols. It's designed for low latency, high throughput, and efficient memory usage.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                     Client Applications                   │
│  (Next.js, Python, Apache Superset, Grafana, etc.)       │
└─────────────┬───────────────────┬───────────────────────┘
              │                   │
              │ HTTP/JSON         │ gRPC/Arrow Flight
              │ Port 8080         │ Port 50051
              ▼                   ▼
┌─────────────────────────────────────────────────────────┐
│                   ERCOT Price Service                     │
├───────────────────────────────────────────────────────────┤
│  ┌─────────────┐        ┌──────────────────┐            │
│  │  JSON API   │        │  Flight Service  │            │
│  │   (Axum)    │        │     (Tonic)      │            │
│  └──────┬──────┘        └────────┬─────────┘            │
│         │                         │                       │
│         └───────────┬─────────────┘                      │
│                     ▼                                     │
│          ┌──────────────────────┐                        │
│          │   Price Data Cache   │                        │
│          │     (DashMap)        │                        │
│          └──────────┬───────────┘                        │
│                     ▼                                     │
│          ┌──────────────────────┐                        │
│          │    Data Loader       │                        │
│          │  (Parquet Reader)    │                        │
│          └──────────┬───────────┘                        │
└─────────────────────┼────────────────────────────────────┘
                      ▼
        ┌──────────────────────────┐
        │   Parquet Data Files     │
        │  /data/rollup_files/     │
        └──────────────────────────┘
```

## Core Components

### 1. Transport Layer

#### JSON API (Axum)
- **Framework**: Axum 0.7
- **Protocol**: HTTP/1.1, HTTP/2
- **Serialization**: JSON via serde_json
- **CORS**: Enabled for browser clients
- **Middleware**: Tower middleware stack

#### Arrow Flight Service (Tonic)
- **Framework**: Tonic 0.12 (gRPC)
- **Protocol**: gRPC over HTTP/2
- **Serialization**: Arrow IPC format
- **Streaming**: Supports large datasets
- **Compression**: Built-in gRPC compression

### 2. Business Logic Layer

#### Price Data Cache
- **Implementation**: DashMap (concurrent HashMap)
- **Strategy**: LRU-like (manual eviction)
- **Key**: File path + query parameters
- **Value**: Arrow RecordBatch
- **Concurrency**: Lock-free reads/writes

#### Data Loader
- **Reader**: Apache Parquet via arrow-rs
- **Async I/O**: Tokio file operations
- **Schema Evolution**: Handles column changes
- **Filtering**: Date range filtering
- **Projection**: Column selection

### 3. Data Layer

#### File Organization
```
/data/rollup_files/
├── flattened/          # Annual wide-format files
│   ├── DA_prices_YYYY.parquet
│   ├── RT_prices_YYYY.parquet
│   └── AS_prices_YYYY.parquet
├── combined/           # Multi-type merged files
│   └── monthly/        # Monthly breakdowns
└── [other directories]
```

#### Data Model
- **Storage**: Apache Parquet columnar format
- **Compression**: Snappy (default)
- **Schema**: Strongly typed Arrow schema
- **Indexing**: None (scan-based queries)

## Performance Optimizations

### Memory Management
1. **Zero-Copy Operations**: Arrow arrays passed by reference
2. **Lazy Loading**: Data loaded on first request
3. **Shared Immutable Data**: Arc<RecordBatch> for sharing
4. **Columnar Format**: Efficient for analytical queries

### Caching Strategy
```rust
Cache Key: "flattened/DA_prices_2023"
Cache Value: Arrow RecordBatch (columnar data)
```

- **Hit Path**: < 1ms response time
- **Miss Path**: 50-200ms (disk read + parse)
- **Eviction**: Manual or on service restart

### Concurrency Model
- **Async Runtime**: Tokio multi-threaded
- **Request Handling**: Concurrent, non-blocking
- **Cache Access**: Lock-free concurrent reads
- **File I/O**: Async, prevents blocking

## Data Flow

### JSON API Request Flow
```
1. HTTP Request → Axum Router
2. Parse Query Parameters
3. Validate Input (dates, hubs, price_type)
4. Generate Cache Key
5. Check Cache
   a. Hit: Return cached RecordBatch
   b. Miss: Load from Parquet file
6. Filter by Date Range
7. Extract Requested Columns
8. Convert to JSON Response
9. HTTP Response
```

### Arrow Flight Request Flow
```
1. gRPC Request → Tonic Service
2. Deserialize Flight Ticket
3. Parse Query from Ticket
4. Load Data (same as JSON)
5. Filter and Project
6. Serialize to Arrow IPC
7. Stream FlightData
8. gRPC Response
```

## Scaling Considerations

### Vertical Scaling
- **CPU**: Benefits from multiple cores
- **Memory**: ~2GB for full dataset cache
- **Disk**: SSD recommended for fast loads

### Horizontal Scaling
```
        Load Balancer
             │
    ┌────────┼────────┐
    ▼        ▼        ▼
Service  Service  Service
Instance Instance Instance
    │        │        │
    └────────┼────────┘
             ▼
      Shared Storage
       (NFS/S3/GCS)
```

### Deployment Options

#### Docker Standalone
```dockerfile
FROM rust:1.75 as builder
# Build binary
FROM debian:slim
# Run service
```

#### Kubernetes
```yaml
apiVersion: apps/v1
kind: Deployment
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: ercot-price-service
        image: ercot-price-service:latest
        ports:
        - containerPort: 8080  # JSON
        - containerPort: 50051 # Flight
```

#### Docker Compose
```yaml
services:
  price-service:
    image: ercot-price-service
    volumes:
      - /data:/data:ro
    ports:
      - "8080:8080"
      - "50051:50051"
```

## Monitoring & Observability

### Metrics (Planned)
- Request rate and latency
- Cache hit ratio
- Memory usage
- Active connections
- Error rates

### Logging
- **Framework**: tracing + tracing-subscriber
- **Levels**: ERROR, WARN, INFO, DEBUG, TRACE
- **Format**: JSON or human-readable
- **Filtering**: RUST_LOG environment variable

### Health Checks
- HTTP: `/api/health`
- gRPC: Standard gRPC health protocol
- Liveness: Process running
- Readiness: Can serve requests

## Security Considerations

### Current State
- No authentication/authorization
- Read-only data access
- No user input in file paths
- Input validation on all parameters

### Future Enhancements
1. **Authentication**: API keys, JWT tokens
2. **Authorization**: Role-based access
3. **TLS**: HTTPS and gRPC with TLS
4. **Rate Limiting**: Per-client limits
5. **Audit Logging**: Access logs

## Integration Patterns

### Next.js Integration
```typescript
// API Route Handler
export async function GET(request: Request) {
  const prices = await fetchFromPriceService(params);
  return Response.json(prices);
}
```

### Apache Superset
```python
# Custom DB Engine Spec
class ArrowFlightEngineSpec:
    def connect(self):
        return flight.connect("grpc://price-service:50051")
```

### Python Data Science
```python
import pyarrow.flight as flight
import pandas as pd

client = flight.connect("grpc://localhost:50051")
reader = client.do_get(ticket)
df = reader.read_pandas()
```

## Future Architecture Enhancements

### 1. Real-Time Updates
- WebSocket support for live prices
- Server-Sent Events (SSE)
- Pub/Sub with Redis/NATS

### 2. Data Pipeline Integration
- Direct connection to ERCOT APIs
- Automatic data refresh
- Change detection and notifications

### 3. Advanced Caching
- Redis for distributed cache
- Cache warming on startup
- Predictive cache loading

### 4. Query Engine
- SQL interface via DataFusion
- Complex aggregations
- Time-series specific operations

### 5. Multi-Region Deployment
- Edge caching with CDN
- Geo-distributed replicas
- Regional data partitioning
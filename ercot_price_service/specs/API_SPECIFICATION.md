# ERCOT Price Service API Specification

## Overview

The ERCOT Price Service provides high-performance access to ERCOT electricity market price data through two protocols:
- **REST/JSON API** (Port 8080) - Web-friendly HTTP interface
- **Apache Arrow Flight** (Port 50051) - High-performance binary protocol

## REST API Endpoints

### Base URL
```
http://localhost:8080/api
```

### 1. Get Prices
Retrieve price data for specified hubs, date range, and price type.

**Endpoint:** `GET /api/prices`

**Query Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| start_date | ISO 8601 | Yes | Start of date range (e.g., 2023-01-01T00:00:00Z) |
| end_date | ISO 8601 | Yes | End of date range (e.g., 2023-01-31T23:59:59Z) |
| hubs | string | Yes | Comma-separated hub codes (e.g., HB_HOUSTON,HB_NORTH) |
| price_type | string | Yes | One of: day_ahead, real_time, ancillary_services, combined |

**Response Format:**
```json
{
  "timestamps": [
    "2023-01-01T00:00:00Z",
    "2023-01-01T01:00:00Z"
  ],
  "data": [
    {
      "hub": "HB_HOUSTON",
      "prices": [45.23, 42.15, null, 48.90]
    },
    {
      "hub": "HB_NORTH", 
      "prices": [44.10, 41.20, 43.55, 47.80]
    }
  ]
}
```

**Example Request:**
```bash
curl "http://localhost:8080/api/prices?start_date=2023-01-01T00:00:00Z&end_date=2023-01-07T23:59:59Z&hubs=HB_HOUSTON,HB_NORTH,LZ_SOUTH&price_type=day_ahead"
```

### 2. Health Check
Check service health and status.

**Endpoint:** `GET /api/health`

**Response:**
```json
{
  "status": "healthy",
  "service": "ercot_price_service",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### 3. Available Hubs
Get list of all available hub codes.

**Endpoint:** `GET /api/available_hubs`

**Response:**
```json
[
  "HB_BUSAVG",
  "HB_HOUSTON",
  "HB_HUBAVG",
  "HB_NORTH",
  "HB_PAN",
  "HB_SOUTH",
  "HB_WEST",
  "LZ_AEN",
  "LZ_CPS",
  "LZ_HOUSTON",
  "LZ_LCRA",
  "LZ_NORTH",
  "LZ_RAYBN",
  "LZ_SOUTH",
  "LZ_WEST",
  "DC_E",
  "DC_L",
  "DC_N",
  "DC_R",
  "DC_S"
]
```

## Apache Arrow Flight API

### Connection
```
grpc://localhost:50051
```

### Flight Descriptor
The service exposes ERCOT price data as flights with the following schema:

**Schema:**
```
Field: datetime (Timestamp[ms])
Field: hub (Utf8)
Field: price (Float64, nullable)
```

### Operations

#### List Flights
Get available data flights.

```python
import pyarrow.flight as flight

client = flight.connect("grpc://localhost:50051")
for flight_info in client.list_flights():
    print(flight_info.descriptor.path)
```

#### Get Flight Info
Get metadata about a specific flight.

```python
descriptor = flight.FlightDescriptor.for_path("ercot_prices")
info = client.get_flight_info(descriptor)
print(f"Schema: {info.schema}")
print(f"Records: {info.total_records}")
```

#### Do Get (Retrieve Data)
Fetch price data using a ticket.

```python
import json

# Create query
query = {
    "start_date": "2023-01-01T00:00:00Z",
    "end_date": "2023-01-31T23:59:59Z",
    "hubs": ["HB_HOUSTON", "HB_NORTH"],
    "price_type": "day_ahead"
}

# Create ticket
ticket = flight.Ticket(json.dumps(query).encode())

# Fetch data
reader = client.do_get(ticket)
table = reader.read_all()

# Convert to pandas
df = table.to_pandas()
```

## Price Types

### day_ahead (DA)
- Day-Ahead Market energy prices
- Cleared day before delivery
- Hourly granularity
- All hubs, load zones, and DC ties

### real_time (RT)
- Real-Time energy prices
- 5-minute intervals aggregated to hourly
- More volatile than DA
- Same locations as DA

### ancillary_services (AS)
- Reserve product prices
- Includes: REGUP, REGDN, RRS, NSPIN, ECRS
- Hourly granularity
- System-wide prices (not locational)

### combined
- All price types in one response
- Columns prefixed: DA_, RT_, AS_
- Useful for spread analysis

## Error Responses

### HTTP Status Codes
- `200 OK` - Successful request
- `400 Bad Request` - Invalid parameters
- `404 Not Found` - Data not available
- `500 Internal Server Error` - Server error

### Error Response Format
```json
{
  "error": "Invalid price type",
  "details": "Valid types: day_ahead, real_time, ancillary_services, combined"
}
```

## Performance Characteristics

### Response Times
- Cached data: < 10ms
- First request: 50-200ms (loads from disk)
- Large date ranges: 100-500ms

### Data Limits
- Maximum date range: 1 year
- Maximum hubs per request: All (20)
- Concurrent requests: Unlimited

### Caching
- In-memory LRU cache
- Cache TTL: Until service restart
- Cache key: file path + date range

## Client Examples

### JavaScript/TypeScript
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

  const response = await fetch(
    `http://localhost:8080/api/prices?${params}`
  );
  
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  
  return await response.json();
}
```

### Python
```python
import requests
from datetime import datetime

def get_prices(start_date, end_date, hubs, price_type):
    params = {
        'start_date': start_date.isoformat() + 'Z',
        'end_date': end_date.isoformat() + 'Z',
        'hubs': ','.join(hubs),
        'price_type': price_type
    }
    
    response = requests.get(
        'http://localhost:8080/api/prices',
        params=params
    )
    response.raise_for_status()
    return response.json()

# Example usage
data = get_prices(
    datetime(2023, 1, 1),
    datetime(2023, 1, 31),
    ['HB_HOUSTON', 'HB_NORTH'],
    'day_ahead'
)
```

### cURL
```bash
# Get week of DA prices for Houston
curl -X GET "http://localhost:8080/api/prices?\
start_date=2023-01-01T00:00:00Z&\
end_date=2023-01-07T23:59:59Z&\
hubs=HB_HOUSTON&\
price_type=day_ahead"

# Get available hubs
curl http://localhost:8080/api/available_hubs

# Health check
curl http://localhost:8080/api/health
```

## WebSocket Support (Future)
Planned for real-time price updates:
- Subscribe to specific hubs
- Receive updates as new data arrives
- Automatic reconnection

## Rate Limiting
Currently no rate limiting. In production:
- Recommended: 100 requests/second per client
- Burst: 1000 requests
- Use caching on client side

## Authentication (Future)
Currently no authentication. Planned:
- API key authentication
- JWT tokens for session management
- Role-based access control
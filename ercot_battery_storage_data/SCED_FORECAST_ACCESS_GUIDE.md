# ERCOT SCED LMP Forecast Data - Access Guide

## Overview

ERCOT publishes **RTD Indicative LMPs** (SCED forecast prices) every 5 minutes showing predicted Locational Marginal Prices for the next 12 intervals (60 minutes ahead). This is the same dataset that gridstatus.io uses for their SCED LMP Forecast Analysis.

## Data Product Details

**Product ID**: NP6-970-CD
**Name**: RTD Indicative LMPs by Resource Nodes, Load Zones and Hubs
**Description**: Posted after every Look Ahead RTD run with indicative LMPs at Resource Nodes, Hub LMPs and Load Zones

**Update Frequency**: Every 5 minutes
**Forecast Horizon**: Next 12 intervals (~60 minutes)
**Historical Data**: Available from Dec 11, 2023 onwards

## API Access

### Registration Required

1. Visit https://apiexplorer.ercot.com/
2. Create an account
3. Obtain:
   - **Username** (for basic auth)
   - **Password** (for basic auth)
   - **Subscription Key** (required in headers)

### API Endpoint

**Base URL**: `https://api.ercot.com/api/public-reports/`
**Endpoint**: `/np6-970-cd/rtd_lmp_node_zone_hub`
**Method**: GET

### Query Parameters

- `RTDTimestampFrom` / `RTDTimestampTo`: Filter by RTD run timestamp
- `intervalEndingFrom` / `intervalEndingTo`: Filter by forecast interval time
- `settlementPoint`: Filter by specific settlement point name
- `settlementPointType`: Filter by type (HU=Hub, LZ=Load Zone, RN=Resource Node)
- `page` / `size`: Pagination (max 10,000 records per request)
- `sort` / `dir`: Sort results

## Using the Existing API Client

Your project already has an auto-generated API client at:
`ercot_webservices/ercot_api/`

### Example 1: Basic Usage

```python
from ercot_webservices.ercot_api.client import Client
from ercot_webservices.ercot_api.api.np6_970_cd import get_data_rtd_lmp_node_zone_hub
from datetime import datetime, timedelta

# Create client
client = Client(
    base_url="https://api.ercot.com/api/public-reports",
    headers={
        "Authorization": "Basic <base64(username:password)>",
    }
)

# Fetch latest forecast data
with client as client:
    response = get_data_rtd_lmp_node_zone_hub.sync(
        client=client,
        ocp_apim_subscription_key="YOUR_SUBSCRIPTION_KEY",
        size=1000  # Get up to 1000 records
    )

    if response:
        print(f"Retrieved {len(response.data)} forecast records")
        for record in response.data[:5]:
            print(f"{record.rtd_timestamp} - {record.settlement_point}: ${record.lmp:.2f}/MWh")
```

### Example 2: Fetch Forecasts for Specific Settlement Point

```python
from datetime import datetime, timedelta

# Get forecasts for HB_NORTH hub
now = datetime.now()
start_time = (now - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S")

with client as client:
    response = get_data_rtd_lmp_node_zone_hub.sync(
        client=client,
        ocp_apim_subscription_key="YOUR_SUBSCRIPTION_KEY",
        settlement_point="HB_NORTH",
        settlement_point_type="HU",  # Hub
        rtd_timestamp_from=start_time,
        size=10000
    )

    if response:
        print(f"HB_NORTH forecasts: {len(response.data)} records")
```

### Example 3: Continuous Collection (Cron-ready)

```python
#!/usr/bin/env python3
"""
Collect SCED LMP forecasts every 5 minutes
"""
import os
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from ercot_webservices.ercot_api.client import Client
from ercot_webservices.ercot_api.api.np6_970_cd import get_data_rtd_lmp_node_zone_hub

# Configuration
SUBSCRIPTION_KEY = os.getenv("ERCOT_SUBSCRIPTION_KEY")
USERNAME = os.getenv("ERCOT_USERNAME")
PASSWORD = os.getenv("ERCOT_PASSWORD")
DATA_DIR = Path("ercot_battery_storage_data/sced_forecasts")
DATA_DIR.mkdir(exist_ok=True)

def fetch_latest_forecasts():
    """Fetch latest SCED forecast data"""

    # Basic auth
    import base64
    auth_string = base64.b64encode(f"{USERNAME}:{PASSWORD}".encode()).decode()

    client = Client(
        base_url="https://api.ercot.com/api/public-reports",
        headers={"Authorization": f"Basic {auth_string}"}
    )

    with client as client:
        response = get_data_rtd_lmp_node_zone_hub.sync_detailed(
            client=client,
            ocp_apim_subscription_key=SUBSCRIPTION_KEY,
            size=10000
        )

        if response.parsed and hasattr(response.parsed, 'data'):
            return response.parsed.data
        return None

def save_forecasts(data):
    """Save forecast data to CSV"""
    if not data:
        return

    # Convert to DataFrame
    records = []
    for item in data:
        records.append({
            'rtd_timestamp': item.rtd_timestamp,
            'interval_ending': item.interval_ending,
            'settlement_point': item.settlement_point,
            'settlement_point_type': item.settlement_point_type,
            'lmp': item.lmp,
            'fetch_time': datetime.now().isoformat()
        })

    df = pd.DataFrame(records)

    # Save to timestamped file
    filename = f"forecasts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(DATA_DIR / filename, index=False)
    print(f"Saved {len(df)} forecast records to {filename}")

def main():
    print(f"Fetching SCED forecasts at {datetime.now()}")
    data = fetch_latest_forecasts()
    save_forecasts(data)

if __name__ == "__main__":
    main()
```

### Example 4: Async Version for High Performance

```python
import asyncio
from ercot_webservices.ercot_api.client import Client
from ercot_webservices.ercot_api.api.np6_970_cd import get_data_rtd_lmp_node_zone_hub

async def fetch_forecasts_async():
    """Async fetch for better performance"""
    client = Client(
        base_url="https://api.ercot.com/api/public-reports",
        headers={"Authorization": f"Basic {auth_string}"}
    )

    async with client as client:
        response = await get_data_rtd_lmp_node_zone_hub.asyncio(
            client=client,
            ocp_apim_subscription_key=SUBSCRIPTION_KEY,
            settlement_point_type="HU",  # Only hubs
            size=10000
        )
        return response

# Run async
data = asyncio.run(fetch_forecasts_async())
```

## Data Structure

### Response Format

```json
{
  "meta": {
    "totalRecords": 1234,
    "page": 1,
    "pageSize": 1000
  },
  "data": [
    {
      "RTDTimestamp": "2025-10-11T14:05:00-05:00",
      "intervalEnding": "2025-10-11T14:10:00-05:00",
      "intervalId": 285,
      "settlementPoint": "HB_NORTH",
      "settlementPointType": "HU",
      "LMP": 26.38,
      "repeatHourFlag": false
    },
    ...
  ]
}
```

### Key Fields

| Field | Description |
|-------|-------------|
| `RTDTimestamp` | When the forecast was published (RTD run time) |
| `intervalEnding` | The future interval being forecasted |
| `settlementPoint` | Location name (hub, zone, or resource node) |
| `settlementPointType` | HU=Hub, LZ=Load Zone, RN=Resource Node |
| `LMP` | Forecasted Locational Marginal Price ($/MWh) |
| `intervalId` | Interval number (1-288 for 5-min intervals) |

## Settlement Points of Interest

### Major Hubs
- **HB_NORTH** - North Hub
- **HB_SOUTH** - South Hub
- **HB_WEST** - West Hub
- **HB_HOUSTON** - Houston Hub

### Load Zones
- **LZ_NORTH** - North Zone
- **LZ_SOUTH** - South Zone
- **LZ_WEST** - West Zone
- **LZ_HOUSTON** - Houston Zone

## Integration with BESS Analysis

### Why This Data Matters for Batteries

SCED forecasts show where ERCOT *expects* prices to go in the next hour. Batteries can use this to:

1. **Arbitrage Optimization**: Forecast helps batteries decide when to charge/discharge
2. **Revenue Prediction**: Compare forecasts to actual settlement to calculate expected revenues
3. **Market Efficiency**: Analyze how well forecasts predict actual prices
4. **Dispatch Strategy**: Understand if batteries are responding to forecasts or actual prices

### Combining with BESS Operational Data

```python
import pandas as pd

# Load BESS operational data
bess_df = pd.read_csv('ercot_battery_storage_data/bess_catalog.csv',
                      parse_dates=['timestamp'])

# Load SCED forecasts
forecast_df = pd.read_csv('ercot_battery_storage_data/sced_forecasts/forecasts_*.csv',
                          parse_dates=['rtd_timestamp', 'interval_ending'])

# Filter for HB_NORTH hub
hub_forecasts = forecast_df[forecast_df['settlement_point'] == 'HB_NORTH']

# Merge: Find forecasts that were predicting the time when BESS operated
merged = pd.merge_asof(
    bess_df.sort_values('timestamp'),
    hub_forecasts.sort_values('interval_ending'),
    left_on='timestamp',
    right_on='interval_ending',
    direction='backward',
    tolerance=pd.Timedelta('5min')
)

# Analyze: Did batteries charge when forecast predicted low prices?
print("Correlation between forecast prices and charging behavior:")
print(merged[['net_output_mw', 'lmp']].corr())
```

## Cron Setup for Continuous Collection

Add to crontab for 5-minute collection:

```bash
*/5 * * * * nice -n 19 python3 /path/to/collect_sced_forecasts.py >> /path/to/sced_forecasts.log 2>&1
```

## Alternative Access Methods

### 1. Data Portal (Manual Download)
- URL: https://data.ercot.com/data-product-details/NP6-970-CD
- Download historical data as CSV/XML
- Good for bulk historical analysis

### 2. Screen Scraping (Not Recommended)
- ERCOT publishes real-time displays at:
  - https://www.ercot.com/content/cdr/html/rtd_ind_lmp_lz_hb.html
- Updated every 5 minutes
- Contains latest forecasts in HTML tables
- Less reliable than API, but doesn't require registration

### 3. gridstatus Library (Third-Party)
```python
from gridstatus import ERCOT

ercot = ERCOT()
forecasts = ercot.get_sced_lmp_forecast()
```
Note: Requires gridstatus package and may have subscription requirements

## Comparison: Forecast vs Actual Prices

To analyze forecast accuracy, you'll also need actual SCED LMP data:

**Actual SCED LMPs**: NP6-788-CD
**Endpoint**: `/np6-788-cd/lmp_node_zone_hub`

Compare forecast at time T for interval T+n vs actual LMP at interval T+n.

## Environment Variables

Add to your `.env` file:

```bash
# ERCOT API Credentials
ERCOT_SUBSCRIPTION_KEY=your_subscription_key_here
ERCOT_USERNAME=your_username
ERCOT_PASSWORD=your_password
```

## Data Retention

**API**: Latest ~24-48 hours available via API
**Historical**: Download from data portal (since Dec 11, 2023)
**Your Collection**: Build your own historical archive with continuous collection

## Resources

- **ERCOT API Explorer**: https://apiexplorer.ercot.com/
- **Data Portal**: https://data.ercot.com/
- **Product Details**: https://www.ercot.com/mp/data-products/data-product-details?id=NP6-970-CD
- **Developer Portal**: https://developer.ercot.com/
- **Gridstatus.io**: https://www.gridstatus.io/live/ercot (reference implementation)

## Support

For API issues:
- ERCOT API Support: apiexplorer@ercot.com
- API Explorer Support Page: https://apiexplorer.ercot.com/support

---

**Status**: Production Ready
**Data Available**: Since Dec 11, 2023
**Update Frequency**: Every 5 minutes
**Access Required**: Free registration at apiexplorer.ercot.com
**Existing Client**: ✅ Already in your project!

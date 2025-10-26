# ENTSO-E European Electricity Market Data Pipeline

This package downloads and processes electricity market data from the ENTSO-E Transparency Platform for European markets, with a primary focus on Germany for BESS (Battery Energy Storage System) projects.

## Overview

### Data Sources

1. **ENTSO-E Transparency Platform API** (Primary)
   - Day-ahead market prices (hourly)
   - Imbalance prices (15-minute for Germany)
   - Balancing energy (activated reserves)
   - Load and generation data

2. **Regelleistung.net** (Germany-specific, future implementation)
   - FCR (Frequency Containment Reserve)
   - aFRR (automatic Frequency Restoration Reserve)
   - mFRR (manual Frequency Restoration Reserve)
   - Tender results and activation prices

### Supported Markets

**Priority 1 (Focus Markets):**
- 🇩🇪 Germany-Luxembourg (DE_LU) - Primary focus
- 🇫🇷 France (FR)
- 🇳🇱 Netherlands (NL)
- 🇧🇪 Belgium (BE)
- 🇦🇹 Austria (AT)
- 🇨🇭 Switzerland (CH)
- 🇮🇹 Italy North (IT_NORTH)

**Priority 2 & 3:** All other European bidding zones (40+ zones total)

See `european_zones.py` for complete list.

## Setup

### 1. Register for ENTSO-E API Access

1. Register at: https://transparency.entsoe.eu/
2. Email: `transparency@entsoe.eu`
3. Subject: "Restful API access"
4. Include: Your registered email address
5. Wait: ~2-3 business days for API key
6. **Cost:** FREE

### 2. Install Dependencies

```bash
# From project root
pip install entsoe-py
```

### 3. Set Environment Variables

Add to your `.env` file:

```bash
# ENTSO-E API credentials
ENTSO_E_API_KEY=your_api_key_here

# Data directory
ENTSO_E_DATA_DIR=/pool/ssd8tb/data/iso/ENTSO_E
```

### 4. Create Data Directory Structure

```bash
mkdir -p $ENTSO_E_DATA_DIR/csv_files/{da_prices,imbalance_prices,balancing_energy}
```

## Usage

### Quick Start: Download Germany Data

```bash
# Download day-ahead prices for Germany (2024)
python iso_markets/entso_e/download_da_prices.py \
    --zones DE_LU \
    --start-date 2024-01-01 \
    --end-date 2024-12-31

# Download imbalance prices (15-minute resolution)
python iso_markets/entso_e/download_imbalance_prices.py \
    --zones DE_LU \
    --start-date 2024-01-01 \
    --end-date 2024-12-31
```

### Auto-Resume Updater (Daily Cron)

The auto-resume updater is the recommended way to keep data up-to-date:

```bash
# Update Germany - automatically resumes from last date
python iso_markets/entso_e/update_entso_e_with_resume.py --zones DE_LU

# Update all priority 1 zones
python iso_markets/entso_e/update_entso_e_with_resume.py --priority 1

# Dry run to see what would be updated
python iso_markets/entso_e/update_entso_e_with_resume.py --zones DE_LU --dry-run

# Force start from specific date
python iso_markets/entso_e/update_entso_e_with_resume.py \
    --zones DE_LU \
    --start-date 2024-01-01

# Update only day-ahead prices
python iso_markets/entso_e/update_entso_e_with_resume.py \
    --zones DE_LU \
    --data-types da_prices
```

### Download Multiple Zones

```bash
# Download priority 1 markets
python iso_markets/entso_e/download_da_prices.py \
    --priority 1 \
    --start-date 2024-01-01 \
    --end-date 2024-12-31

# Download specific zones
python iso_markets/entso_e/download_da_prices.py \
    --zones DE_LU FR NL BE \
    --start-date 2024-01-01 \
    --end-date 2024-12-31
```

### Python API

```python
from iso_markets.entso_e import ENTSOEAPIClient, get_germany_zone
import pandas as pd

# Initialize client
client = ENTSOEAPIClient()

# Download day-ahead prices for Germany
zone = get_germany_zone()
start = pd.Timestamp('2024-01-01', tz=zone.timezone)
end = pd.Timestamp('2024-12-31', tz=zone.timezone)

prices = client.query_day_ahead_prices('DE_LU', start, end)
print(prices.head())

# Download imbalance prices
imbalance = client.query_imbalance_prices('DE_LU', start, end)
print(imbalance.head())
```

## Data Schema

### Day-Ahead Prices

```
datetime_utc              | price_eur_per_mwh
2024-01-01 00:00:00+00:00| 85.50
2024-01-01 01:00:00+00:00| 82.30
```

- **Temporal Resolution:** Hourly (24 hours/day)
- **Timezone:** UTC (converted from local time)
- **Coverage:** Single bidding zone price

### Imbalance Prices

```
datetime_utc              | imbalance_price_eur_per_mwh (or short_price / long_price)
2024-01-01 00:00:00+00:00| 95.20
2024-01-01 00:15:00+00:00| 93.50
```

- **Temporal Resolution:** 15-minute (Germany), varies by zone
- **Timezone:** UTC
- **Note:** Not all zones publish imbalance prices

### File Naming Convention

```
da_prices_DE_LU_2024-01-01_2024-01-31.csv
imbalance_prices_DE_LU_2024-01-01_2024-01-31.csv
```

Format: `{data_type}_{zone}_{start_date}_{end_date}.csv`

## Cron Job Setup

For daily automated updates:

```bash
# Edit crontab
crontab -e

# Add daily update at 10:00 AM (after market data publication)
0 10 * * * cd /home/enrico/projects/power_market_pipeline && \
           /usr/bin/python3 -m iso_markets.entso_e.update_entso_e_with_resume \
           --zones DE_LU --end-date yesterday >> /var/log/entso_e_update.log 2>&1
```

## BESS Revenue Optimization - German Market

### Key Markets for BESS Projects

1. **aFRR (automatic Frequency Restoration Reserve)**
   - **Biggest opportunity** for batteries in Germany (per Modo Energy Sept 2025)
   - 4-hour product blocks (6 blocks/day)
   - Pay-as-bid capacity prices
   - High price volatility = optimization opportunity
   - Data source: Regelleistung.net (to be implemented)

2. **Day-Ahead Energy Arbitrage**
   - Buy low at night, sell high during day
   - Available via ENTSO-E: ✅ Implemented

3. **Imbalance Energy**
   - Provide balancing when system is short/long
   - 15-minute resolution
   - Available via ENTSO-E: ✅ Implemented

4. **FCR & mFRR**
   - Additional ancillary service opportunities
   - Data source: Regelleistung.net (to be implemented)

### European Balancing Platforms

- **MARI:** Manual Frequency Restoration Reserve integration
- **PICASSO:** Automatic Frequency Restoration Reserve integration
- **TERRE:** Replacement Reserve platform

These platforms enable cross-border balancing energy trading.

## Technical Details

### Rate Limiting

- **Max Requests:** 10 per minute (conservative)
- **Min Delay:** 1.0 seconds between requests
- **Pattern:** Sliding window rate limiter (based on PJM pattern)

### Timezone Handling

⚠️ **CRITICAL:** ENTSO-E API returns data in local time (CET/CEST for Germany).
All data is automatically converted to UTC for consistency.

### Chunk Sizes

- **Day-Ahead Prices:** 90-day chunks (hourly data)
- **Imbalance Prices:** 30-day chunks (15-minute data)
- Automatically handles date chunking to avoid API limits

### Error Handling

- Continues downloading other chunks if one fails
- Logs all errors with details
- Returns partial data if available

## Comparison to US ISOs

| Aspect | US Markets (PJM/ERCOT) | European (ENTSO-E) |
|--------|------------------------|-------------------|
| **Geographic Coverage** | Nodal (thousands) | Zonal (single price) |
| **DA Resolution** | Hourly | Hourly |
| **RT Resolution** | 5-minute | 15-minute (imbalance) |
| **Data Volume** | Large (nodal) | Small (zonal) |
| **API Access** | Free (PJM requires key) | Free (requires key) |
| **Ancillary Services** | API available | Web scraping needed (Germany) |

## Future Enhancements

1. **Regelleistung.net Scraper** - FCR/aFRR/mFRR data
2. **Balancing Energy Download** - Activated reserves
3. **Load & Generation** - Market context data
4. **Parquet Conversion Pipeline** - Efficient storage
5. **Multi-zone Combined Dataset** - Regional analysis
6. **Instantaneous Reserve Market** - New German product

## Troubleshooting

### API Key Issues

```
ValueError: API key required
```

**Solution:** Set `ENTSO_E_API_KEY` environment variable or pass to client.

### No Data Returned

Some zones don't publish certain data types (especially imbalance prices).

**Solution:** Check ENTSO-E Transparency Platform web interface to verify data availability.

### Rate Limiting

If you see many "Rate limit reached" messages, reduce `requests_per_minute`:

```python
client = ENTSOEAPIClient(requests_per_minute=5)  # Slower but safer
```

## References

- [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/)
- [ENTSO-E API Guide](https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Guide.html)
- [entsoe-py GitHub](https://github.com/EnergieID/entsoe-py)
- [Regelleistung.net](https://www.regelleistung.net/)
- [Modo Energy - Germany aFRR Report](https://modoenergy.com/research/germany-september-2025-afrr-explained)

## Support

For issues or questions:
1. Check logs in data directory
2. Verify API key is valid
3. Test with small date range first
4. Check ENTSO-E platform status

---

**Version:** 0.1.0
**Last Updated:** 2025-10-25
**Primary Focus:** Germany (DE_LU) for BESS projects
**Coverage:** All European bidding zones

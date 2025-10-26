# ENTSO-E European Market Data - Quick Start Guide

## 🚀 Get Started in 5 Minutes

This guide will get you downloading German electricity market data for your BESS projects.

## Step 1: Request API Access (Do This First!)

While you're waiting for the API key, you can set up everything else.

1. Register at: https://transparency.entsoe.eu/
2. Email: `transparency@entsoe.eu`
3. Subject: **"Restful API access"**
4. Body:
   ```
   Hello,

   I would like to request API access to the ENTSO-E Transparency Platform.

   Registered email: your_email@example.com

   Thank you.
   ```
5. Wait: ~2-3 business days

## Step 2: Install Dependencies

```bash
cd /home/enrico/projects/power_market_pipeline

# Install new dependencies
uv add entsoe-py lxml

# Or if using pip
pip install entsoe-py lxml
```

## Step 3: Configure Environment

Add to your `.env` file (create if it doesn't exist):

```bash
# ENTSO-E API credentials
ENTSO_E_API_KEY=your_key_will_go_here_when_you_get_it

# Data directory (already exists)
ENTSO_E_DATA_DIR=/pool/ssd8tb/data/iso/ENTSO_E
```

## Step 4: Create Data Directories

```bash
mkdir -p /pool/ssd8tb/data/iso/ENTSO_E/csv_files/{da_prices,imbalance_prices,balancing_energy}
```

## Step 5: Test Your Setup (When You Get API Key)

Once you receive your API key, test the connection:

```bash
cd /home/enrico/projects/power_market_pipeline

# Test API connection
python -m iso_markets.entso_e.entso_e_api_client --test
```

Expected output:
```
API CONNECTION TEST SUCCESSFUL
================================================================================

Retrieved 24 records

Sample data:
                          price_eur_per_mwh
datetime_utc
2024-01-01 00:00:00+00:00              85.50
2024-01-01 01:00:00+00:00              82.30
...
```

## Step 6: Download Historical Data for Germany

### Option A: Download 2024 Data

```bash
# Day-ahead prices (hourly)
python iso_markets/entso_e/download_da_prices.py \
    --zones DE_LU \
    --start-date 2024-01-01 \
    --end-date 2024-12-31

# Imbalance prices (15-minute)
python iso_markets/entso_e/download_imbalance_prices.py \
    --zones DE_LU \
    --start-date 2024-01-01 \
    --end-date 2024-12-31
```

**Time estimate:** ~10-15 minutes for full year of both data types

### Option B: Download Recent Data (Last 30 Days)

```bash
# Calculate date 30 days ago
START_DATE=$(date -d '30 days ago' +%Y-%m-%d)
END_DATE=$(date -d 'yesterday' +%Y-%m-%d)

# Download day-ahead and imbalance prices
python iso_markets/entso_e/download_da_prices.py \
    --zones DE_LU \
    --start-date $START_DATE \
    --end-date $END_DATE

python iso_markets/entso_e/download_imbalance_prices.py \
    --zones DE_LU \
    --start-date $START_DATE \
    --end-date $END_DATE
```

**Time estimate:** ~2-3 minutes

## Step 7: Set Up Daily Auto-Updates

For production use, set up a cron job:

```bash
# Edit crontab
crontab -e

# Add this line (runs daily at 10:00 AM):
0 10 * * * cd /home/enrico/projects/power_market_pipeline && \
           /usr/bin/python3 -m iso_markets.entso_e.update_entso_e_with_resume \
           --zones DE_LU --end-date yesterday >> /var/log/entso_e_update.log 2>&1
```

This will:
- Automatically detect the last downloaded date
- Download any missing data up to yesterday
- Handle gaps if the cron job fails for a few days
- Log all activity

## Verify Your Data

Check that data was downloaded:

```bash
ls -lh /pool/ssd8tb/data/iso/ENTSO_E/csv_files/da_prices/
ls -lh /pool/ssd8tb/data/iso/ENTSO_E/csv_files/imbalance_prices/
```

You should see files like:
```
da_prices_DE_LU_2024-01-01_2024-01-31.csv
imbalance_prices_DE_LU_2024-01-01_2024-01-31.csv
```

## Quick Python Example

```python
import pandas as pd
from pathlib import Path

# Read day-ahead prices
data_dir = Path('/pool/ssd8tb/data/iso/ENTSO_E/csv_files/da_prices')
csv_file = list(data_dir.glob('da_prices_DE_LU_*.csv'))[0]

df = pd.read_csv(csv_file, index_col='datetime_utc', parse_dates=True)
print(f"Loaded {len(df)} hourly price records")
print(f"\nPrice statistics (EUR/MWh):")
print(df['price_eur_per_mwh'].describe())
print(f"\nSample data:")
print(df.head())
```

## Next Steps

### For BESS Revenue Optimization

1. **Combine DA and Imbalance Prices**
   - Analyze arbitrage opportunities
   - Identify high-volatility periods

2. **Add Ancillary Services Data** (Next Phase)
   - FCR, aFRR, mFRR from Regelleistung.net
   - This is where the biggest opportunities are!

3. **Expand to Other Markets**
   - Download France, Netherlands, Belgium, etc.
   - Compare cross-border opportunities

### Download More Zones

```bash
# All priority 1 markets (Germany, France, Netherlands, Belgium, Austria, Switzerland, Italy)
python iso_markets/entso_e/download_da_prices.py \
    --priority 1 \
    --start-date 2024-01-01 \
    --end-date 2024-12-31

# Specific zones
python iso_markets/entso_e/download_da_prices.py \
    --zones DE_LU FR NL BE \
    --start-date 2024-01-01 \
    --end-date 2024-12-31
```

## Troubleshooting

### "API key required" Error

Make sure `ENTSO_E_API_KEY` is set in your `.env` file and the file is in the project root.

### "No data returned"

Some zones don't publish certain data types. Germany (DE_LU) has the best coverage.

### Rate Limiting

If downloads are slow, this is normal. The rate limiter prevents API throttling:
- Max 10 requests/minute
- 1 second minimum between requests

### Check Logs

The scripts log all activity. Check for errors:

```bash
# For manual downloads, output is on screen
# For cron jobs, check the log file:
tail -f /var/log/entso_e_update.log
```

## Data Volume Reference

Typical file sizes for Germany (DE_LU):

| Data Type | Resolution | Days | File Size | Records |
|-----------|-----------|------|-----------|---------|
| DA Prices | Hourly | 365 | ~500 KB | 8,760 |
| Imbalance | 15-min | 365 | ~2 MB | 35,040 |
| **Total/Year** | | | **~2.5 MB** | **43,800** |

Compare to PJM nodal data: ~100 GB/year (much larger due to thousands of nodes)

## Available Zones

See all configured zones:

```bash
python -c "from iso_markets.entso_e import BIDDING_ZONES, get_priority_1_zones; \
           print('Priority 1 Markets:'); \
           [print(f'  {z.name} ({k})') for k, z in BIDDING_ZONES.items() if z.priority == 1]"
```

## Support

- Documentation: `iso_markets/entso_e/README.md`
- API Client: `iso_markets/entso_e/entso_e_api_client.py`
- Zone Config: `iso_markets/entso_e/european_zones.py`

---

**Ready to start?** Begin with Step 1 (request API access) while you set up Steps 2-4! 🚀

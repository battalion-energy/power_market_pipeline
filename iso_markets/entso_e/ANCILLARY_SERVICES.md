# German Ancillary Services Data (Regelleistung.net)

## Overview

This document covers the download and processing of German ancillary services data from Regelleistung.net, the official platform for balancing capacity and energy tenders in Germany.

**Key Insight:** aFRR (automatic Frequency Restoration Reserve) represents the **BIGGEST OPPORTUNITY** for BESS projects in Germany according to industry analysis (Modo Energy, Sept 2025).

## Products Available

### 1. FCR (Frequency Containment Reserve)
- **Also known as:** Primary Control Reserve
- **Activation time:** 30 seconds
- **Purpose:** Immediate frequency stabilization
- **Bidding structure:** 6 x 4-hour blocks per day
- **Direction:** Combined positive/negative (NEGPOS)
- **Markets:** CAPACITY only

**BESS Considerations:**
- ✓ Fast response required (30 seconds)
- ✓ Symmetric bid (must provide up and down)
- ⚠ Requires high-performance BMS and inverters

### 2. aFRR (automatic Frequency Restoration Reserve) ⭐
- **Also known as:** Secondary Control Reserve
- **Activation time:** 5 minutes
- **Purpose:** Automatic frequency restoration after FCR
- **Bidding structure:** 6 x 4-hour blocks per day
- **Direction:** Separate POS (positive/up) and NEG (negative/down)
- **Markets:** CAPACITY and ENERGY

**BESS Opportunity:**
- ⭐ **HIGHEST revenue potential** for batteries
- ✓ 4-hour duration requirement matches typical BESS
- ✓ High price volatility = optimization opportunity
- ✓ Pay-as-bid pricing
- ✓ Can bid separately for POS and NEG
- ✓ Energy revenue on top of capacity revenue

**Why aFRR is Best:**
1. Price volatility creates arbitrage opportunities
2. 4-hour blocks align with BESS capabilities
3. Separate POS/NEG allows for strategic bidding
4. Both capacity and energy revenue streams
5. Growing market as renewable penetration increases

### 3. mFRR (manual Frequency Restoration Reserve)
- **Also known as:** Tertiary Reserve, Minute Reserve
- **Activation time:** 15 minutes (originally), now shorter
- **Purpose:** Manual frequency restoration, longer-term balancing
- **Bidding structure:** 6 x 4-hour blocks per day
- **Direction:** Separate POS and NEG
- **Markets:** CAPACITY and ENERGY

**BESS Considerations:**
- ✓ Slower activation allows wider range of assets
- ✓ Good supplementary revenue stream
- ✓ Less competitive than aFRR (more participants)

## 4-Hour Block Structure

All products use 6 x 4-hour blocks per day:

| Block | Time (CET) | Product Names |
|-------|-----------|---------------|
| 1 | 00:00 - 04:00 | FCR: NEGPOS_00_04<br>aFRR: POS_00_04, NEG_00_04<br>mFRR: POS_00_04, NEG_00_04 |
| 2 | 04:00 - 08:00 | _04_08 |
| 3 | 08:00 - 12:00 | _08_12 |
| 4 | 12:00 - 16:00 | _12_16 |
| 5 | 16:00 - 20:00 | _16_20 |
| 6 | 20:00 - 24:00 | _20_24 |

**Block Characteristics:**
- Each block requires 4 hours of continuous availability
- Prices vary significantly by time of day
- Morning/evening peaks typically have higher prices
- Overnight blocks often have lower demand

## Data Types

### Capacity Market
- **What it is:** Procurement of standby capacity (€/MW per hour)
- **Pricing:** Pay-as-bid (awarded bidders get their bid price)
- **Frequency:** Daily auction (d-7, awarded 7 days ahead)
- **Key fields:**
  - `GERMANY_MARGINAL_CAPACITY_PRICE_[(EUR/MW)/h]` - Highest accepted bid
  - `GERMANY_AVERAGE_CAPACITY_PRICE_[(EUR/MW)/h]` - Average of accepted bids
  - `GERMANY_MIN_CAPACITY_PRICE_[(EUR/MW)/h]` - Lowest accepted bid
  - `GERMANY_IMPORT(-)_EXPORT(+)_[MW]` - Cross-border flows

### Energy Market
- **What it is:** Actual activation/dispatch prices (€/MWh)
- **Pricing:** Market-based, varies by quarter-hour
- **Frequency:** Real-time / post-delivery
- **Key fields:**
  - `GERMANY_MARGINAL_ENERGY_PRICE_[EUR/MWh]` - Highest activation price
  - `GERMANY_AVERAGE_ENERGY_PRICE_[EUR/MWh]` - Average activation price
  - `GERMANY_MIN_ENERGY_PRICE_[EUR/MWh]` - Lowest activation price

## Data Schema

### FCR Capacity
```
DATE_FROM                                      datetime
DATE_TO                                        datetime
PRODUCT_TYPE                                   FCR
PRODUCTNAME                                    NEGPOS_00_04, NEGPOS_04_08, etc.
GERMANY_DEMAND_[MW]                           float
GERMANY_SETTLEMENTCAPACITY_PRICE_[EUR/MW]     float
GERMANY_DEFICIT(-)_SURPLUS(+)_[MW]            float
```

### aFRR Capacity
```
DATE_FROM                                      datetime
DATE_TO                                        datetime
TYPE_OF_RESERVES                               aFRR
PRODUCT                                        POS_00_04, NEG_00_04, etc.
GERMANY_MARGINAL_CAPACITY_PRICE_[(EUR/MW)/h]  float
GERMANY_AVERAGE_CAPACITY_PRICE_[(EUR/MW)/h]   float
GERMANY_MIN_CAPACITY_PRICE_[(EUR/MW)/h]       float
GERMANY_IMPORT(-)_EXPORT(+)_[MW]              float
GERMANY_SUM_OF_OFFERED_CAPACITY_[MW]          float
```

### aFRR Energy
```
DELIVERY_DATE                                  datetime
TYPE_OF_RESERVES                               aFRR
PRODUCT                                        POS/NEG + quarter-hour
GERMANY_MARGINAL_ENERGY_PRICE_[EUR/MWh]       float
GERMANY_AVERAGE_ENERGY_PRICE_[EUR/MWh]        float
GERMANY_MIN_ENERGY_PRICE_[EUR/MWh]            float
GERMANY_SUM_OF_OFFERED_CAPACITY_[MW]          float
```

## Usage Examples

### Download All Ancillary Services for 2024

```bash
cd /home/enrico/projects/power_market_pipeline

# Download all products (FCR, aFRR, mFRR) and both markets
python iso_markets/entso_e/download_ancillary_services.py \
    --start-date 2024-01-01 \
    --end-date 2024-12-31

# Time estimate: ~2 hours (365 days × 5 combinations × 2 sec/request)
```

### Download Only aFRR (Recommended for BESS Focus)

```bash
# Download only aFRR capacity and energy
python iso_markets/entso_e/download_ancillary_services.py \
    --products aFRR \
    --start-date 2024-01-01 \
    --end-date 2024-12-31

# Time estimate: ~25 minutes (365 days × 2 markets × 2 sec)
```

### Download Recent Data (Last 30 Days)

```bash
# Quick update for recent market activity
START_DATE=$(date -d '30 days ago' +%Y-%m-%d)
END_DATE=$(date -d 'yesterday' +%Y-%m-%d)

python iso_markets/entso_e/download_ancillary_services.py \
    --products aFRR \
    --start-date $START_DATE \
    --end-date $END_DATE

# Time estimate: ~2 minutes
```

### Download Historical Data (Available from 2012+)

```bash
# Download multiple years of aFRR data for analysis
python iso_markets/entso_e/download_ancillary_services.py \
    --products aFRR \
    --start-date 2020-01-01 \
    --end-date 2024-12-31

# Time estimate: ~2.5 hours (5 years of data)
```

### Python API Usage

```python
from iso_markets.entso_e import RegelleistungAPIClient
from datetime import datetime

# Initialize client (no API key required!)
client = RegelleistungAPIClient()

# Download single day
date = datetime(2024, 1, 1)
df_afrr = client.download_and_parse('aFRR', 'CAPACITY', date)

print(f"Downloaded {len(df_afrr)} records")
print(df_afrr.head())

# Download date range
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 1, 31)
all_dfs = client.download_date_range('aFRR', 'CAPACITY', start_date, end_date)

print(f"Downloaded {len(all_dfs)} days")
```

## Daily Auto-Updates

Use the unified Germany updater for daily automated updates:

```bash
# Update all German market data (ENTSO-E + Regelleistung)
python iso_markets/entso_e/update_germany_with_resume.py

# Update only ancillary services
python iso_markets/entso_e/update_germany_with_resume.py --data-sources regelleistung

# Update only aFRR (focus on best opportunity)
python iso_markets/entso_e/update_germany_with_resume.py \
    --data-types afrr_capacity afrr_energy
```

### Cron Job Setup

```bash
# Edit crontab
crontab -e

# Add daily update at 10:00 AM (after market data publication)
0 10 * * * cd /home/enrico/projects/power_market_pipeline && \
           python3 iso_markets/entso_e/update_germany_with_resume.py \
           >> /var/log/germany_market_update.log 2>&1
```

## BESS Revenue Optimization Strategy

### 1. Capacity Revenue (Primary)
- **aFRR Capacity Auction**
  - Bid in daily auctions (d-7)
  - Strategic bidding per 4-hour block
  - Consider day-ahead price forecasts
  - Account for battery degradation costs

### 2. Energy Revenue (Secondary)
- **aFRR Energy Activations**
  - Revenue when called upon to provide energy
  - Activation prices vary by quarter-hour
  - Can be significant additional revenue

### 3. DA Energy Arbitrage (Complementary)
- **Day-Ahead Market**
  - Buy cheap at night, sell high during day
  - Can be combined with aFRR commitments
  - Requires optimization of schedules

### 4. Imbalance Provision (Opportunistic)
- **Real-Time Balancing**
  - Provide energy when system is short/long
  - 15-minute settlement
  - Can optimize around aFRR commitments

## Data Analysis Tips

### Price Volatility Analysis
```python
import pandas as pd

# Load aFRR capacity data
df = pd.read_csv('afrr_capacity_de_2024-01-01_2024-12-31.csv')

# Filter for Germany positive direction
df_pos = df[df['PRODUCT'].str.contains('POS')]

# Analyze marginal prices by time block
for product in df_pos['PRODUCT'].unique():
    prices = df_pos[df_pos['PRODUCT'] == product]['GERMANY_MARGINAL_CAPACITY_PRICE_[(EUR/MW)/h]']
    print(f"\n{product}:")
    print(f"  Mean: {prices.mean():.2f} EUR/MW/h")
    print(f"  Std:  {prices.std():.2f}")
    print(f"  Min:  {prices.min():.2f}")
    print(f"  Max:  {prices.max():.2f}")
```

### Block Profitability Comparison
```python
# Compare revenue potential by time block
block_revenue = df_pos.groupby('PRODUCT').agg({
    'GERMANY_MARGINAL_CAPACITY_PRICE_[(EUR/MW)/h]': ['mean', 'std', 'max'],
    'GERMANY_AVERAGE_CAPACITY_PRICE_[(EUR/MW)/h]': 'mean'
})

print("\nBlock Revenue Potential (EUR/MW/h):")
print(block_revenue.sort_values(('GERMANY_MARGINAL_CAPACITY_PRICE_[(EUR/MW)/h]', 'mean'), ascending=False))
```

## Market Insights

### Pricing Patterns
1. **Intraday Variation:** Morning (08-12) and evening (16-20) blocks typically have higher prices
2. **Seasonal Variation:** Winter months often have higher volatility
3. **Renewable Impact:** High wind/solar days can create price spikes
4. **Weekend Effect:** Lower demand = lower prices on weekends

### Competition
- FCR: Most competitive, many fast-response assets qualify
- aFRR: Moderate competition, sweet spot for batteries
- mFRR: Less competitive, but also lower prices

### Market Trends
- Growing demand due to renewable integration
- Increasing price volatility (= more opportunity!)
- European integration (MARI, PICASSO platforms)
- Shorter activation times becoming standard

## Files Created

After downloading, you'll find files like:

```
/pool/ssd8tb/data/iso/ENTSO_E/csv_files/de_ancillary_services/
├── fcr_capacity_de_2024-01-01_2024-12-31.csv
├── afrr_capacity_de_2024-01-01_2024-12-31.csv
├── afrr_energy_de_2024-01-01_2024-12-31.csv
├── mfrr_capacity_de_2024-01-01_2024-12-31.csv
└── mfrr_energy_de_2024-01-01_2024-12-31.csv
```

### File Sizes (Typical)
- FCR capacity: ~500 KB/year (12 records/day)
- aFRR capacity: ~500 KB/year (12 records/day)
- aFRR energy: ~10 MB/year (192 records/day)
- mFRR capacity: ~500 KB/year
- mFRR energy: ~10 MB/year

**Total: ~20-25 MB/year** (very manageable!)

## Troubleshooting

### No Data for Specific Dates
Some historical dates may not have data (market not operating, holidays, etc.). The downloader logs these and continues.

### Rate Limiting
The client uses a 2-second delay between requests (conservative). If you see errors, the site may be temporarily unavailable.

### File Format Issues
The API returns Excel files. Ensure `openpyxl` is installed:
```bash
pip install openpyxl
```

## References

- [Regelleistung.net Official Site](https://www.regelleistung.net/)
- [Regelleistung.net Datacenter](https://www.regelleistung.net/apps/datacenter/tenders/)
- [Modo Energy - Germany aFRR Analysis](https://modoenergy.com/research/germany-september-2025-afrr-explained)
- [TransnetBW - Ancillary Services Overview](https://www.transnetbw.de/en/energy-market/ancillary-services/)

---

**Next Steps:**
1. Download historical data for your target year
2. Analyze price patterns and volatility
3. Build bidding optimization model
4. Integrate with day-ahead arbitrage strategy
5. Set up daily auto-updates
6. Start optimizing BESS revenue! 💰

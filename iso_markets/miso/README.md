# MISO Data Download Scripts

Comprehensive collection of scripts to download all MISO (Midcontinent Independent System Operator) market data.

## Overview

MISO provides market data through two primary sources:
1. **Data Exchange API** - For recent pricing data (LMP, MCP) - requires API keys
2. **Market Reports Portal** - For historical data and additional market information - public CSV/XLS files

## Available Scripts

### 1. LMP Price Data (CSV Downloads)
**Script:** `download_historical_lmp.py`

Downloads historical LMP (Locational Marginal Price) data from public CSV files.

**Features:**
- Day-Ahead Ex-Post LMP (Actual prices, hourly)
- Day-Ahead Ex-Ante LMP (Forecasted prices, hourly)
- Real-Time LMP Final (Hourly aggregated)
- Hub-level and nodal-level data
- Available: 2024-present

**Usage:**
```bash
# Download hub-level data (faster, smaller files)
uv run python iso_markets/miso/download_historical_lmp.py \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --markets da_expost da_exante rt_final

# Download nodal-level data (all nodes, larger files)
uv run python iso_markets/miso/download_historical_lmp.py \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --markets da_expost \
  --all-nodes
```

**Output:**
- Location: `/pool/ssd8tb/data/iso/MISO/csv_files/`
- Format: CSV + filtered CSV (hubs only)
- Typical size: ~1MB per day for hub data, ~30MB per day for nodal data

---

### 2. Historical LMP via API (2019-2023)
**Script:** `download_historical_api.py`

Downloads historical LMP data using MISO Data Exchange API (requires API key).

**Features:**
- Day-Ahead Ex-Post LMP
- Real-Time Ex-Post LMP
- Hub-level and nodal-level data
- Available: 2019-2023 (pre-CSV era)

**Setup:**
Add to `.env` file:
```bash
MISO_PRICING_API_KEY=your_pricing_api_key_here
MISO_LOAD_AND_GEN_API_KEY=your_load_gen_api_key_here
```

**Usage:**
```bash
# Download hub-level historical data
uv run python iso_markets/miso/download_historical_api.py \
  --start-date 2019-01-01 \
  --end-date 2023-12-31 \
  --markets da_expost_lmp rt_expost_lmp

# Download nodal-level data (remove hub filter)
# Edit script and set filter_hubs=False or run with --all-nodes flag
```

**Rate Limits:**
- 100 calls per minute
- 24,000 calls per day
- Built-in rate limiting and retry logic

**Output:**
- Location: `/pool/ssd8tb/data/iso/MISO/api_data/pricing/`
- Format: JSON + CSV
- Typical size: ~270KB JSON + ~18KB CSV per day (hub data)

---

### 3. Real-Time 5-Minute LMP Data
**Script:** `download_rt_5min_lmp.py`

Downloads real-time LMP data at 5-minute resolution from weekly ZIP files.

**Features:**
- Real-Time 5-Minute LMP (all nodes, not aggregated)
- Weekly ZIP files containing ~7 days of data
- Each file: ~500MB compressed, ~2-3GB uncompressed
- ~4.9 million records per week (~7,000 nodes × 288 intervals/day × 7 days)
- Available: 2024-present
- Optional API download for last 4 days

**Data Format:**
- Columns: MKTHOUR_EST, PNODENAME, LMP, CON_LMP (congestion), LOSS_LMP (loss)
- 5-minute intervals: 12 per hour, 288 per day
- ~7,000+ nodes per timestamp

**Usage:**
```bash
# Download all nodes for a date range
uv run python iso_markets/miso/download_rt_5min_lmp.py \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --max-concurrent 3

# Download with recent API data (last 4 days)
uv run python iso_markets/miso/download_rt_5min_lmp.py \
  --start-date 2024-10-01 \
  --end-date 2024-10-10 \
  --download-recent-api

# Note: --filter-hubs flag exists but currently saves all data
# Hub filtering can be done in post-processing
```

**Weekly File Coverage:**
- Files are dated on Mondays
- Each file contains data from ~2 weeks before through ~1 week before the file date
- Example: `20241014_5MIN_LMP.zip` contains data from 09/30/2024-10/06/2024

**Output:**
- Location: `/pool/ssd8tb/data/iso/MISO/csv_files/rt_5min_lmp/`
- Format: CSV (extracted from ZIP archives)
- File size: ~500MB compressed ZIP, ~2-3GB uncompressed CSV
- Recent API data: `/pool/ssd8tb/data/iso/MISO/csv_files/rt_5min_lmp/recent/`

**Important Notes:**
- Large files require significant disk space (~20GB per year uncompressed)
- Use `--max-concurrent 2-3` to avoid overwhelming disk I/O
- Files automatically skip 4 header rows and footer disclaimer during parsing

---

### 4. Ancillary Services Data
**Script:** `download_ancillary_services.py`

Downloads ancillary services pricing data (reserves, regulation, ramp capability).

**Features:**
- Day-Ahead ExAnte Ramp MCPs (Hourly)
- Day-Ahead ExPost Ramp MCPs (Hourly)
- Real-Time ExPost Ramp MCPs (Hourly aggregated)
- Real-Time ExPost Ramp 5-Min MCPs (5-minute resolution)

**Usage:**
```bash
# Download all ancillary services
uv run python iso_markets/miso/download_ancillary_services.py \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --report-types da_exante_ramp da_expost_ramp rt_expost_ramp_hourly rt_expost_ramp_5min

# Download only day-ahead data
uv run python iso_markets/miso/download_ancillary_services.py \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --report-types da_expost_ramp
```

**Available Report Types:**
- `da_exante_ramp` - Day-Ahead ExAnte Ramp MCP
- `da_expost_ramp` - Day-Ahead ExPost Ramp MCP
- `rt_expost_ramp_hourly` - Real-Time ExPost Ramp MCP (Hourly)
- `rt_expost_ramp_5min` - Real-Time ExPost Ramp MCP (5-minute)

**Output:**
- Location: `/pool/ssd8tb/data/iso/MISO/csv_files/ancillary_services/`
- Format: XLS (Excel)

**Note:** Ancillary services in MISO include:
- Regulating Reserve
- Spinning Reserve
- Supplemental Reserve
- Ramp Capability
- 30-minute Short-Term Reserve

---

### 5. Load Data (Actual & Forecast)
**Script:** `download_load_data.py`

Downloads system load data (actual and forecast).

**Features:**
- Daily Forecast and Actual Load by Local Resource Zone (LRZ)
- Daily Regional Forecast and Actual Load
- Hourly resolution
- Prior day actual + current day forecast

**Usage:**
```bash
# Download all load data
uv run python iso_markets/miso/download_load_data.py \
  --start-date 2024-01-01 \
  --end-date 2024-12-31

# Download only regional data
uv run python iso_markets/miso/download_load_data.py \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --report-types regional
```

**Available Report Types:**
- `local_resource_zone` - Load by LRZ (more granular)
- `regional` - Load by MISO region (less granular)

**Output:**
- Location: `/pool/ssd8tb/data/iso/MISO/csv_files/load/`
- Format: XLS (Excel)

---

### 6. Generation and Fuel Mix Data (MISO Direct)
**Script:** `download_generation_fuel_mix.py`

Downloads generation and fuel mix data showing production by fuel type from MISO Market Reports.

**⚠️ Current Status:** Files return 404 errors - MISO may have discontinued these reports or moved them to Data Exchange API.

**Features:**
- Real-Time 5-Minute Generation Fuel Mix
- Annual Historical Generation Fuel Mix
- ACE (Area Control Error) Data
- Breakdown by fuel type: Coal, Gas, Nuclear, Wind, Solar, Other

**Usage:**
```bash
# Download 5-minute fuel mix data
uv run python iso_markets/miso/download_generation_fuel_mix.py \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --report-types fuel_mix_5min

# Download with annual historical files
uv run python iso_markets/miso/download_generation_fuel_mix.py \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --report-types fuel_mix_5min \
  --download-annual
```

**Available Report Types:**
- `fuel_mix_5min` - Real-Time 5-Minute Generation Fuel Mix
- `ace_data` - ACE (Area Control Error) Data

**Output:**
- Location: `/pool/ssd8tb/data/iso/MISO/csv_files/generation/`
- Format: XLS (Excel) or CSV
- Annual files: `/pool/ssd8tb/data/iso/MISO/csv_files/generation/fuel_mix_annual/`

---

### 7. EIA-930 Fuel Mix Data (Alternative Source) ⭐ **RECOMMENDED** ⭐
**Script:** `download_eia_fuel_mix.py`

Downloads hourly generation fuel mix data for MISO from the **U.S. Energy Information Administration (EIA) Form 930 API**.

This is the **recommended alternative** to MISO direct fuel mix reports, which are currently unavailable (404 errors).

**Features:**
- **Hourly generation by fuel type**: Coal, Natural Gas, Nuclear, Wind, Solar, Hydro, Other
- **Historical coverage**: July 2015 - present (fuel mix by source added July 2018)
- **Official U.S. government data** - highly reliable
- **Free API** - no rate limits or monthly caps
- **Multiple output formats**: JSON, CSV, and pivoted CSV
- **Standardized across all ISOs** - same data structure for CAISO, PJM, ERCOT, etc.

**Setup:**
1. Get free EIA API key: https://www.eia.gov/opendata/register.php
2. Add to `.env` file:
```bash
EIA_API_KEY=your_api_key_here
```

**Usage:**
```bash
# Download 2024 fuel mix data
uv run python iso_markets/miso/download_eia_fuel_mix.py \
  --start-date 2024-01-01 \
  --end-date 2024-12-31

# Download specific date range
uv run python iso_markets/miso/download_eia_fuel_mix.py \
  --start-date 2024-10-01 \
  --end-date 2024-10-10

# Filter by specific fuel types (optional)
uv run python iso_markets/miso/download_eia_fuel_mix.py \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --fuel-types COL NG WND SUN
```

**Available Fuel Types:**
- `COL` - Coal
- `NG` - Natural Gas
- `NUC` - Nuclear
- `SUN` - Solar
- `WND` - Wind
- `WAT` - Hydro/Water
- `OTH` - Other

**Output:**
- Location: `/pool/ssd8tb/data/iso/MISO/eia_fuel_mix/`
- Formats:
  - **JSON**: Raw API response (`eia_fuel_mix_YYYYMMDD_YYYYMMDD.json`)
  - **CSV**: Long format with timestamp, fuel type, value (`eia_fuel_mix_YYYYMMDD_YYYYMMDD.csv`)
  - **Pivot CSV**: Wide format - rows=timestamp, columns=fuel types (`eia_fuel_mix_YYYYMMDD_YYYYMMDD_pivot.csv`)

**Data Format Example:**
```csv
period,respondent,fueltype,value,value-units
2024-01-01T00,MISO,COL,5234.0,megawatthours
2024-01-01T00,MISO,NG,8765.0,megawatthours
2024-01-01T00,MISO,NUC,4321.0,megawatthours
2024-01-01T00,MISO,WND,2100.0,megawatthours
...
```

**Advantages:**
- ✅ Official U.S. government source
- ✅ Free and unlimited API access
- ✅ Historical data back to July 2015
- ✅ Hourly granularity (same as MISO hourly data)
- ✅ Automatic pagination for large date ranges
- ✅ Multiple output formats for different use cases
- ✅ Works across all U.S. ISOs with same code

**Important Notes:**
- EIA data has ~2-day lag for finalized data
- Data is based on actual metered generation reported by balancing authorities
- More reliable than MISO direct reports for historical analysis
- Can be combined with MISO LMP data for complete market analysis

---

## Data Coverage Summary

| Data Type | Time Period | Resolution | Format | Script |
|-----------|-------------|------------|--------|--------|
| **Energy Prices** |
| DA LMP (CSV) | 2024-present | Hourly | CSV | download_historical_lmp.py |
| RT LMP Hourly (CSV) | 2024-present | Hourly | CSV | download_historical_lmp.py |
| **RT LMP 5-Min** | **2024-present** | **5-minute** | **CSV** | **download_rt_5min_lmp.py** |
| DA LMP (API) | 2019-2023 | Hourly | JSON/CSV | download_historical_api.py |
| RT LMP (API) | 2019-2023 | Hourly | JSON/CSV | download_historical_api.py |
| **Ancillary Services** |
| DA Ramp MCP | 2024-present | Hourly | XLS | download_ancillary_services.py |
| RT Ramp MCP | 2024-present | 5-min/Hourly | XLS | download_ancillary_services.py |
| **Load** |
| Actual Load | 2024-present | Hourly | XLS | download_load_data.py |
| Load Forecast | 2024-present | Hourly | XLS | download_load_data.py |
| **Generation** |
| Fuel Mix | 2024-present | 5-minute | XLS | download_generation_fuel_mix.py |
| Annual Fuel Mix | Historical | Annual | XLS | download_generation_fuel_mix.py |

---

## Complete Download Examples

### Download All Data for 2024
```bash
# 1. LMP Prices (CSV)
uv run python iso_markets/miso/download_historical_lmp.py \
  --start-date 2024-01-01 --end-date 2024-12-31 \
  --markets da_expost da_exante rt_final

# 2. Ancillary Services
uv run python iso_markets/miso/download_ancillary_services.py \
  --start-date 2024-01-01 --end-date 2024-12-31

# 3. Load Data
uv run python iso_markets/miso/download_load_data.py \
  --start-date 2024-01-01 --end-date 2024-12-31

# 4. Generation/Fuel Mix
uv run python iso_markets/miso/download_generation_fuel_mix.py \
  --start-date 2024-01-01 --end-date 2024-12-31 \
  --download-annual

# 5. Real-Time 5-Minute LMP (High-resolution data)
uv run python iso_markets/miso/download_rt_5min_lmp.py \
  --start-date 2024-01-01 --end-date 2024-12-31 \
  --max-concurrent 3
```

### Download Historical Data (2019-2023)
```bash
# Requires API keys in .env file
uv run python iso_markets/miso/download_historical_api.py \
  --start-date 2019-01-01 --end-date 2023-12-31 \
  --markets da_expost_lmp rt_expost_lmp
```

---

## API Keys Setup

### MISO Data Exchange (for historical LMP data 2019-2023)

1. **Register for MISO Data Exchange**
   - Go to: https://data-exchange.misoenergy.org/
   - Create account with MISO public website credentials

2. **Subscribe to APIs**
   - Pricing API: For LMP, MCP data
   - Load & Generation API: For load, fuel mix data

3. **Get API Keys**
   - Navigate to Profile → Subscriptions
   - Copy "Primary key" for each subscription

4. **Add to .env file**
```bash
MISO_PRICING_API_KEY=your_primary_key_here
MISO_LOAD_AND_GEN_API_KEY=your_primary_key_here
MISO_DATA_DIR=/pool/ssd8tb/data/iso/MISO
```

### EIA Open Data API (for fuel mix data)

1. **Register for free EIA API key**
   - Go to: https://www.eia.gov/opendata/register.php
   - Fill out registration form
   - API key will be emailed to you instantly

2. **Add to .env file**
```bash
EIA_API_KEY=your_eia_api_key_here
```

**Benefits:**
- Free and unlimited API access (no rate limits)
- Access to all EIA electricity data (not just MISO)
- Can be used for all ISOs: MISO, CAISO, PJM, ERCOT, ISO-NE, NYISO, SPP
- Historical data back to July 2015

---

## Hub vs Nodal Data
- **Hub data**: ~120-130 major trading hubs (faster downloads, smaller files)
- **Nodal data**: ~7,000+ individual nodes (slower downloads, larger files)
- Default: Hub data only
- To get nodal data: Use `--all-nodes` flag or edit scripts to set `filter_hubs=False`

---

## Related Documentation

- **MISO Data Exchange**: https://data-exchange.misoenergy.org/
- **Market Reports**: https://www.misoenergy.org/markets-and-operations/real-time--market-data/market-reports/
- **ISO Data Requirements**: `/home/enrico/projects/power_market_pipeline/ISO_DATA_REQUIREMENTS.md`

---

**Last Updated:** 2025-10-11

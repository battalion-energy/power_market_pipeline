# EIA-930 Fuel Mix Data Setup Guide

Quick start guide for downloading MISO fuel mix data using the EIA-930 API.

## Why Use EIA Instead of MISO Direct?

**MISO's direct fuel mix reports are currently unavailable (404 errors).** The EIA-930 API is the **recommended alternative** because:

✅ **Official U.S. Government Source** - Highly reliable data from Energy Information Administration
✅ **Free & Unlimited** - No API rate limits or monthly caps
✅ **Historical Coverage** - Data back to July 2015
✅ **Standardized Format** - Works across all ISOs (MISO, CAISO, PJM, ERCOT, etc.)
✅ **Hourly Granularity** - Matches your other MISO hourly data

## Setup (One-Time)

### Step 1: Get Free EIA API Key

1. Go to: **https://www.eia.gov/opendata/register.php**
2. Fill out the simple registration form:
   - First Name, Last Name
   - Organization (optional - can say "Personal Research")
   - Email address
3. **Check your email** - API key arrives instantly
4. Copy the API key (format: `40-character alphanumeric string`)

### Step 2: Add API Key to .env File

Open your project's `.env` file and add:

```bash
# EIA Open Data API (for fuel mix and electricity data)
EIA_API_KEY=your_40_character_api_key_here
```

**Example:**
```bash
EIA_API_KEY=a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0
```

That's it! You're ready to download data.

---

## Usage Examples

### Download Full Year 2024

```bash
uv run python iso_markets/miso/download_eia_fuel_mix.py \
  --start-date 2024-01-01 \
  --end-date 2024-12-31
```

**Output:**
- `eia_fuel_mix_20240101_20241231.json` - Raw API response
- `eia_fuel_mix_20240101_20241231.csv` - Long format (timestamp, fuel type, value)
- `eia_fuel_mix_20240101_20241231_pivot.csv` - Wide format (timestamp as rows, fuel types as columns)

### Download Specific Month

```bash
uv run python iso_markets/miso/download_eia_fuel_mix.py \
  --start-date 2024-10-01 \
  --end-date 2024-10-31
```

### Download Historical Data

```bash
# Get 2023 data
uv run python iso_markets/miso/download_eia_fuel_mix.py \
  --start-date 2023-01-01 \
  --end-date 2023-12-31

# Get 2022 data
uv run python iso_markets/miso/download_eia_fuel_mix.py \
  --start-date 2022-01-01 \
  --end-date 2022-12-31

# Can go back to July 2018 (when fuel mix by source was added)
uv run python iso_markets/miso/download_eia_fuel_mix.py \
  --start-date 2018-07-01 \
  --end-date 2018-12-31
```

### Filter by Specific Fuel Types (Optional)

```bash
# Only wind and solar
uv run python iso_markets/miso/download_eia_fuel_mix.py \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --fuel-types WND SUN

# Only coal and natural gas
uv run python iso_markets/miso/download_eia_fuel_mix.py \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --fuel-types COL NG
```

---

## Data Format Details

### Available Fuel Types

| Code | Fuel Type | Description |
|------|-----------|-------------|
| `COL` | Coal | Coal-fired generation |
| `NG` | Natural Gas | Natural gas-fired generation |
| `NUC` | Nuclear | Nuclear generation |
| `SUN` | Solar | Solar PV and thermal |
| `WND` | Wind | Wind generation |
| `WAT` | Hydro | Hydroelectric generation |
| `OTH` | Other | Biomass, geothermal, other |

### CSV Format (Long Format)

```csv
period,respondent,fueltype,value,value-units
2024-01-01T00,MISO,COL,5234.0,megawatthours
2024-01-01T00,MISO,NG,8765.0,megawatthours
2024-01-01T00,MISO,NUC,4321.0,megawatthours
2024-01-01T00,MISO,WND,2100.0,megawatthours
2024-01-01T00,MISO,SUN,450.0,megawatthours
2024-01-01T00,MISO,WAT,890.0,megawatthours
2024-01-01T00,MISO,OTH,120.0,megawatthours
2024-01-01T01,MISO,COL,5100.0,megawatthours
...
```

### Pivot CSV Format (Wide Format)

**Better for analysis** - Each row is a timestamp, columns are fuel types:

```csv
period,COL,NG,NUC,WND,SUN,WAT,OTH
2024-01-01T00,5234.0,8765.0,4321.0,2100.0,450.0,890.0,120.0
2024-01-01T01,5100.0,8950.0,4321.0,2200.0,0.0,890.0,110.0
2024-01-01T02,4980.0,9100.0,4321.0,2300.0,0.0,890.0,115.0
...
```

---

## Combining with MISO LMP Data

The EIA fuel mix data can be combined with MISO LMP data for comprehensive market analysis:

```python
import pandas as pd

# Load LMP data (hourly)
lmp_df = pd.read_csv('/pool/ssd8tb/data/iso/MISO/csv_files/20240101_da_expost.csv')

# Load fuel mix data (pivot format is easier)
fuel_mix_df = pd.read_csv('/pool/ssd8tb/data/iso/MISO/eia_fuel_mix/eia_fuel_mix_20240101_20241231_pivot.csv')

# Convert timestamps to datetime
lmp_df['datetime'] = pd.to_datetime(lmp_df['HourEnding'])
fuel_mix_df['datetime'] = pd.to_datetime(fuel_mix_df['period'])

# Merge on datetime
combined_df = pd.merge(lmp_df, fuel_mix_df, on='datetime', how='inner')

# Now you have prices and fuel mix for each hour
print(combined_df[['datetime', 'LMP', 'WND', 'SUN', 'NG', 'COL']].head())
```

---

## Downloading Multiple Years

For bulk historical downloads:

```bash
#!/bin/bash

# Download 2018-2024 (all available years with fuel mix data)
for year in {2018..2024}; do
    echo "Downloading $year..."
    uv run python iso_markets/miso/download_eia_fuel_mix.py \
        --start-date ${year}-01-01 \
        --end-date ${year}-12-31
    sleep 2  # Small delay between years
done

echo "All years downloaded!"
```

---

## Troubleshooting

### Error: "EIA API key required"

**Problem:** API key not found in `.env` file.

**Solution:**
1. Make sure `.env` file is in the project root: `/home/enrico/projects/power_market_pipeline/.env`
2. Check the line starts with `EIA_API_KEY=` (no spaces before the variable name)
3. Verify the API key is 40 characters long
4. Try passing API key directly: `--api-key YOUR_KEY`

### Error: "403 Forbidden" or "Invalid API Key"

**Problem:** API key is incorrect or expired.

**Solution:**
1. Double-check you copied the full API key (40 characters)
2. Make sure there are no extra spaces in the `.env` file
3. Request a new API key if yours expired: https://www.eia.gov/opendata/register.php

### Large Date Range Takes Long Time

**This is normal!** The EIA API has pagination (5,000 rows per request).

**Example timing:**
- 1 month (~720 hours × 7 fuel types = ~5,000 rows): ~1 request, < 5 seconds
- 1 year (~8,760 hours × 7 fuel types = ~61,000 rows): ~13 requests, ~30 seconds
- Multiple years: Download year-by-year to see progress

### No Data Returned

**Check these:**
1. Date range is after July 1, 2018 (when fuel mix by source was added)
2. Dates are in correct format: `YYYY-MM-DD`
3. Start date is before end date
4. Internet connection is working

---

## Additional Resources

- **EIA Open Data Portal:** https://www.eia.gov/opendata/
- **EIA Grid Monitor (Interactive):** https://www.eia.gov/electricity/gridmonitor/dashboard/electric_overview/balancing_authority/MISO
- **EIA Wholesale Markets Data:** https://www.eia.gov/electricity/wholesalemarkets/data.php?rto=miso
- **API Documentation:** https://www.eia.gov/opendata/documentation.php

---

## Summary

✅ **Fast Setup** - Get API key in 2 minutes, add to `.env`, start downloading
✅ **Reliable Data** - Official U.S. government source
✅ **Free Forever** - No rate limits, no monthly caps
✅ **Historical Coverage** - July 2018 to present
✅ **Multiple Formats** - JSON, CSV (long), CSV (pivot)
✅ **Easy Analysis** - Combine with MISO LMP data

For questions or issues, check:
- `iso_markets/miso/README.md` - Complete MISO documentation
- `iso_markets/miso/DOWNLOAD_FIXES_SUMMARY.md` - Recent fixes and updates

**Last Updated:** 2025-10-11

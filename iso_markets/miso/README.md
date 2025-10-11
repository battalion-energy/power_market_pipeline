# MISO Market Data Downloader

Midcontinent Independent System Operator (MISO) data downloader.

## Data Sources

### LMP Data (Locational Marginal Pricing)

**Base URL**: `https://docs.misoenergy.org/marketreports/`

**Available Markets:**
1. **Day-Ahead Ex-Post**: `{YYYYMMDD}_da_expost_lmp.csv` - Actual day-ahead prices
2. **Day-Ahead Ex-Ante**: `{YYYYMMDD}_da_exante_lmp.csv` - Forecasted day-ahead prices
3. **Real-Time Final**: `{YYYYMMDD}_rt_lmp_final.csv` - Final real-time prices (hourly)

**Data Availability:**
- Recent data (2024-present): ✅ Available
- Historical data (2019-2023): ❓ Requires investigation (possibly via MISO Data Exchange API)

### Data Structure

CSV format with the following structure:
- **Header rows**: Market name and date
- **Columns**:
  - `Node`: Settlement point identifier
  - `Type`: One of Hub, Loadzone, Interface, Gennode
  - `Value`: Price component (LMP, MCC, MLC)
  - `HE 1` through `HE 24`: Hour Ending prices in EST

**Node Types:**
- **Hub**: Commercial/trading hubs (recommended starting point)
- **Loadzone**: Load aggregation zones
- **Interface**: Import/export interfaces
- **Gennode**: Individual generation nodes

**Price Components:**
- **LMP**: Locational Marginal Price (total)
- **MCC**: Marginal Congestion Component
- **MLC**: Marginal Loss Component

## Usage

### Download Hub-Level Data (Recommended)

```bash
# Download 2024-present hub-level data
python iso_markets/miso/download_historical_lmp.py \
    --start-date 2024-01-01 \
    --end-date 2025-10-10 \
    --markets da_expost da_exante rt_final

# Download specific date range
python iso_markets/miso/download_historical_lmp.py \
    --start-date 2024-06-01 \
    --end-date 2024-06-30 \
    --markets da_expost rt_final
```

### Download All Nodes

```bash
# Download all nodes (not just hubs)
python iso_markets/miso/download_historical_lmp.py \
    --start-date 2024-01-01 \
    --end-date 2025-10-10 \
    --all-nodes
```

### Command-Line Options

```
--start-date YYYY-MM-DD    Start date (required)
--end-date YYYY-MM-DD      End date (default: today)
--markets [TYPES...]       Market types: da_expost, da_exante, rt_final
--all-nodes                Download all nodes (default: hubs only)
--output-dir PATH          Output directory (default: $MISO_DATA_DIR)
--max-concurrent N         Max concurrent downloads (default: 5)
```

## Environment Variables

Create a `.env` file in the project root:

```bash
MISO_DATA_DIR=/pool/ssd8tb/data/iso/MISO/csv_files
```

## Output Structure

```
$MISO_DATA_DIR/
├── da_expost/
│   ├── 20240101_da_expost_lmp.csv
│   ├── 20240101_da_expost_lmp_hubs_only.csv
│   └── ...
├── da_exante/
│   ├── 20240101_da_exante_lmp.csv
│   ├── 20240101_da_exante_lmp_hubs_only.csv
│   └── ...
└── rt_final/
    ├── 20240101_rt_lmp_final.csv
    ├── 20240101_rt_lmp_final_hubs_only.csv
    └── ...
```

## MISO Hub Examples

Major MISO commercial hubs include:
- **INDIANA.HUB**
- **ILLINOIS.HUB**
- **MICHIGAN.HUB**
- **MINNESOTA.HUB**
- **ARKANSAS.HUB**
- **LOUISIANA.HUB**
- **MISSISSIPPI.HUB**
- **TEXAS.HUB**

## Historical Data (2019-2023)

Historical data before 2024 returns 404 errors from the public URL. To access this data:

1. **MISO Data Exchange API**: Requires account at https://data-exchange.misoenergy.org/
2. **U.S. EIA**: Alternative source at https://www.eia.gov/electricity/wholesalemarkets/data.php?rto=miso
3. **Market Report Archives**: Check https://www.misoenergy.org/markets-and-operations/real-time--market-data/market-report-archives/

## Ancillary Services Data

Ancillary services pricing data is mentioned as available via MISO's "consolidated API" but requires further investigation:
- Spinning Reserves
- Supplemental Reserves
- Regulation services

**API Endpoint** (unverified): `https://api.misoenergy.org/MISORTWDDataBroker/DataBrokerServices.asmx?messageType=ancillarymcp&returnType=csv`

## Next Steps

1. ✅ Download 2024-present hub-level data
2. ⏳ Investigate 2019-2023 data access (MISO Data Exchange API)
3. ⏳ Add ancillary services downloader
4. ⏳ Implement nodal data processing pipeline
5. ⏳ Create Parquet conversion pipeline (similar to ERCOT/PJM)

## Notes

- All timestamps are in Eastern Standard Time (EST)
- Data is published on a daily basis
- Hub-level data is sufficient for most market analysis
- Full nodal data (~2,200 nodes) available if needed

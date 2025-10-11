# IESO (Ontario) Downloader

Complete implementation of IESO (Independent Electricity System Operator) data downloader for Ontario's electricity market.

## Overview

The IESODownloaderV2 class downloads public market data from IESO's REST API endpoints. It handles the major market transition from HOEP (Hourly Ontario Energy Price) to LMP (Locational Marginal Pricing) that occurred on May 1, 2025.

## Market Transition (May 1, 2025)

### Pre-May 2025
- **HOEP**: Single province-wide hourly energy price
- **MCP**: Market Clearing Price (zonal prices)
- Simple uniform pricing system

### Post-May 2025
- **LMP**: Locational Marginal Pricing (~1000 nodes)
- **OEMP**: Ontario Energy Market Price (replaces HOEP as reference price)
- **Zonal Prices**: 10 pricing zones
- Co-optimized energy and operating reserves

## Data Sources

### Energy Markets

#### Day-Ahead LMP (Post-May 2025)
- **Report Code**: `PUB_DALMPEnergy`
- **URL Pattern**: `https://reports-public.ieso.ca/public/PUB_DALMPEnergy_YYYYMMDD.csv`
- **Granularity**: Hourly
- **Locations**: ~1000 nodes

#### Real-Time LMP (Post-May 2025)
- **Report Code**: `PUB_RTLMPEnergy`
- **URL Pattern**: `https://reports-public.ieso.ca/public/PUB_RTLMPEnergy_YYYYMMDD.csv`
- **Granularity**: 5-minute
- **Locations**: ~1000 nodes

#### Ontario Zonal Prices (Post-May 2025)
- **Report Code**: `PUB_OntarioZonalPrice`
- **URL Pattern**: `https://reports-public.ieso.ca/public/PUB_OntarioZonalPrice_YYYYMMDD.csv`
- **Granularity**: Hourly
- **Locations**: 10 zones

#### OEMP (Post-May 2025)
- **Report Code**: `PUB_OEMP` (TBD - verify with IESO)
- **Replacement**: For legacy HOEP
- **Granularity**: Hourly

#### Legacy HOEP (Pre-May 2025)
- **Report Code**: `PUB_HOEP` (TBD - verify with IESO)
- **Retired**: April 30, 2025
- **Granularity**: Hourly
- **Location**: Province-wide single price

### Ancillary Services

IESO operates three operating reserve markets:

#### 10-Minute Synchronized Reserve (10S)
- **Report Code**: `PUB_OR_10S` (TBD - verify with IESO)
- **Description**: Fast-response synchronized reserves

#### 10-Minute Non-Synchronized Reserve (10NS)
- **Report Code**: `PUB_OR_10NS` (TBD - verify with IESO)
- **Description**: Fast-response non-synchronized reserves

#### 30-Minute Operating Reserve (30OR)
- **Report Code**: `PUB_OR_30OR` (TBD - verify with IESO)
- **Description**: Slower-response operating reserves

**Note**: Other ancillary services (Regulation, Black Start, Reactive Support, Reliability Must-Run) are contracted services and not publicly available via market pricing.

### Load Data
- **Actual Load**: `PUB_Load_Actual` (TBD)
- **Forecast Load**: `PUB_Load_Forecast` (TBD)

## Usage

### Basic Example

```python
import asyncio
from datetime import datetime
from downloaders.base_v2 import DownloadConfig
from downloaders.ieso import IESODownloaderV2

async def download_ieso_data():
    config = DownloadConfig(
        start_date=datetime(2025, 5, 1),
        end_date=datetime(2025, 5, 7),
        data_types=["lmp", "ancillary_services"],
        output_dir="/data/iso"
    )

    downloader = IESODownloaderV2(config)

    # Download day-ahead LMP
    await downloader.download_lmp("DAM", config.start_date, config.end_date)

    # Download real-time LMP
    await downloader.download_lmp("RT5M", config.start_date, config.end_date)

    # Download operating reserves
    await downloader.download_ancillary_services("ALL", "RTM", config.start_date, config.end_date)

asyncio.run(download_ieso_data())
```

### Download All Markets

```python
async def download_all():
    config = DownloadConfig(
        start_date=datetime(2025, 5, 1),
        end_date=datetime(2025, 5, 31),
        data_types=["all"],
        output_dir="/data/iso"
    )

    downloader = IESODownloaderV2(config)

    # Downloads everything: LMP, OEMP, Zonal, AS, Load
    results = await downloader.download_all_markets(
        config.start_date,
        config.end_date,
        include_legacy=False
    )

    print(f"Downloaded: {results}")

asyncio.run(download_all())
```

### Handle Transition Period

```python
async def download_transition():
    # Download data spanning HOEP -> LMP transition
    config = DownloadConfig(
        start_date=datetime(2025, 4, 1),   # Before transition
        end_date=datetime(2025, 5, 31),     # After transition
        data_types=["all"],
        output_dir="/data/iso"
    )

    downloader = IESODownloaderV2(config)

    # Automatically handles both HOEP and LMP periods
    results = await downloader.download_all_markets(
        config.start_date,
        config.end_date,
        include_legacy=True  # Include HOEP data
    )

asyncio.run(download_transition())
```

### Legacy HOEP Only

```python
async def download_historical_hoep():
    config = DownloadConfig(
        start_date=datetime(2019, 1, 1),
        end_date=datetime(2025, 4, 30),
        data_types=["hoep"],
        output_dir="/data/iso"
    )

    downloader = IESODownloaderV2(config)

    # Download only legacy HOEP data
    count = await downloader.download_legacy_hoep(config.start_date, config.end_date)
    print(f"Downloaded {count} HOEP files")

asyncio.run(download_historical_hoep())
```

## Output Structure

Downloaded files are organized as follows:

```
{output_dir}/IESO_data/csv_files/
├── da_lmp/
│   ├── PUB_DALMPEnergy_20250501.csv
│   ├── PUB_DALMPEnergy_20250502.csv
│   └── ...
├── rt_lmp/
│   ├── PUB_RTLMPEnergy_20250501.csv
│   └── ...
├── zonal_prices/
│   ├── PUB_OntarioZonalPrice_20250501.csv
│   └── ...
├── oemp/
│   ├── PUB_OEMP_20250501.csv
│   └── ...
├── hoep_legacy/
│   ├── PUB_HOEP_20241231.csv
│   └── ...
├── ancillary_services/
│   ├── 10s/
│   │   ├── PUB_OR_10S_20250501.csv
│   │   └── ...
│   ├── 10ns/
│   │   └── ...
│   └── 30or/
│       └── ...
└── load/
    ├── actual/
    │   └── ...
    └── forecast/
        └── ...
```

## Implementation Notes

### Report Codes
Some report codes in this implementation are marked "TBD" because they need to be verified against actual IESO documentation. The codes follow IESO's likely naming conventions but should be confirmed before production use.

**To verify report codes:**
1. Check IESO's public data catalog: https://www.ieso.ca/en/Power-Data
2. Inspect actual file names in their public repository
3. Contact IESO if documentation is unclear

### Date Handling
- All dates are in Eastern Time (America/Toronto)
- LMP transition date is hardcoded as May 1, 2025
- Methods automatically handle transition logic

### Node Discovery
The ~1000 LMP nodes are not hardcoded. To get the actual node list:
1. Download LMP CSV files
2. Extract unique node IDs from the data
3. Cache for future reference

### Error Handling
- 404 errors are logged as warnings (expected for future dates or missing data)
- Other HTTP errors trigger retries (configurable)
- Failed downloads are logged but don't stop the process

### Performance
- Downloads are asynchronous for speed
- No rate limiting (public API)
- Files are checked for existence before download (resumable)

## API Reference

### IESODownloaderV2

#### Methods

##### `download_lmp(market, start_date, end_date, locations=None)`
Download LMP data (post-May 2025 only).

**Parameters:**
- `market`: 'DAM' or 'RT5M'
- `start_date`: datetime
- `end_date`: datetime
- `locations`: Not used (all nodes in single files)

**Returns:** Number of files downloaded

##### `download_ontario_zonal_prices(start_date, end_date)`
Download Ontario zonal prices (10 zones, post-May 2025).

##### `download_oemp(start_date, end_date)`
Download OEMP (Ontario Energy Market Price, post-May 2025).

##### `download_legacy_hoep(start_date, end_date)`
Download legacy HOEP (Hourly Ontario Energy Price, pre-May 2025).

##### `download_ancillary_services(product, market, start_date, end_date)`
Download operating reserve prices.

**Parameters:**
- `product`: '10S', '10NS', '30OR', or 'ALL'
- `market`: 'DAM' or 'RTM'

##### `download_load(forecast_type, start_date, end_date)`
Download load data.

**Parameters:**
- `forecast_type`: 'actual' or 'forecast'

##### `download_all_markets(start_date, end_date, include_legacy=True)`
Convenience method to download all IESO markets at once.

**Returns:** Dictionary with counts for each dataset

##### `get_available_locations()`
Get list of available IESO locations (zones).

**Returns:** List of location dictionaries

## Testing

Run the test suite:

```bash
# From project root
python test_ieso_downloader.py
```

The test suite includes:
1. Day-Ahead LMP download
2. Real-Time LMP download
3. Ontario Zonal Prices
4. OEMP download
5. Operating Reserves (10S, 10NS, 30OR)
6. Load data
7. Convenience method test
8. Legacy HOEP test
9. Transition period test
10. Location listing

## Next Steps

1. **Verify Report Codes**: Confirm actual IESO report codes with their documentation
2. **Test with Real Data**: Run against IESO API to verify file formats
3. **Extract Node List**: Download LMP files and extract complete node inventory
4. **Create Processor**: Build IESO-specific processor to convert CSV to Parquet
5. **Schema Mapping**: Map IESO columns to standardized schema
6. **Database Integration**: Load processed data into TimescaleDB

## References

- [IESO Public Data](https://www.ieso.ca/en/Power-Data)
- [IESO Market Renewal](https://www.ieso.ca/en/Market-Renewal)
- [LMP Transition Documentation](https://www.ieso.ca/en/Market-Renewal/Market-Renewal-Program)
- [Operating Reserve Markets](https://www.ieso.ca/en/Learn/Operating-Reserve)

## Support

For issues or questions:
1. Check IESO public data documentation
2. Verify report codes match current IESO naming
3. Check for API changes after May 2025 transition
4. Contact IESO customer relations for API access issues

---

*Last Updated: 2025-10-10*
*Target Date Range: 2019-01-01 to 2025-10-10*
*LMP Transition Date: 2025-05-01*

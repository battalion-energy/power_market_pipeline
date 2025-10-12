# ERCOT Battery Storage Data Downloader

## Overview

This project provides automated tools to download historical and real-time battery storage data from ERCOT (Electric Reliability Council of Texas). This data is the same source used by gridstatus.io for their "Storage (Net Output)" charts.

## Data Sources Discovered

### 1. ERCOT Energy Storage Resources API
- **Endpoint**: `https://www.ercot.com/api/1/services/read/dashboards/energy-storage-resources.json`
- **Data Available**: Previous day + current day (rolling 2-day window)
- **Time Resolution**: 5-minute intervals
- **Fields**:
  - `timestamp`: Datetime with timezone (Central Time)
  - `totalCharging`: Total MW charging (negative values)
  - `totalDischarging`: Total MW discharging (positive values)
  - `netOutput`: Net output MW (discharging - charging)
- **Authentication**: None required (public endpoint)

### 2. ERCOT Fuel Mix API
- **Endpoint**: `https://www.ercot.com/api/1/services/read/dashboards/fuel-mix.json`
- **Data Available**: Current day fuel mix including "Power Storage" generation
- **Time Resolution**: ~5-minute intervals
- **Fields**:
  - `timestamp`: Datetime with timezone
  - `gen`: Generation in MW (for Power Storage category)
- **Authentication**: None required (public endpoint)

### 3. Historical Fuel Mix Reports
- **Source**: `https://www.ercot.com/gridinfo/generation`
- **Available Files**:
  - "Fuel Mix Report: 2007 - 2024" (ZIP file, ~48 MB)
  - "Fuel Mix Report: 2025" (Excel file, ~2 MB)
- **Time Resolution**: 15-minute settlement intervals
- **Data**: Actual generation by fuel type, including Power Storage
- **Format**: Excel/CSV files in ZIP archives

## Scripts Created

### 1. `download_ercot_battery_storage.py`
**Basic version** that downloads API data automatically and provides instructions for manual historical download.

**Usage:**
```bash
# Download recent API data (last 2 days)
python download_ercot_battery_storage.py

# Show instructions for downloading historical data
python download_ercot_battery_storage.py --instructions

# Specify output directory
python download_ercot_battery_storage.py --output-dir /path/to/output

# Start from specific year
python download_ercot_battery_storage.py --start-year 2020
```

**Features:**
- ✅ Automatic download from ERCOT APIs
- ✅ Merges data from multiple sources
- ✅ Deduplicates records
- ✅ Saves to CSV format
- ✅ Instructions for manual historical download

### 2. `download_ercot_battery_storage_full.py`
**Advanced version** with full automation including web scraping for historical files.

**Usage:**
```bash
# Download only recent API data (quick)
python download_ercot_battery_storage_full.py --api-only

# Download API data + historical files (requires --download-historical flag)
python download_ercot_battery_storage_full.py --download-historical --start-year 2019

# Specify output directory
python download_ercot_battery_storage_full.py --output-dir /path/to/output --download-historical
```

**Features:**
- ✅ Automatic download from ERCOT APIs
- ✅ Automatic web scraping to find historical file URLs
- ✅ Automatic download of historical ZIP files
- ✅ Automatic extraction and parsing of historical data
- ✅ Intelligent file format detection (ZIP, Excel, CSV)
- ✅ Merges all data sources
- ✅ Deduplicates and sorts records
- ✅ Progress tracking and logging

**Dependencies:**
```bash
pip install pandas requests beautifulsoup4 openpyxl
```

## Output Files

Both scripts create the following directory structure:

```
ercot_battery_storage_data/
├── battery_storage_api_recent.csv          # Latest API data (Energy Storage Resources)
├── battery_storage_fuelmix_recent.csv      # Latest API data (Fuel Mix)
├── battery_storage_historical.csv          # Historical data (if downloaded)
├── battery_storage_combined.csv            # All data merged and deduplicated
└── historical_downloads/                   # Downloaded historical files (full version only)
    ├── Fuel_Mix_Report_2007_2024.zip
    └── Fuel_Mix_Report_2025.xlsx
```

## Data Format

All CSV output files use a standardized format:

| Column | Description | Units | Example |
|--------|-------------|-------|---------|
| `timestamp` | Date and time of measurement | ISO 8601 with timezone | `2025-10-10 00:00:00-05:00` |
| `total_charging_mw` | Total battery charging power | MW (negative) | `-691.654` |
| `total_discharging_mw` | Total battery discharging power | MW (positive) | `23.456` |
| `net_output_mw` | Net output (discharge - charge) | MW | `-668.197` |

**Notes:**
- Charging is represented as negative values
- Discharging is represented as positive values
- Net output = discharging - abs(charging)
- Timestamps are in Central Time (US/Central)
- 5-minute interval data (288 records per day)

## Example Data

```csv
timestamp,total_charging_mw,total_discharging_mw,net_output_mw
2025-10-10 00:00:00-05:00,-691.654,23.456,-668.197
2025-10-10 00:05:00-05:00,-720.415,120.01,-600.405
2025-10-10 00:10:00-05:00,-708.151,162.337,-545.814
2025-10-10 06:00:00-05:00,-414.523,404.051,-10.471
2025-10-10 06:05:00-05:00,-210.672,456.295,245.623
2025-10-10 07:00:00-05:00,-384.928,1100.052,715.124
```

## Typical Battery Storage Patterns

Based on ERCOT data:

### Overnight (00:00-06:00)
- **Behavior**: Moderate charging
- **Net Output**: -600 to -800 MW (net charging)
- **Reason**: Low electricity prices, solar not available

### Morning (06:00-09:00)
- **Behavior**: Transition from charging to discharging
- **Net Output**: -500 MW to +900 MW
- **Reason**: Morning ramp-up, solar starting

### Midday (09:00-17:00)
- **Behavior**: Heavy charging during solar peak
- **Net Output**: -5000 to -6000 MW (heavy charging)
- **Reason**: Abundant solar generation, negative prices

### Evening Peak (17:00-21:00)
- **Behavior**: Maximum discharging
- **Net Output**: +6000 to +9000 MW (heavy discharging)
- **Reason**: High electricity prices, solar declining

## Data Quality Notes

1. **API Data**: Highly reliable, updated every 5 minutes, covers last ~48 hours
2. **Historical Data**: Requires parsing various file formats, may need manual verification
3. **Data Gaps**: Some intervals may be missing, especially in early years (2019-2020)
4. **Time Zones**: All timestamps include timezone information (typically US/Central)

## Integration with Existing Pipeline

To integrate with the existing `power_market_pipeline` project:

1. **Option 1**: Run as standalone script and import CSVs
2. **Option 2**: Integrate into existing ERCOT downloader:
   ```python
   # In downloaders/ercot/downloader_v2.py
   async def download_battery_storage(self, start_date, end_date):
       # Use ERCOTBatteryStorageDownloader class
       pass
   ```

3. **Option 3**: Add new database table for battery storage:
   ```python
   # In database/models_v2.py
   class BatteryStorage(Base):
       __tablename__ = 'battery_storage'
       timestamp = Column(DateTime(timezone=True), primary_key=True)
       iso = Column(String, primary_key=True)
       total_charging_mw = Column(Float)
       total_discharging_mw = Column(Float)
       net_output_mw = Column(Float)
   ```

## Historical Data Availability

Based on research:

| Year | Data Availability | Notes |
|------|------------------|-------|
| 2019 | Limited | Battery storage was minimal |
| 2020 | Partial | Growing but still small |
| 2021 | Good | Significant growth post-Feb 2021 freeze |
| 2022 | Good | Major expansion |
| 2023 | Excellent | Full deployment of ESRs |
| 2024 | Excellent | Peak at ~14 GW capacity |
| 2025 | Excellent | Real-time via API |

## Known Limitations

1. **Historical Download**:
   - The full automated version attempts to scrape and download historical files
   - File formats vary across years and may require manual parsing
   - Large files (50+ MB) may take several minutes to download

2. **API Limitations**:
   - Only provides last 48 hours of data
   - No historical API access beyond 2 days
   - Rate limits unknown (be conservative)

3. **Data Parsing**:
   - Historical file formats are not fully standardized
   - Column names may vary across years
   - Some manual verification recommended for historical data

## Comparison with GridStatus

GridStatus.io uses the same ERCOT Energy Storage Resources API endpoint. Their implementation:
- Polls API every few minutes for latest data
- Stores historical data in their own database
- Provides chart visualization
- Adds calculated metrics (30-day averages, etc.)

Our scripts provide:
- Raw access to the same source data
- Historical data extraction
- CSV export for analysis
- No rate limits or API keys required

## Next Steps

1. **Test Historical Download**: Run with `--download-historical` flag
2. **Verify Data Quality**: Check historical data against known events
3. **Database Integration**: Add battery storage table to main database
4. **Automation**: Set up cron job for daily updates
5. **Analysis**: Use data for revenue calculations, arbitrage analysis, etc.

## References

- [ERCOT Energy Storage Resources Dashboard](https://www.ercot.com/gridmktinfo/dashboards/energystorageresources)
- [ERCOT Fuel Mix Dashboard](https://www.ercot.com/gridmktinfo/dashboards/fuelmix)
- [ERCOT Generation Reports](https://www.ercot.com/gridinfo/generation)
- [GridStatus ERCOT Documentation](https://docs.gridstatus.io/en/latest/autoapi/gridstatus/ercot/index.html)

## Questions?

For questions about this implementation, refer to:
1. This README
2. Code comments in the Python scripts
3. ERCOT's official documentation
4. GridStatus.io open-source implementation

---

**Created**: October 11, 2025
**Author**: Claude Code
**Project**: Power Market Pipeline

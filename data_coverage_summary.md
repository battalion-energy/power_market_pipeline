# ERCOT Historical Data Coverage Summary

## Current Database Status

### LMP Data
- **DAM (Day-Ahead Market)**: 
  - Coverage: December 13, 2023 - July 26, 2025
  - Records: 561,000
  - Locations: 1,095
  - Source: ERCOT WebService API (via power_market_pipeline)

### Available Historical Data (from cron job)

1. **DAM Hourly LMPs** (`/Users/enrico/data/ERCOT_data/DAM_Hourly_LMPs/`)
   - Coverage: June 5, 2024 - March 18, 2025
   - Format: Hourly prices by delivery point
   - Overlap: June 2024 - March 2025 already in database

2. **RT Settlement Point Prices** (`/Users/enrico/data/ERCOT_data/Settlement_Point_Prices_at_Resource_Nodes,_Hubs_and_Load_Zones/`)
   - Coverage: August 23, 2024 - July 26, 2025
   - Format: 15-minute real-time prices
   - Status: Not yet in database

3. **DAM Clearing Prices for Capacity** (`/Users/enrico/data/ERCOT_data/DAM_Clearing_Prices_for_Capacity/`)
   - Coverage: April 30, 2024 - July 25, 2025
   - Format: Ancillary services prices (REGUP, REGDN, RRS, NSPIN, ECRS)
   - Status: Not yet in database

## Data Gaps

### Historical Data Needed (Pre-Dec 2023)
To get data from January 1, 2019 to December 12, 2023, we need:
- Implement ERCOT Selenium scraper for MIS portal access
- Download historical files from ERCOT's data products page
- Process approximately 5 years of data

### Missing Real-Time Data
- RT data before August 2024 is not available in the cron job downloads
- Need to implement RT data collection for historical periods

## Recommendations

1. **Import RT Data**: Process the available RT settlement point prices (Aug 2024 - July 2025)
2. **Import Ancillary Services**: Process DAM ancillary services data (April 2024 - July 2025)
3. **Implement Selenium Scraper**: For pre-2023 historical data
4. **Set up Historical Download**: Create script to fetch missing historical files

## File Formats

### DAM LMP Files
- Columns: DeliveryDate, HourEnding, BusName, LMP, DSTFlag
- ~416,520 records per day (17,355 locations × 24 hours)

### RT SPP Files  
- Columns: DeliveryDate, DeliveryHour, DeliveryInterval, SettlementPointName, SettlementPointType, SettlementPointPrice, DSTFlag
- 15-minute intervals (96 per day)

### Ancillary Services Files
- Columns: DeliveryDate, HourEnding, AncillaryType, MCPC, DSTFlag
- Types: REGUP, REGDN, RRS, NSPIN, ECRS
- Hourly clearing prices
# ISO API Reference Guide

Complete reference for downloading energy market data from all North American ISOs.

Date Range: **2019-01-01 to 2025-10-10**

## Data Requirements Summary

For each ISO, download:

### Energy Markets
- **Day-Ahead Market (DAM)** - Hourly LMP data
- **Real-Time Market (RTM)** - 5-minute or 15-minute LMP data
  - Hub prices (major trading hubs)
  - Zonal prices (load zones)
  - Nodal prices (all pricing nodes)

### Ancillary Services Markets
- **Regulation Up/Down** (REG_UP, REG_DOWN)
- **Spinning Reserves** (SPIN)
- **Non-Spinning Reserves** (NON_SPIN)
- **Supplemental/Secondary Reserves** (if available)
- **30-minute Operating Reserves** (30OR, OPER_30)
- **10-minute Reserves** (10S, 10NS)

---

## 1. NYISO (New York ISO)

### API Type
**Public CSV Downloads** (No authentication required)

### Base URL
```
http://mis.nyiso.com/public/csv/
```

### Data Endpoints

#### Day-Ahead Market LMP
- **Zonal**: `damlbmp/{YYYYMMDD}damlbmp_zone.csv`
- **Generator**: `damlbmp/{YYYYMMDD}damlbmp_gen.csv`

#### Real-Time Market LMP
- **Zonal**: `realtime/{YYYYMMDD}realtime_zone.csv`
- **Generator**: `realtime/{YYYYMMDD}realtime_gen.csv`

#### Ancillary Services
- **DAM AS Prices**: `damasp/{YYYYMMDD}damasp.csv`
- **RT AS Prices**: `rtasp/{YYYYMMDD}rtasp.csv`

#### Load Data
- **Actual Load**: `pal/{YYYYMMDD}pal.csv`
- **Forecast Load**: `isolf/{YYYYMMDD}isolf.csv`

### Available Products
- **10-Minute Spinning Reserve** (10S)
- **10-Minute Non-Sync Reserve** (10NS)
- **30-Minute Reserve** (30OR)
- **Regulation Capacity** (REG_CAP)
- **Regulation Movement** (REG_MOV)

### Major Zones
- CAPITL, CENTRL, DUNWOD, GENESE, HUD VL, LONGIL, MHK VL, MILLWD, N.Y.C., NORTH, WEST

### Implementation Notes
- CSV format with timestamp, zone/location, and price columns
- 5-minute real-time granularity
- Hourly day-ahead granularity
- Download by date (one file per day)
- No rate limiting on public CSVs

---

## 2. CAISO (California ISO)

### API Type
**OASIS REST API** (Authentication required)

### Base URL
```
http://oasis.caiso.com/oasisapi/SingleZip
```

### Authentication
- Username/Password required
- Set in environment: `CAISO_USERNAME`, `CAISO_PASSWORD`

### Query Structure
```
?resultformat=6
&queryname={QUERY_NAME}
&version={VERSION}
&market_run_id={MARKET}
&node={NODE_ID}
&startdatetime={YYYYMMDD}T00:00-0000
&enddatetime={YYYYMMDD}T23:59-0000
```

### Key Query Names

#### Energy Prices
- **PRC_LMP** - Locational Marginal Prices
  - Markets: DAM, RTM, HASP
  - Components: LMP_PRC (total), LMP_CONG_PRC (congestion), LMP_LOSS_PRC (losses), LMP_ENE_PRC (energy)

#### Ancillary Services
- **PRC_AS** - Ancillary Service Prices
  - Products: RU (Reg Up), RD (Reg Down), SR (Spinning Reserve), NR (Non-Spinning Reserve)
  - Markets: DAM, RTM

#### Load
- **SLD_FCST** - System Load Forecast
- **SLD_REN_FCST** - Renewable Forecast

### Available Locations

#### Trading Hubs
- **TH_NP15_GEN-APND** - North Path 15
- **TH_SP15_GEN-APND** - South Path 15
- **TH_ZP26_GEN-APND** - Zone Path 26

#### Aggregated Zones
- **PGAE-APND** - PG&E
- **SCE-APND** - Southern California Edison
- **SDGE-APND** - San Diego Gas & Electric
- **VEA-APND** - Valley Electric Association

### Implementation Notes
- ZIP files containing CSV data
- 15-minute real-time granularity
- Hourly day-ahead granularity
- Need to extract ZIP and parse CSV
- Rate limit: ~10 requests/minute (implement retry logic)
- Version numbers change - use latest (12+ for LMP)

---

## 3. SPP (Southwest Power Pool)

### API Type
**Marketplace REST API** (Digital certificate authentication required)

### Base URL
```
https://marketplace.spp.org/
```

### Authentication
**Digital Certificate Required**
- Contact: Customer Relations (501) 614-3200
- Email customer relations rep with certificate info
- Request API subscription approval

### API Documentation
- **Latest Version**: marketplace markets web service 41.0 (Jan 23, 2025)
- Available at: `https://www.spp.org/spp-documents-filings/?id=21070`

### Data Access Endpoints

#### Day-Ahead Market
- **LMP by Location**: `pages/da-lmp-by-location`
- **LMP by Bus**: `pages/da-lmp-by-bus`

#### Real-Time Balancing Market
- **LMP by Location**: `pages/rtbm-lmp-by-location`
- **LMP by Bus**: `pages/rtbm-lmp-by-bus`

#### Ancillary Services
- **Operating Reserves**: `pages/operating-reserves`
  - Regulation Reserve
  - Spinning Reserve
  - Supplemental Reserve

### Markets
- Day-Ahead Market with TCRs
- Real-Time Balancing Market
- Co-optimized energy and ancillary services

### Implementation Notes
- All data transported via HTTPS with digital certificates
- Need to register and wait for subscription approval
- May require weeks for certificate approval process
- Alternative: Use gridstatus.io library (if available)

---

## 4. IESO (Independent Electricity System Operator - Ontario)

### API Type
**REST API** (Public and confidential endpoints)

### Base URL
```
https://reports-public.ieso.ca/
https://www.ieso.ca/localcontent/reports_api/api/v1.1/
```

### Authentication
- Public reports: No authentication
- Confidential reports: API key required

### Market Changes (May 1, 2025)
**Major Market Renewal:**
- HOEP (Hourly Ontario Energy Price) retired April 30, 2025
- New: **Locational Marginal Pricing (LMP)** system
- New: **Ontario Energy Market Price (OEMP)**
- New: Single schedule day-ahead market

### Data Reports

#### Energy Prices (Post May 1, 2025)
- **Day-Ahead LMP**: Report code `PUB_DALMPEnergy`
- **Real-Time LMP**: Report code `PUB_RTLMPEnergy`
- **Ontario Zonal Prices**: Report code `PUB_OntarioZonalPrice`
- **OEMP**: Ontario Energy Market Price (replacement for HOEP)

#### Energy Prices (Pre May 1, 2025)
- **HOEP**: Hourly Ontario Energy Price (legacy)
- **Market Clearing Price (MCP)**: Zonal prices

#### Ancillary Services
- **Operating Reserve Markets**:
  - 10-minute Synchronized (10S)
  - 10-minute Non-Synchronized (10NS)
  - 30-minute Operating Reserve (30OR)

- **Contracted Services**:
  - Regulation Service
  - Black Start
  - Reactive Support/Voltage Control
  - Reliability Must-Run

### Ontario Zones (10 zones)
- Zone names available in zonal price reports

### API Access Pattern
```
GET https://reports-public.ieso.ca/public/{report_code}_{YYYYMMDD}.csv
```

### Implementation Notes
- Public CSV files available via direct URL
- REST API for programmatic access: v1.1
- Almost 1,000 LMP nodes in Ontario (post-May 2025)
- Need to handle both legacy (HOEP) and new (LMP) data formats
- Time series split at 2025-05-01

---

## 5. AESO (Alberta Electric System Operator)

### API Type
**File Downloads** (No formal REST API)

### Base URL
```
http://ets.aeso.ca/
```

### Available Data

#### Pool Price and Generation
- **Hourly Generation and Pool Price**: 2001-2025
- **Files**: Available in multi-year chunks
  - 2001-2009
  - 2010-2019
  - 2020-2025 (through July 2025)

#### Reports
- **Daily Average Pool Price**: `ets_web/ip/Market/Reports/DailyAveragePoolPriceReportServlet`
- **CSD Report**: `ets_web/ip/Market/Reports/CSDReportServlet`

### Ancillary Services
- Operating Reserve
- Transmission Must-Run
- Blackstart
- Load Shed Services

### Data Request Process
**For non-standard data:**
- Download data request form from website
- Email to: manalysis@aeso.ca
- Manual processing by AESO staff

### Implementation Notes
- No formal API - file downloads only
- Large files covering multiple years
- May need web scraping for certain reports
- Consider gridstatus.io library as alternative
- Pool price = single-price market (no nodal/zonal LMP)

---

## Implementation Strategy

### Phase 1: Direct API Access (NYISO, CAISO, IESO)
These have public or well-documented APIs - implement first.

### Phase 2: Certificate-Based (SPP)
Requires registration and digital certificates - may take weeks to get access.

### Phase 3: File-Based (AESO)
Download bulk files, parse and convert to parquet.

### Phase 4: Alternative via gridstatus.io
If direct API access is difficult, consider using the `gridstatus` Python library:
```python
import gridstatus
nyiso = gridstatus.NYISO()
caiso = gridstatus.CAISO()
spp = gridstatus.SPP()
# etc.
```

---

## Output Format

All data will be standardized to:

### CSV Format (Download Stage)
- One directory per ISO: `{ISO}_data/csv_files/`
- Subdirectories by data type: `da_lmp/`, `rt_lmp/`, `ancillary_services/`
- File naming: `{dataset}_{YYYYMMDD}.csv`

### Parquet Format (Processing Stage)
- Annual files: `{ISO}_data/rollup_files/{dataset}_{YYYY}.parquet`
- Standardized columns:
  - `datetime` (UTC timestamp)
  - `settlement_point` (location identifier)
  - `da_lmp`, `rt_lmp` (energy prices)
  - `as_reg_up`, `as_reg_down`, `as_spin`, `as_non_spin` (ancillary service prices)
  - `location_type` (hub/zone/node)

---

## Timeline Estimate

### NYISO: 1-2 days
- Simple CSV downloads
- No authentication

### CAISO: 2-3 days
- OASIS API implementation
- ZIP extraction logic
- Authentication handling

### IESO: 2-3 days
- REST API implementation
- Handle market transition (May 2025)
- Dual format support

### SPP: 1-2 weeks
- Digital certificate registration
- Wait for approval
- API implementation

### AESO: 3-4 days
- File download automation
- Multi-year file parsing
- Limited data availability

### Total: 2-4 weeks
(Depending on SPP certificate approval time)

---

## Next Steps

1. ✅ Research complete
2. Create downloader implementations
3. Create orchestration script for parallel downloads
4. Test with small date ranges (Jan 2024)
5. Execute full historical downloads (2019-2025)
6. Generate annual parquet files
7. Verify data integrity

---

*Document created: 2025-10-10*
*Data range: 2019-01-01 to 2025-10-10*

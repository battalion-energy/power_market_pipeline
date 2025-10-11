# ISO Data Requirements and Current Coverage

## Data Requirements Overview

### Required Data Types (All ISOs)

1. **Energy Markets**
   - Day-Ahead Energy Prices (Hub & Nodal, Hourly)
   - Real-Time Energy Prices (Hub & Nodal, 5min/15min)

2. **Ancillary Services**
   - Regulation Up
   - Regulation Down
   - Spinning Reserves
   - Non-Spinning Reserves
   - Supplemental/Secondary Reserves
   - ECRS (Energy & Capacity Reserve Service - ERCOT specific)
   - RRS (Responsive Reserve Service - ERCOT specific)

3. **Load & Generation**
   - Total System Load (Actual)
   - Load Forecast
   - Fuel Mix (by fuel type)
   - Wind Production (Actual)
   - Wind Forecast
   - Solar Production (Actual)
   - Solar Forecast
   - Battery Discharge/Charge (if available)

4. **System Conditions**
   - Available Reserves / Reserve Margin
   - Generation Outages (Planned & Forced)
   - Curtailment (Wind/Solar)
   - Transmission Constraints/Congestion

---

## ERCOT - Current Coverage

### ✅ Currently Downloading
- ✅ DA Energy Prices (Settlement Point Prices, Hourly)
- ✅ RT Energy Prices (SCED LMPs, 5-minute)
- ✅ AS Prices: Reg Up, Reg Down, RRS, ECRS (Hourly)
- ✅ System Load (Actual, 15-minute)
- ✅ Wind Production (Actual, 15-minute via SCED telemetry)
- ✅ Solar Production (Actual, 15-minute via SCED telemetry)
- ✅ Generation Resource Data (60-day disclosure)
- ✅ Load Resource Data (60-day disclosure)
- ✅ DAM Awards/Bids (60-day disclosure)

### ❌ Missing / Need to Add
- ❌ Load Forecast (multiple horizons: STLF, MTLF, LTLF)
- ❌ Wind Forecast (DAM, STWPF, HRWPF)
- ❌ Solar Forecast (DAM, STSPF)
- ❌ Fuel Mix by Type (Coal, Gas, Nuclear, etc.)
- ❌ Available Reserves / Reserve Margin
- ❌ Generation Outages (Planned & Forced)
- ❌ Curtailment Data
- ❌ Transmission Constraints

**ERCOT Data Sources:**
- Public MIS: Load forecasts, Wind/Solar forecasts, Fuel Mix
- API: Real-time operational data
- 60-Day Disclosure: Historical generation/load/bids

---

## MISO - Current Coverage

### ✅ Currently Downloading
- ✅ DA Ex-Post LMP (Hub only, Hourly, 2019-2025)
- ✅ DA Ex-Ante LMP (Hub only, Hourly, 2024-2025)
- ✅ RT LMP Final (Hub only, Hourly aggregated, 2024-2025)

### ❌ Missing / Need to Add
- ❌ **DA LMP - Nodal Level** (currently only hubs)
- ❌ **RT LMP - 5-minute** (currently only hourly aggregated)
- ❌ **RT LMP - Nodal Level** (currently only hubs)
- ❌ **Ancillary Services Prices:**
  - ❌ Regulation (Reg Up, Reg Down)
  - ❌ Spinning Reserves
  - ❌ Supplemental Reserves
- ❌ **System Load** (Actual & Forecast)
- ❌ **Generation Mix** (Fuel type breakdown)
- ❌ **Wind Production** (Actual & Forecast)
- ❌ **Solar Production** (Actual & Forecast)
- ❌ **Battery Data** (Discharge/Charge)
- ❌ **Available Reserves**
- ❌ **Generation Outages**
- ❌ **Curtailment Data**
- ❌ **Transmission Constraints**

**MISO Data Sources:**
- **Pricing API**: LMP (ex-ante/ex-post), MCP (ancillary services)
- **Load & Generation API**: Load, generation mix, wind/solar
- **Public CSV**: Historical LMP data
- **Market Reports Portal**: Various operational data

**Critical Gaps to Address:**
1. Enable nodal-level data (remove hub filter or run separately)
2. Add MCP (Market Clearing Price) downloads for ancillary services
3. Add Load & Generation API endpoints
4. Add 5-minute RT data (not just hourly aggregated)

---

## PJM - Current Coverage

### ✅ Currently Downloading
- ✅ DA LMP (Nodal, Hourly)
- ✅ RT LMP (Nodal, 5-minute)
- ✅ System Load (Actual, 5-minute)

### ❌ Missing / Need to Add
- ❌ **Ancillary Services Prices:**
  - ❌ Regulation
  - ❌ Spinning Reserves
  - ❌ Non-Spinning Reserves
  - ❌ Synchronized Reserves
- ❌ **Load Forecast**
- ❌ **Generation Mix** (Fuel type)
- ❌ **Wind Production** (Actual & Forecast)
- ❌ **Solar Production** (Actual & Forecast)
- ❌ **Battery Data**
- ❌ **Available Reserves**
- ❌ **Generation Outages**
- ❌ **Curtailment Data**

**PJM Data Sources:**
- API/FTP: LMP, load, ancillary services
- Data Miner: Historical data queries
- OASIS: Various market data

---

## CAISO - Current Coverage

### ✅ Currently Downloading
- ✅ DA LMP (Nodal, Hourly)
- ✅ RT LMP (Nodal, 5-minute)

### ❌ Missing / Need to Add
- ❌ **Ancillary Services Prices:**
  - ❌ Regulation Up
  - ❌ Regulation Down
  - ❌ Spinning Reserves
  - ❌ Non-Spinning Reserves
- ❌ **System Load** (Actual & Forecast)
- ❌ **Generation Mix** (Fuel type)
- ❌ **Wind Production** (Actual & Forecast)
- ❌ **Solar Production** (Actual & Forecast)
- ❌ **Battery Data**
- ❌ **Available Reserves**
- ❌ **Generation Outages**
- ❌ **Curtailment Data**

**CAISO Data Sources:**
- OASIS API: LMP, AS prices, load, generation
- Today's Outlook: Real-time system conditions
- EIM Transfer Limits: Inter-tie data

---

## NYISO - Current Coverage

### ✅ Currently Downloading
- ✅ DA LBMP (Zone level, Hourly, 2024-2025)
- ✅ RT LBMP (Zone level, 5-minute, 2024-2025)
- ✅ DA Ancillary Services (All products, Hourly)
- ✅ RT Ancillary Services (All products, 5-minute)
- ✅ Actual Load (Zone level, Hourly)
- ✅ Load Forecast (Zone level, Hourly)

### ❌ Missing / Need to Add
- ❌ **Generation Mix** (Fuel type)
- ❌ **Wind Production** (Actual & Forecast)
- ❌ **Solar Production** (Actual & Forecast)
- ❌ **Battery Data**
- ❌ **Available Reserves**
- ❌ **Generation Outages**
- ❌ **Curtailment Data**
- ❌ **Nodal-level prices** (currently only zones)

**NYISO Data Sources:**
- Public CSV Downloads: LMP, AS, Load
- Real-Time Dashboard: System conditions
- OASIS: Historical data

---

## ISO-NE - Current Coverage

### ✅ Currently Downloading
- ✅ DA LMP (Nodal, Hourly)
- ✅ RT LMP (Nodal, 5-minute)

### ❌ Missing / Need to Add
- ❌ **Ancillary Services Prices:**
  - ❌ Regulation
  - ❌ Spinning Reserves
  - ❌ Non-Spinning Reserves
  - ❌ 10-minute Reserves
  - ❌ 30-minute Reserves
- ❌ **System Load** (Actual & Forecast)
- ❌ **Generation Mix** (Fuel type)
- ❌ **Wind Production** (Actual & Forecast)
- ❌ **Solar Production** (Actual & Forecast)
- ❌ **Battery Data**
- ❌ **Available Reserves**
- ❌ **Generation Outages**
- ❌ **Curtailment Data**

**ISO-NE Data Sources:**
- Web Services API: LMP, AS, load, generation
- ISO Express: Historical data
- OASIS: Various market data

---

## Summary - Data Coverage by ISO

| Data Type | ERCOT | MISO | PJM | CAISO | NYISO | ISO-NE |
|-----------|-------|------|-----|-------|-------|--------|
| **Energy Markets** |
| DA LMP (Hub) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| DA LMP (Nodal) | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ |
| RT LMP (5min/Nodal) | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ |
| **Ancillary Services** |
| Regulation Up/Down | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ |
| Spinning Reserves | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ |
| Non-Spin Reserves | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ |
| **Load & Generation** |
| System Load (Actual) | ✅ | ❌ | ✅ | ❌ | ✅ | ❌ |
| Load Forecast | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |
| Fuel Mix | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Wind Production | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Solar Production | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Wind/Solar Forecast | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Battery Data | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **System Conditions** |
| Available Reserves | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Generation Outages | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Curtailment | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

---

## Priority Actions

### Immediate (Current MISO Work)
1. ✅ Complete MISO hub-level LMP download (in progress - 50% done)
2. ⚠️ **Add MISO nodal-level LMP download**
3. ⚠️ **Add MISO ancillary services (MCP) download**
4. ⚠️ **Add MISO 5-minute RT LMP (not just hourly)**
5. ⚠️ **Add MISO load data download**

### High Priority (Next Steps)
1. Add ancillary services for all ISOs
2. Add load data (actual & forecast) for all ISOs
3. Add generation mix/fuel type data
4. Add wind/solar production and forecasts

### Medium Priority
1. Available reserves / reserve margin
2. Generation outages
3. Curtailment data
4. Transmission constraints

---

## Next Steps

1. **MISO - Complete Current Downloads:**
   - Let API download finish (currently at April 2020, ~50% done)
   - Should complete in ~30-40 minutes

2. **MISO - Expand Data Coverage:**
   - Remove hub filter to get nodal data
   - Add MCP (ancillary services) endpoints
   - Add Load & Generation API endpoints
   - Add 5-minute RT data

3. **All ISOs - Add Missing Data Types:**
   - Implement ancillary services downloads
   - Implement load data downloads
   - Implement generation/fuel mix downloads
   - Implement renewable production/forecast downloads

4. **Documentation:**
   - Create endpoint mapping documents for each ISO
   - Document data availability windows
   - Create data dictionary for standardization

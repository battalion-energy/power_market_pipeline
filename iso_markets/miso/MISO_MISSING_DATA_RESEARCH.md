# MISO Missing Data Types - Research Summary

This document provides comprehensive research on MISO data types not currently downloaded, including data sources, APIs, and implementation strategies.

**Date:** 2025-10-11

---

## Overview

MISO is transitioning to a new **MISO Data Exchange** portal that will be the primary data access method starting December 12, 2025. Many of the data sources below are available through this portal.

**Main Portals:**
- **MISO Data Exchange**: https://data-exchange.misoenergy.org/
- **RT Data API**: https://www.misoenergy.org/markets-and-operations/rtdataapis/
- **Market Reports**: https://www.misoenergy.org/markets-and-operations/real-time--market-data/market-reports/
- **MISO OASIS**: http://www.oasis.oati.com/MISO/index.html

---

## 1. Wind and Solar Forecasts

### Data Availability: ✅ Available

**Sources:**
1. **MISO Data Exchange API**
   - Wind and solar forecasts available through the new Data Exchange portal
   - Requires API key from https://data-exchange.misoenergy.org/
   - Refreshed once per day

2. **Real-Time Data API**
   - Base URL: `https://api.misoenergy.org/MISORTWDDataBroker/`
   - Solar forecast display: `https://api.misoenergy.org/MISORTWD/operations.html?dayAheadAndRealTimeSolar=`
   - Wind forecast display: Available through similar endpoint pattern

**Forecast Details:**
- **Update Frequency**: Every 15 minutes
- **Forecast Horizon**: Up to 168 hours (7 days) into the future
- **Granularity**: Hourly
- **Methodology**: Combines numerical weather prediction models with real-time data
- **Factors Considered**: Wind speed, solar irradiance, cloud cover, precipitation

**Data Format:**
- Available as JSON/XML via API
- Real-time displays also available via web interface

**Implementation Priority:** 🟡 Medium
- Need to explore Data Exchange API documentation
- Should create downloader script similar to existing LMP downloaders
- Useful for renewable energy analysis and forecasting

**Next Steps:**
1. Subscribe to appropriate APIs in Data Exchange portal
2. Review API documentation for wind/solar forecast endpoints
3. Create `download_renewables_forecast.py` script
4. Implement historical data download (if available)

---

## 2. Battery Energy Storage (Charge/Discharge Data)

### Data Availability: ⚠️ Limited

**Sources:**
1. **Market Reports - Day-Ahead and Real-Time Offers**
   - Electric Storage Resources (ESR) data included in offer reports
   - Available through MISO Data Exchange starting Dec 12, 2025
   - Current location: Market Reports page

2. **Data Access Methods:**
   - **MISO Data Exchange**: https://data-exchange.misoenergy.org/
   - **Market Report Archives**: Historical ESR data
   - **RT Data API**: Real-time ESR operations data

**What Data Is Available:**
- Day-Ahead offers from battery resources
- Real-Time offers and dispatch instructions
- Settlement data (charge/discharge events)
- State of charge information may be limited or restricted

**Data Limitations:**
- Battery-specific operational data may require NDA or be restricted
- Real-time state of charge data may not be publicly available
- Historical data likely limited to market settlement periods

**MISO ESR Context:**
- MISO introduced Electric Storage Resource (ESR) designation for batteries
- Batteries can participate in both energy and ancillary services markets
- ESRs registered as both generation (discharge) and load (charge) resources

**Implementation Priority:** 🔴 High (if available)
- Critical for battery arbitrage and market participation analysis
- Check what specific ESR data is available publicly
- May need to request special access or NDA for detailed operational data

**Next Steps:**
1. Review ESR data availability in MISO Data Exchange
2. Check if real-time telemetry data is accessible
3. Investigate Day-Ahead and Real-Time offer reports
4. Create `download_battery_esr.py` if data is accessible
5. Consider alternative sources (e.g., EIA Form 930 for aggregate storage data)

---

## 3. Available Reserves / Reserve Margin

### Data Availability: ✅ Available

**Sources:**
1. **MISO Data Exchange API**
   - Reserve margin data available through APIs
   - Documentation: https://data-exchange.misoenergy.org/apis

2. **RT Data API**
   - Real-time reserve data
   - Base URL: https://www.misoenergy.org/markets-and-operations/rtdataapis/

3. **Planning Reserve Margin Reports**
   - Resource adequacy page: https://www.misoenergy.org/planning/resource-adequacy2/resource-adequacy/
   - Annual Loss of Load Expectation (LOLE) studies
   - Planning Reserve Margin determinations

**Types of Reserve Data:**
1. **Operating Reserves:**
   - Regulation Reserve (RegUp, RegDown)
   - Spinning Reserve
   - Supplemental Reserve
   - Ramp Capability
   - 30-minute Short-Term Reserve

2. **Reserve Margin:**
   - Planning Reserve Margin (annual)
   - Operating Reserve Margin (real-time)
   - Available capacity vs. required reserves

**Data Format:**
- Real-time operating reserves via API (JSON/XML)
- Historical reserves in Market Reports (XLS/CSV)
- Planning reserve margin in annual reports (PDF)

**Implementation Priority:** 🟡 Medium
- Operating reserves useful for market analysis
- Reserve margin data useful for capacity planning
- Some data may already be in ancillary services downloads (if we fix those URLs)

**Next Steps:**
1. Check if operating reserve data is included in ancillary services reports
2. Review RT Data API for real-time reserve status
3. Create downloader for historical reserve margin data
4. Add to existing `download_ancillary_services.py` if applicable

---

## 4. Generation Outages

### Data Availability: ✅ Available (Multiple Sources)

**Sources:**
1. **Grid Status** (Third-Party)
   - Dataset: https://www.gridstatus.io/datasets/miso_generation_outages_estimated
   - MISO Records: https://www.gridstatus.io/records/miso
   - Provides estimated outage data

2. **MISO OASIS** (Official)
   - Portal: http://www.oasis.oati.com/MISO/index.html
   - Official transmission and market data platform
   - Outage scheduling with generator impacts

3. **EIA Grid Monitor**
   - URL: https://www.eia.gov/electricity/gridmonitor/dashboard/electric_overview/balancing_authority/MISO
   - Real-time operating grid data
   - Includes generation outage information

4. **Potomac Economics State of the Market Reports**
   - Annual reports with detailed outage analysis
   - URL: https://www.potomaceconomics.com/
   - MISO is the Independent Market Monitor

**Data Types:**
- **Planned Outages**: Scheduled maintenance
- **Forced Outages**: Unplanned equipment failures
- **Partial Outages**: Reduced capacity operations
- **Outage Duration**: Start/end times
- **Generator Identification**: Unit-level data

**Data Format:**
- OASIS: Various formats (XLS, CSV, XML)
- Grid Status: CSV/JSON via API
- EIA: JSON via API
- Market Reports: PDF with aggregated analysis

**Implementation Priority:** 🟢 Low-Medium
- Useful for reliability analysis and market modeling
- Grid Status provides easiest access
- OASIS provides most authoritative data but may require login

**Next Steps:**
1. Explore Grid Status API for outage data
2. Review MISO OASIS outage scheduling system
3. Check EIA Grid Monitor API documentation
4. Create `download_generation_outages.py`
5. Consider if unit-level detail is needed or aggregate is sufficient

---

## 5. Curtailment Data

### Data Availability: ⚠️ Limited

**Sources:**
1. **EIA Grid Monitor**
   - URL: https://www.eia.gov/electricity/gridmonitor/
   - Balancing authority level curtailment data
   - Real-time and historical access

2. **State of the Market Reports**
   - Potomac Economics annual reports
   - Contains analysis of curtailments and reasons
   - URL: https://www.potomaceconomics.com/

3. **MISO Market Reports**
   - May contain curtailment information in operational reports
   - Available through Market Reports page

**Types of Curtailment:**
- **Wind Curtailment**: Wind generation reduced below available capacity
- **Solar Curtailment**: Solar generation reduced below available capacity
- **Transmission Constraints**: Curtailments due to congestion
- **Economic Curtailment**: Based on negative prices
- **Reliability Curtailment**: For system stability

**Data Challenges:**
- Detailed curtailment data may not be publicly available
- EIA provides aggregate data but not unit-level detail
- Real-time curtailment harder to track than ex-post analysis
- Need to distinguish between unavailable capacity vs. curtailment

**Implementation Priority:** 🟡 Medium
- Important for renewable energy analysis
- May require combining multiple data sources
- Consider if EIA aggregate data is sufficient

**Next Steps:**
1. Check EIA Today in Energy articles on MISO wind curtailment
2. Review State of the Market reports for historical patterns
3. Investigate if RT Data API includes curtailment signals
4. Consider creating derived metric from generation forecast vs. actual
5. Check if Market Reports include detailed curtailment data

---

## 6. Transmission Constraints

### Data Availability: ✅ Available

**Sources:**
1. **MISO Data Exchange**
   - Binding constraints data moving to Data Exchange portal
   - URL: https://data-exchange.misoenergy.org/
   - Starting December 12, 2025

2. **Market Reports - Binding Constraints**
   - Daily Day-Ahead Binding Constraints (XLS format)
   - URL: https://www.misoenergy.org/markets-and-operations/real-time--market-data/market-reports/
   - Lists constraints that were binding in day-ahead market

3. **MUI 2.0 (Market User Interface)**
   - Query Binding Limits function
   - Queries for constraints on the system by date
   - Requires MISO market participant access

4. **State of the Market Reports**
   - Annual congestion cost analysis
   - Identifies frequently binding constraints
   - URL: https://www.potomaceconomics.com/

**Data Available:**
- **Constraint Name**: Identification of transmission limit
- **Shadow Price**: Cost of constraint ($/MW)
- **Binding Hours**: When constraint was active
- **Constraint Type**: Thermal, voltage, stability
- **Location**: Transmission line or interface affected

**Related Data:**
- **Congestion Costs**: Available in LMP data (congestion component)
- **FTR/ARR Data**: Financial transmission rights related to constraints
- **Available Transfer Capability (ATC)**: OASIS data

**Implementation Priority:** 🟢 Low-Medium
- Useful for congestion analysis and power flow modeling
- Binding constraints data readily available
- Complements LMP data (which includes congestion component)

**Next Steps:**
1. Download Market Reports directory to identify binding constraints files
2. Create `download_binding_constraints.py`
3. Parse daily XLS files for binding constraint lists
4. Consider linking constraint data to LMP congestion components
5. Investigate MUI 2.0 access for more detailed constraint data

---

## 7. Real-Time 5-Minute LMP (Not Hourly Aggregated)

### Data Availability: ✅ Available

**Sources:**
1. **MISO RT Data API (Last 4 Days)**
   - Current interval: `https://api.misoenergy.org/MISORTWDBIReporter/Reporter.asmx?messageType=currentinterval&returnType=csv`
   - Today (rolling): `https://api.misoenergy.org/MISORTWDBIReporter/Reporter.asmx?messageType=rollingmarketday&returnType=csv`
   - Yesterday: `https://api.misoenergy.org/MISORTWDBIReporter/Reporter.asmx?messageType=previousmarketday&returnType=csv`

2. **Weekly Historical Files**
   - URL Pattern: `https://docs.misoenergy.org/marketreports/YYYYMMDD_5MIN_LMP.zip`
   - Date should be a Monday
   - Contains 5-minute data from two weeks before through one week before file date
   - Files are ZIP archives containing CSV data

3. **MISO Data Exchange**
   - 5-minute LMP data available through APIs
   - URL: https://data-exchange.misoenergy.org/

4. **Grid Status (Third-Party)**
   - Dataset: https://www.gridstatus.io/datasets/miso_lmp_real_time_5_min
   - Weekly dataset: https://www.gridstatus.io/datasets/miso_lmp_real_time_5_min_weekly
   - May simplify data access

**Data Format:**
- CSV format in ZIP archives
- Columns: Timestamp, Node, LMP, MLC (Loss), MCC (Congestion), MEC (Energy)
- 5-minute intervals (12 intervals per hour, 288 per day)
- ~7,000+ nodes per timestamp

**File Size:**
- Weekly files are large (~500MB+ compressed)
- Daily data: ~30-50MB uncompressed
- Full year: ~15-20GB uncompressed

**Implementation Priority:** 🔴 High
- Critical for high-frequency trading and optimization
- Essential for battery dispatch modeling
- Required for accurate arbitrage calculations
- Complements existing hourly RT LMP data

**Next Steps:**
1. Create `download_rt_5min_lmp.py`
2. Implement weekly ZIP file download and extraction
3. Add real-time download for last 4 days via API
4. Consider hub filtering vs. all nodes
5. Implement parallel processing for large files
6. Add to automated daily update process

**URL Pattern Details:**
```
Historical (weekly):
https://docs.misoenergy.org/marketreports/YYYYMMDD_5MIN_LMP.zip
where YYYYMMDD is a Monday date

Recent (API):
- Current: messageType=currentinterval
- Today: messageType=rollingmarketday
- Yesterday: messageType=previousmarketday
- 2 days ago: messageType=previousmarketday2
- 3 days ago: messageType=previousmarketday3
```

---

## Implementation Recommendations

### Immediate Priority (Implement Next)

1. **5-Minute RT LMP** 🔴
   - High value for trading and optimization
   - Data readily available via known URLs
   - Create downloader script

2. **Wind/Solar Forecasts** 🟡
   - Available through Data Exchange
   - Requires API subscription
   - Important for renewable analysis

3. **Transmission Constraints** 🟡
   - Daily binding constraints available
   - Complements existing LMP data
   - Relatively easy to implement

### Medium Priority (Future Enhancement)

4. **Available Reserves** 🟡
   - May overlap with ancillary services data
   - Check if already covered once AS downloads fixed

5. **Battery ESR Data** 🟡
   - Check availability first
   - May require special access
   - High value if accessible

6. **Curtailment Data** 🟡
   - Multiple data sources to evaluate
   - May need data fusion approach
   - Consider EIA aggregate data as starting point

### Lower Priority (As Needed)

7. **Generation Outages** 🟢
   - Multiple sources available (Grid Status, OASIS, EIA)
   - Less critical for day-to-day trading
   - Useful for reliability analysis

---

## MISO Data Exchange Transition

**Important:** Starting December 12, 2025, MISO is transitioning most market reports to the new Data Exchange portal.

**Action Items:**
1. Create account at https://data-exchange.misoenergy.org/
2. Review available APIs and subscribe to needed products
3. Update existing downloaders to use Data Exchange where applicable
4. Test API authentication and data retrieval
5. Document new API endpoints and parameters

**Learning Resources:**
- MISO Data Exchange User Guide (Learning Center)
- API documentation: https://data-exchange.misoenergy.org/apis
- FAQs: https://help.misoenergy.org/knowledgebase/article/KA-01489/en-us

---

## Summary Table

| Data Type | Availability | Priority | Source | Implementation Effort |
|-----------|--------------|----------|--------|---------------------|
| Wind/Solar Forecasts | ✅ Available | 🟡 Medium | Data Exchange API | Medium |
| Battery Charge/Discharge | ⚠️ Limited | 🔴 High | Market Reports | High (if available) |
| Available Reserves | ✅ Available | 🟡 Medium | RT Data API | Low |
| Generation Outages | ✅ Available | 🟢 Low | Grid Status, OASIS | Medium |
| Curtailment | ⚠️ Limited | 🟡 Medium | EIA, Market Reports | High |
| Transmission Constraints | ✅ Available | 🟡 Medium | Market Reports | Low |
| 5-Min RT LMP | ✅ Available | 🔴 High | docs.misoenergy.org | Medium |

---

## References

1. MISO Data Exchange: https://data-exchange.misoenergy.org/
2. MISO RT Data API: https://www.misoenergy.org/markets-and-operations/rtdataapis/
3. MISO Market Reports: https://www.misoenergy.org/markets-and-operations/real-time--market-data/market-reports/
4. MISO OASIS: http://www.oasis.oati.com/MISO/index.html
5. Grid Status: https://www.gridstatus.io/
6. EIA Grid Monitor: https://www.eia.gov/electricity/gridmonitor/
7. Potomac Economics: https://www.potomaceconomics.com/

---

**Last Updated:** 2025-10-11

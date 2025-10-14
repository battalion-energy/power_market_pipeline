# ISO Data Coverage - Requested vs Available vs Currently Downloading

**Legend:**
- ✅ Currently Downloading
- 🟡 Available but Not Downloading
- ❌ Not Available in gridstatus
- ⚠️ Failing (gridstatus bug)

## Energy Prices (LMP)

| Data Type | NYISO | ISO-NE | CAISO | Notes |
|-----------|-------|--------|-------|-------|
| **Day-Ahead Energy (Hourly)** | ✅ ALL | ✅ ALL | ✅ ALL | All nodes/locations |
| **Real-Time Energy (5-min)** | ✅ ALL | ⚠️ FAIL | ✅ ALL | ISO-NE: gridstatus bug |
| **Real-Time Energy (15-min)** | N/A | N/A | ✅ ALL | CAISO only |
| **Hub Prices** | ✅ Included | ✅ Included | ✅ Included | In ALL locations |
| **Zonal Prices** | ✅ Included | ✅ Included | ✅ Included | In ALL locations |
| **Nodal Prices** | ✅ ALL | ✅ ALL | ✅ ALL | ~350 NYISO, ~1000 ISONE, ~4000 CAISO |

## Ancillary Services

| Data Type | NYISO | ISO-NE | CAISO | Notes |
|-----------|-------|--------|-------|-------|
| **AS Day-Ahead (Hourly)** | ✅ YES | ❌ NO | ⚠️ FAIL | ISO-NE not in gridstatus |
| **AS Real-Time (5-min)** | ✅ YES | ❌ NO | ⚠️ FAIL | |
| **Regulation Up** | ✅ YES | ❌ NO | 🟡 YES | NYISO: in AS prices |
| **Regulation Down** | ✅ YES | ❌ NO | 🟡 YES | |
| **Spinning Reserve** | ✅ YES | ❌ NO | 🟡 YES | |
| **Non-Spinning Reserve** | ✅ YES | ❌ NO | 🟡 YES | |
| **10-Min Spin** | ✅ YES | ❌ NO | 🟡 YES | |
| **10-Min Non-Spin** | ✅ YES | ❌ NO | 🟡 YES | |
| **30-Min Reserve** | ✅ YES | ❌ NO | 🟡 YES | |

**NYISO AS Types Downloaded:**
- Regulation, 10-Min Spinning, 10-Min Non-Spin, 30-Min Reserve

**CAISO AS Types Available (not downloading due to failures):**
- Regulation Up/Down, Spinning Reserve, Non-Spinning Reserve

## Load & Demand

| Data Type | NYISO | ISO-NE | CAISO | Notes |
|-----------|-------|--------|-------|-------|
| **Total Load (Actual)** | ✅ YES | ⚠️ FAIL | ✅ YES | ISO-NE: no files created |
| **Zonal Load** | 🟡 YES | ❌ NO | 🟡 YES | Not currently downloading |
| **Load Forecast (DA)** | 🟡 YES | 🟡 YES | 🟡 YES | Available in gridstatus |
| **Load Forecast (Short-term)** | 🟡 YES | 🟡 YES | 🟡 YES | 15-min, 5-min, hourly |

## Generation Mix & Resources

| Data Type | NYISO | ISO-NE | CAISO | Notes |
|-----------|-------|--------|-------|-------|
| **Fuel Mix** | 🟡 YES | ✅ YES | ✅ YES | ISO-NE: 120-180 rows/day |
| **Wind Production (Actual)** | 🟡 In Mix | 🟡 In Mix | 🟡 YES | CAISO: separate method |
| **Solar Production (Actual)** | 🟡 In Mix | 🟡 In Mix | 🟡 YES | CAISO: separate method |
| **Behind-the-Meter Solar** | 🟡 YES | 🟡 YES | ❌ NO | Available but not downloading |
| **Renewables Hourly** | ❌ NO | ❌ NO | 🟡 YES | CAISO only |

## Forecasts

| Data Type | NYISO | ISO-NE | CAISO | Notes |
|-----------|-------|--------|-------|-------|
| **Wind Forecast** | ❌ NO | 🟡 YES | 🟡 YES | |
| **Solar Forecast** | 🟡 YES | 🟡 YES | 🟡 YES | BTM solar for NYISO |
| **Renewables Forecast (DA)** | ❌ NO | ❌ NO | 🟡 YES | CAISO only |
| **Renewables Forecast (RT)** | ❌ NO | ❌ NO | 🟡 YES | CAISO: RTD, RTPD |

## Storage & Batteries

| Data Type | NYISO | ISO-NE | CAISO | Notes |
|-----------|-------|--------|-------|-------|
| **Battery Charge/Discharge** | 🟡 YES | 🟡 YES | 🟡 YES | get_storage() available |
| **Storage State Data** | 🟡 YES | 🟡 YES | 🟡 YES | Not currently downloading |

## System Conditions & Constraints

| Data Type | NYISO | ISO-NE | CAISO | Notes |
|-----------|-------|--------|-------|-------|
| **Available Reserves** | ❌ NO | ❌ NO | ❌ NO | Not in gridstatus |
| **Reserve Margin** | ❌ NO | ❌ NO | ❌ NO | Not in gridstatus |
| **Generation Outages** | ❌ NO | ❌ NO | 🟡 YES | CAISO: get_curtailed_... |
| **Curtailment** | ❌ NO | ❌ NO | 🟡 YES | CAISO only |
| **Tie Flows** | ❌ NO | ❌ NO | 🟡 YES | CAISO: RT 5-min & 15-min |
| **Interface Limits/Flows** | 🟡 YES | ❌ NO | ❌ NO | NYISO: 5-min |

## Summary Statistics

### NYISO - Currently Downloading (5 datasets)
1. ✅ LMP Day-Ahead Hourly (~360 rows/day)
2. ✅ LMP Real-Time 5-Min (~4,320 rows/day)
3. ✅ AS Day-Ahead Hourly (~264 rows/day)
4. ✅ AS Real-Time 5-Min (~3,168 rows/day)
5. ✅ Load (~580 rows/day)

**Missing but Available:**
- Fuel Mix
- BTM Solar & Forecast
- Load Forecast
- Zonal Load Forecast
- Interface Limits & Flows
- Storage Data

### ISO-NE - Currently Downloading (2 datasets)
1. ✅ LMP Day-Ahead Hourly (~30,600 rows/day - LARGE!)
2. ✅ Fuel Mix (~120-180 rows/day)

**Failing:**
- ⚠️ LMP Real-Time 5-Min (gridstatus NoneType bug)
- ⚠️ Load (no files created)

**Missing but Available:**
- Load Forecast
- Wind Forecast
- Solar Forecast
- BTM Solar
- Storage Data

### CAISO - Currently Downloading (5+ datasets)
1. ✅ LMP Day-Ahead Hourly (~390,936 rows/day - VERY LARGE!)
2. ✅ LMP Real-Time 5-Min (large)
3. ✅ LMP Real-Time 15-Min (~71,368 rows/day)
4. ✅ Load (~288 rows/day)
5. ✅ Fuel Mix (~288 rows/day)

**Failing:**
- ⚠️ AS Prices (both DA & RT)

**Missing but Available:**
- AS Procurement
- Renewables Hourly
- Renewables Forecasts (DA, HASP, RTD, RTPD)
- Curtailment
- Generation Outages
- Tie Flows
- Storage Data
- All Load Forecasts (5-min, 15-min, DA, 2-day, 7-day)

## Data Volume Estimates (2019-2025, ~2,475 days)

| ISO | Currently Downloading | Estimated Size | Est. Rows |
|-----|----------------------|----------------|-----------|
| **NYISO** | 5 datasets | ~2.1 GB | ~21M rows |
| **ISO-NE** | 2 datasets | ~10-15 GB | ~75M rows |
| **CAISO** | 5 datasets | ~25-30 GB | ~1.2B rows |
| **TOTAL** | | **~40-45 GB** | **~1.3B rows** |

## Recommended Additions (High Value)

### Priority 1 - Missing Critical Data
1. **NYISO Fuel Mix** - Easy to add, completes generation picture
2. **CAISO AS Prices** - Need to fix gridstatus call
3. **ISO-NE LMP RT** - Need to fix gridstatus bug or skip
4. **ISO-NE Load** - Need to debug why files not creating

### Priority 2 - Valuable Context
5. **All Load Forecasts** - Useful for ML models
6. **Wind/Solar Forecasts** - Renewable forecasting
7. **Storage Data (all ISOs)** - Battery analysis
8. **CAISO Renewables Hourly** - Actual renewable generation

### Priority 3 - Advanced Analytics
9. **CAISO Curtailment** - Grid constraints analysis
10. **NYISO Interface Flows** - Transmission analysis
11. **BTM Solar (all ISOs)** - Behind-meter solar
12. **CAISO Tie Flows** - Inter-ISO exchange

## Next Steps

### Immediate Actions
1. **Add NYISO Fuel Mix** to downloader (1 line of code)
2. **Debug ISO-NE Load** issue (check method signature)
3. **Fix CAISO AS Prices** call (check gridstatus AS method)

### Enhancement Actions
4. Add load forecast downloads to all ISOs
5. Add wind/solar forecast downloads
6. Add storage data downloads
7. Add renewables hourly for CAISO

### Research Required
- Available reserves/margin: May need direct ISO API calls
- Generation outages: May need scraping ISO websites
- Some data may require authentication/credentials

## Current Download Progress (as of 2025-10-11)

- **NYISO**: Day ~40/2,475 (1.6%) - ETA: 2 hours
- **ISO-NE**: Day ~50/2,475 (2.0%) - ETA: 7-10 hours
- **CAISO**: Day ~1/2,475 (0.04%) - ETA: 15-20 hours

**All 3 ISOs downloading in parallel!**

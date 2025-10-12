# ISO Remaining Datasets - Detailed Todo List

**Date:** 2025-10-11
**Current Status:** NYISO, ISO-NE, CAISO downloading from 2019-01-01

## Priority 1: Fix Currently Failing Downloads (Critical)

### 1.1 ISO-NE Real-Time LMP ⚠️ HIGH PRIORITY
**Status:** Failing with gridstatus NoneType error
**Impact:** Missing 350K rows/day of RT pricing data
**Action Required:**
- [ ] Debug gridstatus ISONE.get_lmp() for RT market
- [ ] Check gridstatus GitHub issues for known bugs
- [ ] If gridstatus broken, implement direct ISO-NE API call
- [ ] API: https://www.iso-ne.com/isoexpress/web/reports/pricing
- [ ] Estimated Effort: 2-4 hours

### 1.2 ISO-NE Load Data ⚠️ MEDIUM PRIORITY
**Status:** Method runs but no files created
**Impact:** Missing actual load data
**Action Required:**
- [ ] Check ISONE.get_load() method signature
- [ ] Verify date parameter format
- [ ] Test with recent dates (gridstatus may not have 2019 data)
- [ ] Implement direct API call if needed
- [ ] Estimated Effort: 1-2 hours

### 1.3 CAISO Ancillary Services ⚠️ HIGH PRIORITY
**Status:** get_as_prices() method call failing
**Impact:** Missing all AS price data (Reg Up/Down, Spin, Non-Spin)
**Action Required:**
- [ ] Check CAISO.get_as_prices() method signature in gridstatus
- [ ] Fix market parameter (currently passing market_run_id which is wrong)
- [ ] Test with correct parameters
- [ ] API: http://oasis.caiso.com/oasisapi/ (PRC_AS dataset)
- [ ] Estimated Effort: 1-2 hours

## Priority 2: Add Easy High-Value Datasets (Quick Wins)

### 2.1 NYISO Fuel Mix ✅ EASY
**Status:** Available in gridstatus, not downloading
**Impact:** Missing generation mix data
**Action Required:**
- [ ] Add `self.download_fuel_mix_day(current_date)` to download_date_range()
- [ ] Test with 1 day
- [ ] Restart NYISO download with --auto-resume
- [ ] Estimated Effort: 15 minutes

### 2.2 Load Forecasts (All ISOs) ✅ EASY
**Status:** Available in gridstatus for all 3 ISOs
**Methods Available:**
- NYISO: `get_load_forecast()`, `get_zonal_load_forecast()`
- ISONE: `get_load_forecast()`
- CAISO: `get_load_forecast_day_ahead()`, `get_load_forecast_two_day_ahead()`, `get_load_forecast_seven_day_ahead()`

**Action Required:**
- [ ] Add load forecast methods to each downloader
- [ ] Create separate data type directories (load_forecast_da, load_forecast_2day, etc.)
- [ ] Test with 1-2 days
- [ ] Estimated Effort: 1 hour per ISO (3 hours total)

### 2.3 Wind & Solar Forecasts ✅ MEDIUM
**Status:** Available for ISO-NE and CAISO
**Methods:**
- ISONE: `get_wind_forecast()`, `get_solar_forecast()`
- CAISO: `get_renewables_forecast_dam()`, `get_renewables_forecast_hasp()`, `get_renewables_forecast_rtd()`, `get_renewables_forecast_rtpd()`
- NYISO: `get_btm_solar_forecast()` only

**Action Required:**
- [ ] Add forecast methods to ISONE downloader
- [ ] Add renewables forecast methods to CAISO downloader
- [ ] Add BTM solar forecast to NYISO downloader
- [ ] Estimated Effort: 2 hours

### 2.4 Battery/Storage Data (All ISOs) ✅ MEDIUM
**Status:** Available via `get_storage()` for all 3 ISOs
**Impact:** Battery charge/discharge state data
**Action Required:**
- [ ] Add `get_storage()` method to all 3 downloaders
- [ ] Investigate data structure and row counts
- [ ] Test with recent dates first (may not have 2019 data)
- [ ] Estimated Effort: 2 hours

## Priority 3: Add Advanced CAISO Datasets

### 3.1 CAISO Actual Renewables Production 🔶 HIGH VALUE
**Status:** Available in gridstatus
**Methods:**
- `get_renewables_hourly()` - Actual wind/solar production
- `get_fuel_regions()` - Regional fuel mix
- `get_caiso_renewables_report()` - Detailed renewables

**Action Required:**
- [ ] Add renewables_hourly download
- [ ] Test data structure and volume
- [ ] Estimated Effort: 1 hour

### 3.2 CAISO Curtailment Data 🔶 HIGH VALUE
**Status:** Available in gridstatus
**Methods:**
- `get_curtailment()` - Current curtailment API
- `get_curtailment_legacy()` - Historical curtailment
- `get_curtailed_non_operational_generator_report()` - Generator outages

**Action Required:**
- [ ] Add curtailment downloads
- [ ] Important for grid constraint analysis
- [ ] Estimated Effort: 2 hours

### 3.3 CAISO Tie Flows 🔶 MEDIUM VALUE
**Status:** Available in gridstatus
**Methods:**
- `get_tie_flows_real_time()` - RT 5-min tie flows
- `get_tie_flows_real_time_15_min()` - RT 15-min tie flows

**Action Required:**
- [ ] Add tie flow downloads
- [ ] Useful for inter-ISO analysis
- [ ] Estimated Effort: 1 hour

### 3.4 CAISO System Schedules 🔶 MEDIUM VALUE
**Status:** Available in gridstatus
**Methods:**
- `get_system_load_and_resource_schedules_day_ahead()`
- `get_system_load_and_resource_schedules_hasp()`
- `get_system_load_and_resource_schedules_real_time_5_min()`
- `get_system_load_and_resource_schedules_ruc()`

**Action Required:**
- [ ] Add system schedule downloads
- [ ] Large datasets - test volume first
- [ ] Estimated Effort: 3 hours

## Priority 4: SPP & IESO Fixes (If Needed)

### 4.1 SPP Historical Data 🔴 BLOCKED
**Status:** No historical data available via gridstatus API
**Issue:** API returns 404 for 2019-2022 data
**Action Required:**
- [ ] Research SPP's historical data availability
- [ ] Check if data exists on SPP website/FTP
- [ ] Determine earliest available date via gridstatus
- [ ] If needed, implement direct SPP Marketplace API calls
- [ ] SPP Portal: https://portal.spp.org/
- [ ] Estimated Effort: 4-6 hours (requires research)

### 4.2 IESO Data Access 🔴 BLOCKED
**Status:** get_lmp methods return 404 errors
**Issue:** Gridstatus may not have historical IESO data
**Action Required:**
- [ ] Test with recent dates (2024) to see if API works
- [ ] Check IESO website for direct API access
- [ ] IESO Reports: http://reports.ieso.ca/
- [ ] May need to implement custom IESO scraper
- [ ] Estimated Effort: 4-6 hours

## Priority 5: Data NOT in Gridstatus (Requires Direct API)

### 5.1 Available Reserves / Reserve Margin ❌ NOT AVAILABLE
**Description:** Real-time available operating reserves
**ISOs:** NYISO, ISONE, CAISO
**Why Important:** Critical for reliability analysis
**Action Required:**
- [ ] Research NYISO Operations Report API
- [ ] Research ISO-NE System Conditions API
- [ ] Research CAISO Today's Outlook API
- [ ] Estimated Effort: 8-12 hours (3-4 hours per ISO)

### 5.2 Generation Outages (except CAISO) ❌ NOT AVAILABLE
**Description:** Planned and forced generator outages
**ISOs:** NYISO, ISONE (CAISO has curtailment)
**Action Required:**
- [ ] Check if ISOs publish outage data via API
- [ ] May require scraping ISO websites
- [ ] NYISO: Check OASIS for outage reports
- [ ] ISONE: Check System Information reports
- [ ] Estimated Effort: 6-8 hours per ISO

### 5.3 Transmission Outages & Constraints ❌ NOT AVAILABLE
**Description:** Transmission line outages, flowgates, constraints
**Note:** Partial data available (NYISO interface flows)
**Action Required:**
- [ ] NYISO: Already have interface limits/flows (need to add)
- [ ] ISONE: Research transmission data availability
- [ ] CAISO: Have shadow prices (constraint-related)
- [ ] Estimated Effort: 4-6 hours per ISO

### 5.4 Reserve Procurement / Shortage Events ❌ LIMITED
**Description:** AS procurement volumes, shortage pricing events
**CAISO:** Has `get_as_procurement()` method
**Action Required:**
- [ ] Add CAISO AS procurement to downloader
- [ ] Research NYISO reserve shortage data
- [ ] Research ISONE reserve adequacy data
- [ ] Estimated Effort: 4 hours

## Implementation Order & Timeline

### Week 1: Fix Failures & Add Quick Wins (12-16 hours)
1. ✅ Fix CAISO AS prices (2 hours)
2. ✅ Debug ISO-NE RT LMP (3 hours)
3. ✅ Fix ISO-NE Load (1 hour)
4. ✅ Add NYISO Fuel Mix (15 min)
5. ✅ Add Load Forecasts (all ISOs) (3 hours)
6. ✅ Add Wind/Solar Forecasts (2 hours)
7. ✅ Add Storage Data (all ISOs) (2 hours)

### Week 2: CAISO Advanced Features (8-10 hours)
1. ✅ CAISO Renewables Hourly (1 hour)
2. ✅ CAISO Curtailment (2 hours)
3. ✅ CAISO Tie Flows (1 hour)
4. ✅ CAISO System Schedules (3 hours)
5. ✅ CAISO AS Procurement (1 hour)

### Week 3: Direct API Implementation (12-16 hours)
1. 🔶 Research & implement available reserves (per ISO)
2. 🔶 Implement NYISO interface flows download
3. 🔶 Research generation outage data sources
4. 🔶 Test and validate all new data downloads

### Week 4: SPP/IESO & Edge Cases (8-12 hours)
1. 🔴 Research SPP historical data availability
2. 🔴 Test IESO with recent dates, implement if possible
3. 🔶 Add any remaining high-value datasets identified

## Data Coverage After Completion

### Currently Downloading (12 datasets)
- **Energy:** LMP DA + RT (3 ISOs) = 6 datasets
- **Ancillary Services:** NYISO AS DA + RT = 2 datasets
- **Load:** Total load (3 ISOs) = 3 datasets
- **Generation:** Fuel mix (CAISO, ISONE) = 2 datasets
- **Missing:** ISO-NE RT LMP, ISO-NE Load, CAISO AS

### After Week 1 Fixes (~25 datasets)
- All current downloads fixed
- Load forecasts (3 ISOs × 2-3 types) = 6-9 datasets
- Wind/Solar forecasts (2 ISOs × 2-4 types) = 6-8 datasets
- Storage data (3 ISOs) = 3 datasets

### After Week 2 Additions (~35 datasets)
- CAISO renewables actual = 2 datasets
- CAISO curtailment = 2 datasets
- CAISO tie flows = 2 datasets
- CAISO system schedules = 4 datasets
- CAISO AS procurement = 1 dataset

### Full Implementation (~45-50 datasets)
- Available reserves (3 ISOs) = 3 datasets
- Generation outages (limited) = 2-3 datasets
- Transmission data (NYISO) = 2 datasets
- Reserve adequacy (partial) = 2 datasets

## Storage & Performance Estimates

### Current Downloads (2019-2025)
- **Size:** ~40-45 GB
- **Rows:** ~1.3B rows
- **Time:** 15-20 hours

### After All Priority 1-3 Additions
- **Size:** ~80-100 GB (2x current)
- **Rows:** ~2.5-3B rows
- **Time:** 25-35 hours for historical backfill

### Incremental Daily Updates (After Initial Download)
- **Current:** ~5-10 MB/day
- **With All Datasets:** ~15-25 MB/day
- **Cron Runtime:** 5-15 minutes/day

## Success Metrics

### Must-Have (Priority 1-2)
- ✅ Fix all failing downloads (ISO-NE RT, CAISO AS)
- ✅ Add load forecasts (all ISOs)
- ✅ Add wind/solar forecasts
- ✅ Add NYISO fuel mix

### Should-Have (Priority 3)
- ✅ CAISO renewables actual
- ✅ CAISO curtailment
- ✅ Storage data (all ISOs)

### Nice-to-Have (Priority 4-5)
- 🔶 SPP historical (if available)
- 🔶 IESO (if fixable)
- 🔶 Available reserves
- 🔶 Generation outages

## Next Immediate Steps

1. **Fix CAISO AS Prices** (30 minutes)
   - Check gridstatus source code for correct method call
   - Test with single date
   - Add to downloader

2. **Add NYISO Fuel Mix** (15 minutes)
   - One line of code
   - Immediate value

3. **Debug ISO-NE Issues** (2-3 hours)
   - RT LMP gridstatus bug
   - Load data not saving

4. **Test & Validate** (1 hour)
   - Verify all fixes work
   - Check file output
   - Monitor downloads

**Ready to start implementation!**

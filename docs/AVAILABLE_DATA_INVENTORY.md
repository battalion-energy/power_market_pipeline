# ERCOT Data Inventory for ML Training
**Generated:** October 11, 2025
**Purpose:** Battery auto-bidding price forecast models

---

## ✅ Data We Have (2010-2025)

### 1. Market Prices (Rollup Files - Ready to Use)

**Real-Time Prices (15-min resolution):**
- Location: `/rollup_files/flattened/RT_prices_15min_YYYY.parquet`
- Years: 2010-2025 (16 years)
- Hubs: HB_HOUSTON, HB_NORTH, HB_SOUTH, HB_WEST, HB_PAN, HB_BUSAVG
- Total spikes >$400: 2,187 (2019-2025)
- Winter Storm Uri: 813 spikes in 2021 alone

**Day-Ahead Prices (hourly resolution):**
- Location: `/rollup_files/flattened/DA_prices_YYYY.parquet`
- Years: 2010-2025
- Same hubs as RT prices
- Use for: DAM vs RT spread prediction

**Ancillary Service Prices (hourly):**
- Location: `/rollup_files/flattened/AS_prices_YYYY.parquet`
- Years: 2010-2025
- Services:
  - **REGUP** - Regulation Up
  - **REGDN** - Regulation Down
  - **RRS** - Responsive Reserve Service (primary freq response)
  - **NSPIN** - Non-Spinning Reserves
  - **ECRS** - ERCOT Contingency Reserve Service
- Use for: AS vs RT arbitrage decisions

### 2. Weather Data (NASA POWER Satellite)

**Location:** `/pool/ssd8tb/data/weather_data/parquet_by_iso/ERCOT_weather_data.parquet`
**Coverage:** 2019-01-01 to 2025-10-11 (6.8 years)
**Locations:** 55 sites
- 10 major cities (Houston, Dallas, Austin, San Antonio, etc.)
- 30 largest wind farms
- 15 largest solar farms

**Variables:**
- Temperature: T2M, T2M_MAX, T2M_MIN (2m above ground)
- Wind: WS50M, WD50M (50m hub height)
- Solar: ALLSKY_SFC_SW_DWN, CLRSKY_SFC_SW_DWN (irradiance)
- Humidity: RH2M
- Precipitation: PRECTOTCORR

**Key Features Created:**
- Heat waves (temp > 35°C)
- Cold snaps (temp < 0°C)
- Cooling/Heating degree days
- System-wide wind speed averages
- System-wide solar irradiance
- Cloud cover indicators

### 3. Wind/Solar Forecasts & Actuals

**Wind Power Production:**
- Location: `/Wind_Power_Production/*.csv`
- Years: 2024-2025 (earlier years being downloaded)
- Resolution: Hourly
- System-wide + Regional breakouts
- Columns:
  - Actual generation (GEN)
  - STWPF forecast (Short-Term Wind Power Forecast)
  - Forecast errors calculated

**Solar Power Production:**
- Location: `/Solar_Power_Production/*.csv`
- Years: 2024-2025 (earlier years being downloaded)
- Resolution: Hourly
- System-wide + Regional breakouts
- Columns:
  - Actual generation
  - STPPF forecast (Short-Term Photovoltaic Power Forecast)
  - Forecast errors calculated

### 4. Battery Trading Results (Historical Performance)

**Location:** `/tbx_results/` and `/tbx_results_all_nodes/`

**TBX Analysis Files (Hub-level):**
- Daily results: `tbx_daily_YYYY.parquet`
- Monthly results: `tbx_monthly_YYYY.parquet`
- Annual results: `tbx_annual_YYYY.parquet`
- Comprehensive: `tbx_comprehensive_2024.parquet`
- Leaderboard: `tbx_leaderboard.parquet`

**All-Nodes Analysis:**
- Same structure but for ALL settlement point nodes
- Years: 2021-2025

**Use Case:** Learn from actual battery performance and trading patterns

### 5. BESS Dispatch Data (Individual Batteries)

**Location:** `/bess_analysis/hourly/dispatch/`
- Individual battery dispatch files: `{BATTERY_NAME}_{YEAR}_dispatch.parquet`
- Years: 2021-2025
- 50+ different batteries tracked
- Includes: Charge/discharge patterns, SOC estimates, actual operations

**Example Batteries:**
- BLUEJAY_BESS1, GAMBIT_BESS1, BRAZORIA_UNIT1
- SWEENY_UNIT1, LON_BES1, JAR_BES1
- etc.

---

## ⏳ Data Being Downloaded (In Progress)

### Wind/Solar Forecasts (Historical)
- Target: 2019-2023 historical forecasts
- Status: Download in progress
- ETA: User is working on it

### Load Forecasts
- ERCOT publishes hourly load forecasts
- Multiple forecast types: STF (short-term), MTF (mid-term), LTF (long-term)
- Status: To be added

### ORDC Reserve Data
- Operating Reserve Demand Curve metrics
- Reserve margins, trigger prices
- Online/offline reserves
- Status: Available in raw ERCOT data, needs processing

---

## 📊 Data Summary by Availability

| Data Type | Years Available | Resolution | Records | Use Case |
|-----------|----------------|------------|---------|----------|
| **RT Prices** | 2010-2025 | 15-min | ~550K/year | Primary target |
| **DA Prices** | 2010-2025 | Hourly | ~8.8K/year | DAM vs RT spread |
| **AS Prices** | 2010-2025 | Hourly | ~8.8K/year | AS vs RT decision |
| **Weather** | 2019-2025 | Daily | 2,476 days × 55 locs | Temp extremes, demand |
| **Wind Forecasts** | 2024-2025* | Hourly | 16K records | Forecast errors |
| **Solar Forecasts** | 2024-2025* | Hourly | 16K records | Forecast errors |
| **BESS Dispatch** | 2021-2025 | Hourly | Varies | Learn from actual |
| **Load Forecasts** | TBD | Hourly | TBD | Demand prediction |

*Historical 2019-2023 being added

---

## 🎯 Data Strategy for ML Model

### Phase 1: Baseline Model (Can Build Now)
**Use:**
- RT prices (2019-2025) - 2,187 spikes >$400
- DA prices (2019-2025) - DAM position
- AS prices (2019-2025) - AS opportunity cost
- Weather (2019-2025) - Temperature extremes, wind/solar potential
- Temporal features - Hour, day, season, year

**Missing but acceptable:**
- Wind/solar forecast errors (only 2024-2025)
- Load forecasts (can use actual load as proxy)

### Phase 2: Enhanced Model (When Historical Forecasts Ready)
**Add:**
- Wind/solar forecast errors (2019-2025)
- Load forecast errors
- ORDC reserve metrics

### Phase 3: Production Model
**Add:**
- Real-time ORDC data
- Transmission constraints
- Generator outages

---

## 💡 Key Insights from Data

### Market Evolution (2019-2025)
- **2021**: 813 spikes >$400 (2.42% of hours) - Winter Storm Uri
- **2024-2025**: 103 spikes (0.19%) - 10x calmer market
- **Renewable Growth**: Low-price hours increased from 52% (2019) to 75% (2020)
- **Volatility Shift**: Need time-aware features to handle regime changes

### Price Thresholds for Battery Trading
**Charge Opportunities (<$20/MWh):**
- 2019: 52% of hours
- 2020: 74% (COVID, low demand)
- 2021: 29% (extreme scarcity)
- 2024: 50% (new normal)

**Discharge Opportunities (>$400/MWh):**
- Average: 0.94% of hours (2019-2025)
- Range: 0.10% (2025) to 2.42% (2021)
- Avg spread when arbitrage available: $1,198/MWh

### Ancillary Services
**Typical AS Prices (2024):**
- REGUP: $1-5/MW (regulation up)
- REGDN: $1-3/MW (regulation down)
- RRS: $1-2/MW (responsive reserves)
- NSPIN: $0.50-2/MW (non-spin)
- ECRS: $0.50-1.5/MW (contingency)

**Strategy:** Compare AS revenue vs holding for RT spike

---

## 🔧 Data Processing Status

### ✅ Ready to Use
- All price data (RT, DA, AS) 2010-2025
- Weather data 2019-2025
- BESS historical dispatch
- Feature engineering pipelines

### 🔄 In Progress
- Wind/solar forecasts 2019-2023
- Load forecasts integration

### ⏰ Planned
- ORDC reserve metrics
- Transmission constraints
- Real-time data feeds

---

## 📈 Recommended Next Steps

1. **Build multi-horizon spike classifier using available data:**
   - RT prices (2019-2025)
   - Weather (2019-2025)
   - AS prices (as opportunity cost proxy)
   - DA-RT spread (market stress indicator)

2. **Add forecast errors when historical data ready:**
   - Will significantly improve performance
   - Current model can be baseline

3. **Incorporate time-aware features:**
   - Year, quarter (to handle market evolution)
   - Post-Winter-Storm indicator
   - Renewable penetration proxy

4. **Ensemble across market regimes:**
   - High volatility model (2019-2021)
   - Transition model (2022-2023)
   - Current regime model (2024-2025)

---

**Bottom Line:** We have 16 years of price data, 6.8 years of weather, and can build a strong baseline multi-horizon model NOW. Enhanced version with forecast errors can be added when historical data is ready.

**Total Training Examples:**
- High prices >$400: 2,187 events (2019-2025)
- vs. Current model: 103 events (2024-2025 only)
- **21x more spike training data available!**

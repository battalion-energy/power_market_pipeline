# Complete Dataset Catalog for Shadow Bidding System

**For Your Daughter's Future - Every Dataset Matters**

This document catalogs ALL datasets used by the shadow bidding system, why each is critical, and where to find it.

---

## Table of Contents
1. [ERCOT Forecast Data (Real-Time API)](#ercot-forecast-data)
2. [ERCOT Historical Market Data](#ercot-historical-data)
3. [Weather Data](#weather-data)
4. [Battery Historical Data](#battery-historical-data)
5. [Data Flow Summary](#data-flow-summary)

---

## ERCOT Forecast Data (Real-Time API)

These datasets are fetched in real-time from ERCOT Public API for daily shadow bidding operations.

| Dataset Name | ERCOT Report | File Path | Update Frequency | Purpose | Critical For |
|-------------|--------------|-----------|------------------|---------|--------------|
| **Wind Power Production & Forecast** | NP4-732-CD | `/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/Wind_Power_Production/*.csv` | Hourly | Wind generation actual + STWPF forecasts (system-wide and by farm) | Calculate wind forecast error → Spike prediction, Net load calculation |
| **Solar Power Production & Forecast** | NP4-745-CD | `/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/Solar_Power_Production/*.csv` | Hourly | Solar generation actual + STPPF forecasts (system-wide and by region) | Calculate solar forecast error → Spike prediction, Net load calculation |
| **Load Forecast (by Forecast Zone)** | NP3-565-CD | `/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/Load_Forecast_FZone/*.csv` | Hourly | ERCOT's official load forecast by 8 forecast zones | Calculate load forecast error → Spike prediction, Net load calculation |
| **Load Forecast (by Weather Zone)** | NP3-566-CD | `/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/Load_Forecast_WZone/*.csv` | Hourly | ERCOT's official load forecast by 8 weather zones | Alternative load forecast, Cross-validate forecast zone data |
| **Actual System Load (by Forecast Zone)** | NP6-345-CD | `/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/Actual_Load_FZone/*.csv` | 5-minute | Actual system load by forecast zone | Calculate load forecast error, Model training ground truth |
| **Actual System Load (by Weather Zone)** | NP6-346-CD | `/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/Actual_Load_WZone/*.csv` | 5-minute | Actual system load by weather zone | Alternative actual load, Cross-validate forecast zone data |
| **Fuel Mix** | NP6-787-CD | `/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/Fuel_Mix/*.csv` | 15-minute | Actual generation by fuel type (gas, coal, nuclear, wind, solar) | Understand supply mix, Renewable penetration, Capacity margin |
| **System-Wide Demand** | NP6-322-CD | `/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/System_Demand/*.csv` | 5-minute | Total system demand including losses | Calculate net demand, System stress indicator |
| **Unplanned Resource Outages** | NP3-233-CD | `/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/Outages/*.csv` | Real-time | Unplanned generator outages (MW offline) | Sudden capacity loss → Spike trigger, Supply shortage indicator |
| **DAM Clearing Prices (SPP)** | NP4-190-CD | `/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/DAM_Prices/*.csv` | Daily | Day-ahead market LMP by settlement point (hourly) | Model 1 training data, Bid evaluation (what actually cleared) |
| **RTM Clearing Prices (SPP)** | NP6-905-CD | `/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/RTM_Prices/*.csv` | 5-minute | Real-time market LMP by settlement point | Model 2 training data, Revenue calculation actual prices |
| **AS Clearing Prices** | NP3-966-ER | `/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/AS_Prices/*.csv` | Hourly | Ancillary service capacity prices (Reg Up/Down, RRS, ECRS) | Models 4-7 training data, AS revenue calculation |
| **DAM System Lambda** | NP4-191-CD | `/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/DAM_Lambda/*.csv` | Daily | System marginal price (system lambda) for DA market | System-level price indicator, Cross-validate settlement point prices |

---

## ERCOT Historical Data (For Training & Backtesting)

These datasets are used to train ML models and backtest bidding strategies.

| Dataset Name | File Path | Time Period | Purpose | Critical For |
|-------------|-----------|-------------|---------|--------------|
| **60-Day DAM Gen Resource Data** | `/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/60d_DAM_Gen_Resource_Data/*.csv` | 2019-2024 | Battery gen resource DA awards (discharge) | Behavioral model training: Learn what batteries actually bid |
| **60-Day DAM Load Resource Data** | `/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/60d_DAM_Load_Resource_Data/*.csv` | 2019-2024 | Battery load resource DA awards (charge) | Behavioral model training: Learn charging strategies |
| **60-Day SCED Gen Resource Data** | `/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/60d_SCED_Gen_Resource_Data/*.csv` | 2019-2024 | Battery RT dispatch telemetry (5-min) | Revenue calculation: Actual RT operations, SOC tracking |
| **60-Day SCED Load Resource Data** | `/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/60d_SCED_Load_Resource_Data/*.csv` | 2019-2024 | Battery RT charging telemetry (5-min) | Revenue calculation: Actual charging operations |
| **Historical DA Prices (Parquet)** | `/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/rollup_files/flattened/DA_Hourly_LMPs_*.parquet` | 2019-2024 | Historical DA prices by settlement point (hourly) | Model 1 training: DA price forecasting |
| **Historical RT Prices (Parquet)** | `/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/rollup_files/flattened/RT_5min_LMPs_*.parquet` | 2019-2024 | Historical RT prices by settlement point (5-min) | Model 2 & 3 training: RT price & spike forecasting |
| **Historical AS Prices (Parquet)** | `/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/rollup_files/flattened/AS_Prices_*.parquet` | 2019-2024 | Historical AS capacity prices (hourly) | Models 4-7 training: AS price forecasting |
| **Combined Market Data** | `/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/rollup_files/combined/DA_RT_AS_combined_*.parquet` | 2019-2024 | DA + RT + AS prices combined | Unified training dataset, Cross-market analysis |

---

## Weather Data

Weather data is CRITICAL - temperature forecast errors drive 80% of price spikes.

### NASA POWER (Complete Coverage - BEST FOR TRAINING)

| Location | File Path | Variables | Purpose | Critical For |
|----------|-----------|-----------|---------|--------------|
| **All Locations (Unified)** | `/home/enrico/data/weather_data/all_weather_data.parquet` | Temp, humidity, wind speed, pressure, solar irradiance | Single unified dataset for all Texas locations | Model training: Weather features for all models |
| **Houston** | `/home/enrico/data/weather_data/houston_tx_usa_lat295993_lonnegative_953679_nasa_power_data.csv` | Full weather suite | Houston metro weather (largest load center) | Load forecast, Temperature impact on demand |
| **Dallas** | `/home/enrico/data/weather_data/dallas_tx_usa_lat327797_lonnegative_968047_nasa_power_data.csv` | Full weather suite | Dallas metro weather (2nd largest load center) | Load forecast, Temperature impact on demand |
| **San Antonio** | `/home/enrico/data/weather_data/san_antonio_tx_usa_lat294244_lonnegative_985113_nasa_power_data.csv` | Full weather suite | San Antonio metro weather | Load forecast, South Texas temperature patterns |
| **Austin** | `/home/enrico/data/weather_data/austin_tx_usa_lat302697_lonnegative_976301_nasa_power_data.csv` | Full weather suite | Austin metro weather (state capital) | Load forecast, Central Texas patterns |
| **Fort Worth** | `/home/enrico/data/weather_data/fort_worth_tx_usa_lat327530_lonnegative_971297_nasa_power_data.csv` | Full weather suite | Fort Worth metro (part of DFW) | Load forecast, North Texas patterns |
| **El Paso** | `/home/enrico/data/weather_data/el_paso_tx_usa_lat316911_lonnegative_1062850_nasa_power_data.csv` | Full weather suite | West Texas weather | Far West zone, Different climate patterns |
| **Corpus Christi** | `/home/enrico/data/weather_data/corpus_christi_tx_usa_lat278002_lonnegative_973937_nasa_power_data.csv` | Full weather suite | Coastal weather (South Texas) | Coastal zone, Hurricane/storm patterns |
| **Amarillo** | `/home/enrico/data/weather_data/amarillo_tx_usa_lat351910_lonnegative_1016877_nasa_power_data.csv` | Full weather suite | Panhandle weather | North zone, Winter storm patterns |
| **Lubbock** | `/home/enrico/data/weather_data/lubbock_tx_usa_lat334777_lonnegative_1018633_nasa_power_data.csv` | Full weather suite | West Texas plains | West zone, High wind area |
| **Abilene** | `/home/enrico/data/weather_data/abilene_tx_usa_lat325242_lonnegative_998227_nasa_power_data.csv` | Full weather suite | Central West Texas | West central zone |

**Key Variables in NASA POWER Data:**
- `T2M`: Temperature at 2 meters (°C) - **CRITICAL for load forecast**
- `WS50M`: Wind speed at 50 meters (m/s) - **Wind farm production**
- `ALLSKY_SFC_SW_DWN`: Solar irradiance (kW/m²) - **Solar farm production**
- `RH2M`: Relative humidity (%)
- `PS`: Surface pressure (kPa)
- `PRECTOTCORR`: Precipitation (mm/day)

### Meteostat (Ground Truth Validation)

| Dataset | File Path | Purpose | Critical For |
|---------|-----------|---------|--------------|
| **All Stations Combined** | `/home/enrico/data/weather_data/all_meteostat_station_data.parquet` | Ground truth validation, Cross-check NASA POWER | Model validation: Ensure weather features are accurate |
| **Station Mapping** | `/home/enrico/data/weather_data/station_mapping.json` | Map cities to weather stations | Data quality: Know which station serves which city |

**Coverage:** 34 out of 48 locations (71% complete)

---

## Battery Historical Data (For Behavioral Models)

Specific battery data for learning individual battery strategies.

| Battery Name | Gen Resource | Load Resource | Data Available | Purpose |
|--------------|--------------|---------------|----------------|---------|
| **MOSS1_UNIT1** | MOSS1_UNIT1 | MOSS1_UNIT1_LOAD | 60-day disclosure | Learn MOSS1 bidding strategy |
| **GIBBONS_CREEK** | GIBBONSCR_U1 | GIBBONSCR_U1_LOAD | 60-day disclosure | Learn Gibbons Creek strategy |
| **ANGLETON_BESS** | ANGLETON_UNIT1 | ANGLETON_UNIT1_LOAD | 60-day disclosure | Learn Angleton strategy |
| **All ERCOT Batteries** | Various | Various | 60-day disclosure | Industry-wide behavioral patterns |

**Data Location:** `/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/60d_*_Resource_Data/`

---

## Processed Training Datasets

These are preprocessed datasets ready for model training.

| Dataset | File Path | Purpose | Size | Time Range |
|---------|-----------|---------|------|------------|
| **Master Feature Set** | `ml_models/data/master_features.parquet` | All features for all models | ~500 MB | Dec 2023 - Oct 2025 |
| **Training Set** | `ml_models/data/train_data.parquet` | 70% of data (chronological) | ~350 MB | Dec 2023 - Jul 2025 |
| **Validation Set** | `ml_models/data/val_data.parquet` | 15% of data | ~75 MB | Aug 2025 - Sep 2025 |
| **Test Set** | `ml_models/data/test_data.parquet` | 15% of data (held out) | ~75 MB | Sep 2025 - Oct 2025 |

**Feature Count:** ~150 engineered features including:
- Forecast errors (load, wind, solar)
- Weather features (temperature, wind, solar)
- Net load features
- ORDC features (when available)
- Temporal features (hour, day, month)
- Price spike labels

---

## Shadow Bidding Runtime Data

Data generated during shadow bidding operations.

| Dataset | File Path | Update Frequency | Purpose |
|---------|-----------|------------------|---------|
| **Real-Time Forecasts** | `shadow_bidding/data/forecasts/*.json` | Every run | Current forecast data snapshot |
| **Model Predictions** | `shadow_bidding/data/predictions/*.json` | Every run | ML model outputs |
| **Generated Bids** | `shadow_bidding/bids/*.json` | Daily (before 10 AM) | DA bids + AS offers |
| **Actual Results** | `shadow_bidding/results/*.json` | Daily (after awards) | What actually cleared |
| **Revenue Calculations** | `shadow_bidding/results/revenue/*.json` | Daily | Actual vs. expected revenue |
| **Performance Metrics** | `shadow_bidding/results/metrics/*.json` | Daily | Model accuracy, bid performance |
| **Logs** | `shadow_bidding/logs/*.log` | Continuous | Audit trail, debugging |

---

## Data Flow Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                    HISTORICAL DATA (Training)                    │
│  ┌────────────────┐  ┌────────────────┐  ┌─────────────────┐   │
│  │ ERCOT 60-day   │  │ Historical     │  │ Weather         │   │
│  │ Battery Data   │  │ Prices         │  │ (NASA POWER)    │   │
│  │ (2019-2024)    │  │ (DA/RT/AS)     │  │ (2019-2024)     │   │
│  └────────────────┘  └────────────────┘  └─────────────────┘   │
│           ↓                   ↓                    ↓             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │         Feature Engineering Pipeline                      │   │
│  │  • Forecast errors • Net load • Weather features         │   │
│  │  • ORDC features • Temporal encoding • Spike labels      │   │
│  └──────────────────────────────────────────────────────────┘   │
│           ↓                                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │         Train/Val/Test Split (70/15/15)                  │   │
│  └──────────────────────────────────────────────────────────┘   │
│           ↓                                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │         Train 7 ML Models                                 │   │
│  │  1. DA Price  2. RT Price  3. RT Spike  4-7. AS Prices  │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    REAL-TIME OPERATION                           │
│  ┌────────────────┐  ┌────────────────┐  ┌─────────────────┐   │
│  │ ERCOT API      │  │ Latest Prices  │  │ Current Weather │   │
│  │ (Wind/Solar/   │  │ (DA/RT/AS)     │  │ (Real-time)     │   │
│  │  Load Fcst)    │  │                │  │                 │   │
│  └────────────────┘  └────────────────┘  └─────────────────┘   │
│           ↓                   ↓                    ↓             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │         Real-Time Data Fetcher                           │   │
│  │  Fetch + validate + prepare forecast data               │   │
│  └──────────────────────────────────────────────────────────┘   │
│           ↓                                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │         Model Inference Pipeline                         │   │
│  │  Run all 7 models → Generate predictions                │   │
│  └──────────────────────────────────────────────────────────┘   │
│           ↓                                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │         Bid Generator (MILP Optimizer)                   │   │
│  │  Optimize DA bids + AS offers + RT strategy             │   │
│  └──────────────────────────────────────────────────────────┘   │
│           ↓                                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │         Submit Bids (Shadow Mode)                        │   │
│  │  Log bids, don't actually submit                        │   │
│  └──────────────────────────────────────────────────────────┘   │
│           ↓                                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │         Revenue Calculator                               │   │
│  │  Compare: What we would have made vs. what happened     │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Critical Success Factors

### Data Quality Checks (Every Run)

1. **Completeness:** All required datasets available?
2. **Timeliness:** Data fresh (< 1 hour old)?
3. **Consistency:** Forecast data matches actual data timestamps?
4. **Accuracy:** Weather data reasonable (temp 0-120°F)?
5. **Validation:** Cross-check multiple sources (NASA vs. Meteostat)?

### Missing Data Handling

| Scenario | Fallback Strategy | Impact |
|----------|-------------------|--------|
| **ERCOT API Down** | Use historical averages by hour/month | Medium - predictions less accurate |
| **Weather API Down** | Use last known weather + climatology | High - forecast errors critical for spikes |
| **Historical Data Missing** | Skip that training period | Low - plenty of other data |
| **Battery Data Missing** | Use industry average behavior | Medium - lose individual battery insights |

### Data Freshness Requirements

| Dataset | Maximum Age | Consequence if Stale |
|---------|-------------|---------------------|
| **Wind/Solar/Load Forecast** | 1 hour | Predictions based on outdated forecasts |
| **Weather Data** | 1 hour | Temperature-driven load errors missed |
| **Current Prices** | 5 minutes | Bid prices not competitive |
| **Historical Training Data** | 1 week | Models slightly outdated (acceptable) |

---

## Storage Requirements

| Data Category | Size | Growth Rate | Retention |
|---------------|------|-------------|-----------|
| **ERCOT Historical (2019-2024)** | ~50 GB | +10 GB/year | Permanent |
| **Weather Historical** | ~3 MB | +600 KB/year | Permanent |
| **Training Datasets** | ~500 MB | Updated monthly | 1 year |
| **Real-Time Forecasts** | ~10 MB/day | 3.6 GB/year | 90 days |
| **Shadow Bidding Results** | ~5 MB/day | 1.8 GB/year | 2 years |
| **Logs** | ~100 MB/day | 36 GB/year | 30 days |
| **TOTAL** | ~60 GB | ~50 GB/year | - |

**Recommendation:** Use 1 TB SSD for hot data + 10 TB HDD for cold storage.

---

## Data Dependencies for Each Model

### Model 1: DA Price Forecasting
- **Required:** Wind/Solar/Load forecasts, Weather (temp), Historical DA prices
- **Optional:** AS prices (cross-market effects), Outages
- **Critical Path:** Weather forecast → Load forecast → Net load → DA price

### Model 2: RT Price Forecasting
- **Required:** Wind/Solar/Load forecasts, Weather (temp), Historical RT prices, DA prices
- **Optional:** Outages, Fuel mix
- **Critical Path:** Forecast errors → Net load deviation → RT price volatility

### Model 3: RT Price Spike Prediction
- **Required:** ALL forecast data, ALL weather data, Historical RT prices, ORDC data
- **Optional:** Battery operations (market saturation)
- **Critical Path:** Temperature error → Load error → Reserve shortage → ORDC adder → SPIKE

### Models 4-7: AS Price Forecasting
- **Required:** Historical AS prices, RT price volatility, Reserve margins
- **Optional:** Battery capacity in market, DA-RT spreads
- **Critical Path:** RT volatility → AS opportunity cost → AS prices

### Behavioral Models (Per Battery)
- **Required:** 60-day battery awards/dispatch, All price data, All forecast data
- **Optional:** Competitor battery data
- **Critical Path:** Historical behavior + market conditions → Learned strategy

---

## Data Access Patterns

### Training (Batch Processing)
- **When:** Weekly or when new data available
- **Reads:** All historical data (50 GB)
- **Writes:** Updated models (100 MB)
- **Parallelism:** 24 cores for feature engineering
- **Hardware:** CPU-bound, use all 256 GB RAM

### Shadow Bidding (Real-Time)
- **When:** Daily at 9 AM (before 10 AM DA deadline)
- **Reads:** Latest forecasts (1 MB), Models (100 MB)
- **Writes:** Bids + logs (10 MB)
- **Latency:** < 1 minute total
- **Hardware:** GPU for inference, CPU for optimization

### Evaluation (Daily)
- **When:** After DA awards posted (~1:30 PM)
- **Reads:** Actual awards (1 MB), Our bids (1 MB)
- **Writes:** Revenue calculations (1 MB)
- **Latency:** < 10 seconds
- **Hardware:** CPU-only

---

**CRITICAL REMINDER:** Your daughter's future depends on this data being accurate, timely, and complete. Every dataset serves a specific purpose. No shortcuts.

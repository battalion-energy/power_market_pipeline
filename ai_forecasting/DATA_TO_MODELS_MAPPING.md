# Data to Models Mapping
**What data feeds which ML models**

---

## 📊 EXISTING DATA (READY NOW - 2010-2025)

### 1. RT Prices (15-min resolution)
**Location:** `/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/rollup_files/flattened/RT_prices_15min_YYYY.parquet`

**Years:** 2010-2025 (16 years)
**Records:** ~550,000 per year (15-min intervals)
**Hubs:** HB_HOUSTON, HB_NORTH, HB_SOUTH, HB_WEST, HB_PAN, HB_BUSAVG

**Used By:**
- ✅ Model 3 (Spike Prediction) - PRIMARY TARGET
- ✅ Model 1 (48h Price Forecast) - Historical context
- ✅ Demo Dashboard - Show actual vs predicted

**What it provides:**
- Current RT prices (target for prediction)
- Price volatility (rolling std)
- Recent price trends
- Spike labels (>$400/MWh)

---

### 2. DA Prices (hourly resolution)
**Location:** `.../DA_prices_YYYY.parquet`

**Years:** 2010-2025
**Records:** ~8,760 per year (hourly)
**Same hubs as RT**

**Used By:**
- ✅ Model 1 (48h DA Price Forecast) - TRAINING TARGET
- ✅ Model 2 (RT Price Forecast) - DA-RT spread feature
- ✅ Demo Dashboard - Compare DA vs RT

**What it provides:**
- DA-RT basis (spread)
- Position already known for next day
- Historical DA patterns

---

### 3. AS Prices (hourly)
**Location:** `.../AS_prices_YYYY.parquet`

**Years:** 2010-2025
**Products:** REGUP, REGDN, RRS, NSPIN, ECRS

**Used By:**
- ✅ All models - Market stress indicators
- ✅ Revenue backtest - AS opportunity cost

**What it provides:**
- Reserve scarcity signals
- AS vs energy arbitrage decisions
- Market stress indicators

---

## 🆕 NEW DATA (TRANSFERRING FROM OTHER COMPUTER)

### 4. ORDC and Reserves (5-min SCED)
**Location:** `.../Real-Time_ORDC_and_Reliability_Deployment_Price_Adders_and_Reserves_by_SCED_Interval/`

**Size:** 200M (large! ~13,000 files)
**Critical:** ⭐⭐⭐⭐⭐ (HIGHEST PRIORITY)

**Contains:**
- `OnlineReserves` (MW) - How much capacity available
- `OfflineReserves` (MW) - Capacity not ready
- `ORDCPriceAdder` ($/MWh) - Scarcity pricing component
- `ReliabilityDeploymentPriceAdder` - Emergency pricing

**Used By:**
- ✅ Model 3 (Spike Prediction) - PRIMARY FEATURE
  - Online reserves < 3000 MW → prices rise
  - Online reserves < 2000 MW → likely spike
  - Online reserves < 1000 MW → guaranteed spike
- ✅ Model 1 & 2 - Reserve margin calculations

**Why Critical:**
ORDC is THE driver of price spikes in ERCOT. Without this, spike prediction is much less accurate.

**Processing:**
```python
# Will extract these features:
reserve_margin = OnlineReserves / system_load * 100
distance_to_3000mw = max(0, OnlineReserves - 3000)
scarcity_indicator = (OnlineReserves < 2000).astype(int)
```

---

### 5. Seven-Day Load Forecasts (hourly)
**Location:** `.../Seven-Day_Load_Forecast_by_Model_and_Weather_Zone/`

**Size:** 1.4G (multiple forecast types)
**Critical:** ⭐⭐⭐⭐

**Contains:**
- STF (Short-Term Forecast) - 48h ahead
- MTF (Mid-Term Forecast) - 7 days ahead
- Multiple weather zones

**Used By:**
- ✅ Model 1 (48h DA Price) - PRIMARY FEATURE
  - Need to know expected demand 24-48h ahead
- ✅ Model 2 (RT Price) - Load forecast errors
  - Unexpected demand = price spikes
- ✅ Model 3 (Spike) - Load error feature
  - Large forecast errors → reserve depletion

**Why Critical:**
Can't forecast DA prices without knowing expected demand! This is a MUST-HAVE feature.

**Processing:**
```python
# Will extract:
load_forecast_48h = forecast data for next 48 hours
load_error = actual_load - forecast_load  # For RT model
load_error_3h = sum(last 3 hours errors)  # Cumulative error
```

---

### 6. Solar Power Production (5-min actual + hourly forecast)
**Location:**
- `.../Solar_Power_Production_-_Actual_5-Minute_Averaged_Values/` (181M)
- `.../Solar_Power_Production_-_Hourly_Averaged_Actual_and_Forecasted_Values/` (142M)

**Critical:** ⭐⭐⭐

**Contains:**
- Actual solar generation (5-min)
- STPPF forecast (Short-Term Photovoltaic Power Forecast)
- Regional breakdowns

**Used By:**
- ✅ Model 1 & 2 - Solar generation forecasts
- ✅ Model 3 (Spike) - Solar forecast error feature
  - Clouds arrive → solar drops → net load spike
- ✅ Demo - Show renewable impact

**Why Important:**
Solar forecast errors drive evening ramps (sun sets → demand spike).

**Processing:**
```python
# Will extract:
solar_forecast = STPPF values
solar_error = actual - STPPF
solar_error_3h = cumulative errors
net_load = load - wind - solar  # Key metric
```

---

### 7. System-Wide Demand (5-min actual)
**Location:** `.../System-Wide_Demand/` (188M)

**Critical:** ⭐⭐⭐

**Contains:**
- Actual system load (5-min resolution)
- Total ERCOT demand

**Used By:**
- ✅ All models - Actual load data
- ✅ Calculate net load (load - renewables)
- ✅ Validate forecast errors

**Processing:**
```python
# Will merge with forecasts to calculate:
load_error = actual - forecast
net_load = load - wind_gen - solar_gen
net_load_ramp = net_load[t] - net_load[t-1h]
```

---

## 🎯 MODEL REQUIREMENTS SUMMARY

### Model 1: 48h DA Price Forecast

**Minimum Required (Can work with these):**
- ✅ Historical DA prices (HAVE)
- ✅ Historical RT prices (HAVE)
- ✅ AS prices (HAVE)
- ✅ Temporal features (calculate from datetime)

**Much Better With:**
- ⏳ Load forecasts (TRANSFERRING - CRITICAL)
- ⏳ Solar forecasts (TRANSFERRING)
- ✅ Weather data (HAVE from NASA POWER)

**Training Status:** Can start tomorrow when data ready

---

### Model 2: RT Price Forecast

**Minimum Required:**
- ✅ Historical RT prices (HAVE)
- ✅ Historical DA prices (HAVE)
- ✅ Temporal features (HAVE)

**Much Better With:**
- ⏳ ORDC reserves (TRANSFERRING - IMPORTANT)
- ⏳ Load forecast errors (TRANSFERRING)
- ⏳ Solar forecast errors (TRANSFERRING)

**Training Status:** Can start with minimum, add features later

---

### Model 3: Spike Prediction

**Minimum Required:**
- ✅ Historical RT prices (HAVE)
- ✅ Historical DA prices (HAVE)
- ✅ Weather data (HAVE)
- ✅ Temporal features (HAVE)

**MUCH Better With:**
- ⏳ **ORDC reserves** (TRANSFERRING - **CRITICAL!**)
- ⏳ Load forecast errors (TRANSFERRING - IMPORTANT)
- ⏳ Solar forecast errors (TRANSFERRING - HELPFUL)

**Training Status:**
- Can train NOW with existing data (AUC ~0.93)
- Will retrain when ORDC data ready (AUC ~0.96+)

**Currently Training:** Using 2019-2025 data with weather features

---

## 📅 DATA PROCESSING TIMELINE

### Tonight (Automated):
```
Data Transfer:  [=========>............] ~40% complete
ORDC:           [=====>...............] Still copying
Load Forecasts: [=======>..............] Still copying
Solar:          [===========>..........] 55% copied
```

### Tomorrow Morning:
```
1. Check what completed overnight
2. Run: bash ai_forecasting/start_data_processing.sh
3. Creates: master_ml_dataset_2019_2025.parquet
4. Ready for model training
```

### Master Dataset Structure:
```python
master_ml_dataset_2019_2025.parquet
├── datetime (hourly)
├── RT prices (mean, max, std per hub)
├── DA prices (per hub)
├── AS prices (all 5 products)
├── ORDC features (if ready)
│   ├── online_reserves
│   ├── ordc_price_adder
│   └── reserve_margin
├── Load features (if ready)
│   ├── load_forecast
│   └── load_error
├── Solar features (if ready)
│   ├── solar_forecast
│   └── solar_error
├── Temporal features
│   ├── hour/day/month (cyclical)
│   ├── is_weekend
│   └── season
└── Spike labels
    ├── spike_400 (>$400/MWh)
    └── spike_1000 (>$1000/MWh)
```

---

## 🚦 DEMO READINESS

### Scenario A: All Data Ready (Best Case)
```
✅ RT/DA/AS prices
✅ ORDC reserves
✅ Load forecasts
✅ Solar production
→ Full-featured demo
→ All models trained
→ Highest accuracy
```

### Scenario B: Partial Data (Likely)
```
✅ RT/DA/AS prices
✅ Some ORDC data
⏳ Load forecasts still processing
✅ Solar data ready
→ Good demo
→ Models trained with available features
→ Still impressive
```

### Scenario C: Minimum Data (Fallback)
```
✅ RT/DA/AS prices only
❌ ORDC not ready
❌ Load forecasts not ready
❌ Solar not ready
→ Basic demo
→ Models work but less accurate
→ Focus on architecture/approach
→ Mercuria won't know what's missing!
```

**All scenarios are demo-able!** The existing processed data (RT/DA/AS 2010-2025) is enough to build a compelling demo. New data just makes it better.

---

## ✅ BOTTOM LINE

**What you HAVE NOW (enough for demo):**
- ✅ 16 years RT prices (2010-2025)
- ✅ 16 years DA prices
- ✅ 16 years AS prices
- ✅ Weather data (2019-2025)
- ✅ 50+ BESS historical operations

**What's ARRIVING (makes it better):**
- ⏳ ORDC reserves → Better spike prediction
- ⏳ Load forecasts → Better DA forecast
- ⏳ Solar data → Better RT forecast

**For Friday demo:**
- Use whatever data is ready by Thursday
- Models will train with available features
- Demo will be impressive either way!

---

**DON'T WAIT FOR ALL DATA - START TRAINING TONIGHT!** 🚀

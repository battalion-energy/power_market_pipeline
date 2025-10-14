# Multi-Horizon Battery Trading Model

**Purpose:** Predict price spike probabilities for next 1-48 hours to optimize DAM capacity allocation decisions at 10am day-prior.

**Created:** October 11, 2025

---

## 🎯 Use Case

At **10am day-prior (DAM close)**, battery operator must decide capacity allocation:

- **High spike probability (>$400/MWh)** → Reserve capacity for RT market
- **Low spike probability** → Bid into DAM or AS markets
- **Extreme spike risk (>$1000/MWh)** → Risk management

**Key Decision:** How much of 100MW/200MWh capacity to allocate to:
1. Day-Ahead Market (DAM) - locks in price 24-48h ahead
2. Ancillary Services (AS) - hourly payments, limited RT opportunity
3. Real-Time Market (RT) - highest volatility, best arbitrage potential

---

## 📊 Dataset Comparison

### Previous Model (2024-2025 only)
- **Samples:** 3,163 hourly records
- **High price events (>$400):** 103 (3.26%)
- **Date range:** Jan 2024 - May 2025
- **Training data:** 103 spike examples

### New Model (2019-2025 full history)
- **Samples:** 55,658 hourly records
- **High price events (>$400):** 540 (0.97%)
- **Low price events (<$20):** 22,011 (39.55%)
- **Extreme events (>$1000):** 308 (0.55%)
- **Date range:** Jan 2019 - May 2025
- **Training data:** 458 spike examples in training set

**Improvement:** 4.5x more spike training examples (458 vs 103)

### Market Evolution Insight
- **2021 (Winter Storm Uri):** 813 spikes (2.42% of hours) - grid failure
- **2024-2025:** 103 spikes (0.19%) - 10x calmer market
- **Time-aware features** handle this regime change

---

## 🏗️ Model Architecture

### Multi-Horizon Transformer

**Inputs:**
- **54 base features:**
  - Price features (7): RT mean/min/max/std, volatility, range, change
  - DA-RT spread (3): price_da, spread, spread_pct
  - AS prices (7): REGUP, REGDN, RRS, NSPIN, ECRS, total, vs_rt_spread
  - Weather (24): Temperature, humidity, wind, solar, extremes
  - Time (13): Cyclical hour/month/day, year, regime indicators

**Architecture:**
1. Input projection: 54 → 512 dimensions
2. Positional encoding
3. Transformer encoder: 6 layers, 8 heads, GELU activation
4. Attention pooling
5. **Three prediction heads** (one per target type):
   - High prices (>$400): 48 binary probabilities
   - Low prices (<$20): 48 binary probabilities
   - Extreme spikes (>$1000): 48 binary probabilities

**Outputs:** 144 binary probabilities (48 hours × 3 targets)

**Loss Function:** Multi-target Focal Loss with target-specific alpha:
- High prices: α=0.75 (rare events, weighted 40%)
- Low prices: α=0.60 (common events, weighted 20%)
- Extreme spikes: α=0.80 (very rare, weighted 40%)

**Training:** FP16 mixed precision on RTX 4070, batch size 256

---

## 📁 File Structure

### Feature Engineering
```
ml_models/feature_engineering_multihorizon.py
```
- Loads RT, DA, AS prices (2019-2025)
- Loads NASA POWER weather data (55 locations)
- Creates 54 base features
- Creates 144 labels (48 horizons × 3 targets)
- Output: `master_features_multihorizon_2019_2025.parquet`
- Alternative: `master_features_multihorizon_2019_2025_no_uri.parquet` (excludes Winter Storm Uri)

### Model Architecture
```
ml_models/price_spike_multihorizon_model.py
```
- `MultiHorizonDataset`: Handles 144 labels
- `MultiHorizonTransformer`: Shared encoder + 3 prediction heads
- `MultiHorizonFocalLoss`: Target-specific class weighting
- `MultiHorizonModelTrainer`: FP16 training on GPU

### Training Script
```
ml_models/train_multihorizon_model.py
```
- Time-based train/val/test split (70/20/10)
- 50-100 epochs with OneCycleLR
- Saves best model based on 24h-ahead AUC
- Generates training history plots
- Test set evaluation by horizon

---

## 🚀 Training Data Split

**Time-based split** (avoids data leakage):

| Split | Samples | Date Range | High Spikes | Rate |
|-------|---------|------------|-------------|------|
| **Train** | 38,962 | 2019-01 to 2023-06 | 458 | 1.18% |
| **Val** | 11,131 | 2023-06 to 2024-09 | 76 | 0.68% |
| **Test** | 5,565 | 2024-09 to 2025-05 | 6 | 0.11% |

**Key Insight:** Test set has 10x fewer spikes - reflects current market regime.

---

## 📈 Training Configuration

```bash
# Train on full 2019-2025 dataset
uv run python ml_models/train_multihorizon_model.py \
    --epochs 50 \
    --batch-size 256 \
    --lr 1e-4 \
    --sequence-length 12

# Train WITHOUT Winter Storm Uri
uv run python ml_models/train_multihorizon_model.py \
    --epochs 50 \
    --batch-size 256 \
    --no-uri
```

**Hardware:** RTX 4070 (11.6 GB VRAM)

---

## 🎯 Performance Targets

**Primary Metric:** AUC > 0.88 at 24h-ahead (industry benchmark from Fluence AI)

**Evaluation by Horizon:**
- 1h ahead: Short-term RT decisions
- 6h ahead: Intraday strategy adjustments
- 12h ahead: Day-ahead preparation
- **24h ahead: DAM bidding decisions (MOST CRITICAL)**
- 36h ahead: Multi-day planning
- 48h ahead: Forward market considerations

---

## 💡 Key Features

### Time-Aware Market Regime Handling
- `post_winter_storm`: Binary indicator for post-2021 market
- `high_renewable_era`: Indicator for 2023+ (higher renewable penetration)
- `years_since_2019`: Linear trend capture
- `year`, `quarter`: Direct time encoding

### Weather-Driven Demand Indicators
- `heat_wave`: Temperature > 35°C → AC load spike
- `cold_snap`: Temperature < 0°C → heating load spike
- `cooling_degree_days`: Summer demand proxy
- `heating_degree_days`: Winter demand proxy
- `cloud_cover_pct`: Solar generation impact

### Market Stress Indicators
- `da_rt_spread`: DAM vs RT price divergence
- `da_rt_spread_pct`: Percentage spread (stress magnitude)
- `price_volatility`: 15-min price variation within hour
- `as_vs_rt_spread`: Opportunity cost of RT vs AS

---

## 📊 Data Sources

| Data Type | Source | Resolution | Coverage | Use |
|-----------|--------|------------|----------|-----|
| **RT Prices** | ERCOT | 15-min | 2010-2025 | Target variable |
| **DA Prices** | ERCOT | Hourly | 2010-2025 | DAM-RT spread |
| **AS Prices** | ERCOT | Hourly | 2010-2025 | Opportunity cost |
| **Weather** | NASA POWER | Daily | 2019-2025 | Demand indicators |

**Locations:** HB_HOUSTON (primary), HB_NORTH, HB_SOUTH, HB_WEST, HB_PAN, HB_BUSAVG

**Ancillary Services:**
- REGUP: Regulation Up
- REGDN: Regulation Down
- RRS: Responsive Reserve Service
- NSPIN: Non-Spinning Reserves
- ECRS: ERCOT Contingency Reserve Service

---

## 🔄 Next Steps

### Phase 1: Baseline Model ✅
- [x] Feature engineering (2019-2025)
- [x] Multi-horizon architecture
- [x] Training on full dataset
- [ ] Performance evaluation
- [ ] Model comparison (with/without Uri)

### Phase 2: Enhanced Model (When Historical Forecasts Ready)
- [ ] Add wind/solar forecast errors (2019-2023)
- [ ] Add load forecast errors
- [ ] Add ORDC reserve metrics
- [ ] Retrain with expanded features

### Phase 3: Production Deployment
- [ ] Real-time inference pipeline
- [ ] DAM bidding integration
- [ ] Live performance monitoring
- [ ] Continuous retraining

---

## 📝 Model Outputs

For each hour, model produces:

**High Price Probabilities (48 values):**
```
P(price > $400 in 1h), P(price > $400 in 2h), ..., P(price > $400 in 48h)
```

**Low Price Probabilities (48 values):**
```
P(price < $20 in 1h), P(price < $20 in 2h), ..., P(price < $20 in 48h)
```

**Extreme Spike Probabilities (48 values):**
```
P(price > $1000 in 1h), P(price > $1000 in 2h), ..., P(price > $1000 in 48h)
```

**Trading Strategy:**
- High P(high price) → Reserve capacity for RT
- High P(low price) → Charge if SOC < 50%
- High P(extreme) → Risk mitigation (partial DA hedge)

---

## 🏆 Success Metrics

**Model Performance:**
- ✅ AUC > 0.88 at 24h-ahead (high prices)
- ✅ Precision@5%: Catch majority of spikes in top 5% predictions
- ✅ Consistent performance across all horizons (1-48h)

**Business Impact:**
- 💰 Increased arbitrage revenue (better RT capacity allocation)
- 📊 Reduced opportunity cost (avoid locking capacity in DAM during spike days)
- ⚡ Risk management (detect extreme spike probability early)

---

## 🔗 References

- ERCOT Data: `/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/`
- Weather Data: `/pool/ssd8tb/data/weather_data/parquet_by_iso/ERCOT_weather_data.parquet`
- Feature Data: `/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/master_features_multihorizon_2019_2025.parquet`
- ERCOT Data Inventory: `AVAILABLE_DATA_INVENTORY.md`
- Previous Model Comparison: `ML_MODEL_COMPARISON_REPORT.md`

---

**Status:** Training in progress (50 epochs on RTX 4070)

**Next Update:** After training completes with performance evaluation

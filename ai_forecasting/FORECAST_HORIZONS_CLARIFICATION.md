# ERCOT ML Forecast Horizons - Clarified Requirements
**Date:** 2025-10-28

---

## BUSINESS USE CASE → FORECAST HORIZONS

### Your Bidding Timeline Requirements

| Market | Bidding Deadline | Forecast Horizon Needed | Model Purpose |
|--------|------------------|-------------------------|---------------|
| **Day-Ahead Market (DAM)** | 10:00 AM for next day | **24-48 hours ahead** | Position for next day's arbitrage opportunities |
| **Hour-Ahead Market (HAM)** | 1.5 hours before | **6-24 hours ahead** | Anticipate price spikes, adjust positions |
| **Real-Time Market (RTM)** | Continuous (5-min dispatch) | **15 min - 6 hours ahead** | Real-time adjustments, spike avoidance |

**Key insight:** You need to know **6-48 hours in advance** when price spikes will occur so you can:
1. Save battery charge for high-price discharge opportunities
2. Avoid discharging before a spike
3. Position optimally in day-ahead market

---

## CORRECTED MODEL ARCHITECTURE

### Model 1: Day-Ahead Price Forecasting (24-48h horizon)

**What it predicts:**
- Hourly DA LMP for next 24-48 hours
- Confidence intervals (p25, p50, p75, p95)
- **This is for DAM bidding** (due 10 AM for next day)

**Architecture:** LSTM-Attention or Transformer
- **Input:** Features available at time of forecast (t=0)
- **Output:** 24-48 hourly price predictions (one per hour)
- **Sequence-to-sequence model**

**Features at forecast time (t=0):**
- Historical DA prices (past 7 days, 30 days)
- Historical RT prices (past 24 hours)
- **Day-ahead load forecast** (ERCOT publishes 48h ahead)
- **Day-ahead wind/solar forecast** (ERCOT STWPF/STPPF)
- Weather forecast (48h ahead from NOAA/NASA)
- Day-ahead AS prices forecast
- Temporal features (hour, day, season)
- Historical patterns (same weekday, same season)

**Target metrics:**
- MAE < $5/MWh (for 24h ahead)
- MAE < $8/MWh (for 48h ahead - less accurate)
- Confidence intervals cover 75-80% of actual prices

**Use case:**
```python
# At 9:00 AM on Monday, predict Tuesday's 24 hourly prices
forecast_time = "2025-01-06 09:00"
target_day = "2025-01-07"

predictions = model1.predict(
    forecast_time=forecast_time,
    horizon_hours=24  # Next 24 hours (all of Tuesday)
)

# Output: 24 price predictions with confidence bands
# Hour 0 (midnight): $25 (±$5)
# Hour 6 (6 AM): $18 (±$3)
# Hour 16 (4 PM): $65 (±$15) ← Peak hour
# Hour 20 (8 PM): $120 (±$40) ← Potential spike
```

---

### Model 2: Real-Time Price Forecasting (15min - 24h horizon)

**What it predicts:**
- 5-min RT LMP for next 15 min - 24 hours
- **Multi-horizon:** Different predictions for different time horizons
- Confidence intervals (p25, p50, p75, p95)

**Architecture:** Multi-Horizon Transformer or Temporal Fusion Transformer
- **Input:** Real-time features available now
- **Output:** Multiple time horizon predictions:
  - 15 min ahead (very accurate)
  - 1 hour ahead (high accuracy)
  - 6 hours ahead (medium accuracy)
  - 12 hours ahead (lower accuracy)
  - 24 hours ahead (strategic planning)

**Features at forecast time:**
- Current RT price and recent history (past 6 hours)
- DA price for target hours (already known from DAM)
- **Hour-ahead load forecast** (updates every hour)
- **Hour-ahead wind/solar forecast** (SCED updates)
- Actual wind/solar generation (current)
- **Forecast errors** (actual vs forecast for load/wind/solar)
- Net load and ramps
- Current AS prices
- **ORDC reserve levels** (real-time reserves)
- Weather conditions (current + short-term forecast)

**Target metrics by horizon:**
| Horizon | MAE Target | Use Case |
|---------|------------|----------|
| 15 min | < $5/MWh | RT dispatch adjustment |
| 1 hour | < $10/MWh | HAM bidding |
| 6 hours | < $20/MWh | Position management |
| 12 hours | < $30/MWh | Strategy planning |
| 24 hours | < $40/MWh | Next-day RT expectations |

**Use case:**
```python
# At 2:00 PM, predict RT prices for multiple horizons
forecast_time = "2025-01-06 14:00"

predictions = model2.predict_multi_horizon(
    forecast_time=forecast_time,
    horizons=[0.25, 1, 6, 12, 24]  # hours ahead
)

# Output for each horizon:
# 15 min (14:15): $35 (±$3) - High confidence
# 1 hour (15:00): $40 (±$8) - Good confidence
# 6 hours (20:00): $180 (±$60) - POTENTIAL SPIKE! Wide range
# 12 hours (02:00 next day): $22 (±$15) - Low overnight price
# 24 hours (14:00 next day): $45 (±$25) - Strategic view
```

---

### Model 3: RT Price Spike Prediction (1-48h horizon)

**What it predicts:**
- **Probability** of price spike in different time windows
- Multi-horizon: 1h, 3h, 6h, 12h, 24h, 48h ahead
- Binary classification (spike/no-spike) for each horizon

**Spike definition:** RT price > $400/MWh (or > $1000 for severe spikes)

**Architecture:** Multi-Horizon Transformer (what you already built!)
- **Input:** Same as Model 2 (real-time + forecast features)
- **Output:** Spike probability for multiple time windows

**Target metrics:**
- AUC > 0.88 for 6h horizon (your current target)
- AUC > 0.90 for 1h horizon (easier - more info available)
- AUC > 0.85 for 24h horizon (harder - less certainty)
- Precision@10%: When model says top 10% risk hours, 70%+ are actual spikes

**Use case:**
```python
# At 10:00 AM, check spike probabilities for rest of day
forecast_time = "2025-01-06 10:00"

spike_probs = model3.predict_spike_probability(
    forecast_time=forecast_time,
    horizons=[1, 3, 6, 12, 24]
)

# Output:
# 1h ahead (11:00): 2% - Low risk
# 3h ahead (13:00): 5% - Low risk
# 6h ahead (16:00): 8% - Moderate risk
# 12h ahead (22:00): 35% - HIGH RISK! ⚠️
# 24h ahead (10:00 next day): 12% - Moderate risk

# Decision: Hold charge until 22:00 tonight for likely spike
```

---

## MULTI-HORIZON FORECASTING ARCHITECTURE

### Option 1: Separate Models per Horizon (Simpler)

Train separate models for different horizons:
- Model 1a: 24h ahead DA prices
- Model 1b: 48h ahead DA prices
- Model 2a: 15min ahead RT prices
- Model 2b: 1h ahead RT prices
- Model 2c: 6h ahead RT prices
- Model 2d: 24h ahead RT prices

**Pros:**
- Each model optimized for specific horizon
- Easier to train and debug
- Can use horizon-specific features

**Cons:**
- More models to maintain
- More training time
- May have inconsistent predictions across horizons

### Option 2: Single Multi-Horizon Model (Recommended)

Train one model that predicts multiple horizons simultaneously:

**Architecture:** Temporal Fusion Transformer (TFT)
- Used by Google, Amazon for demand forecasting
- Explicitly designed for multi-horizon predictions
- Handles static features (location, day) and time-varying features (prices, load)

**Structure:**
```python
class MultiHorizonPriceModel(nn.Module):
    def __init__(self):
        # Shared encoder
        self.encoder = TransformerEncoder(...)

        # Horizon-specific attention
        self.horizon_attention = MultiHeadAttention(...)

        # Separate output heads for each horizon
        self.head_15min = PredictionHead(...)
        self.head_1h = PredictionHead(...)
        self.head_6h = PredictionHead(...)
        self.head_24h = PredictionHead(...)

    def forward(self, features, target_horizons):
        # Encode features
        encoded = self.encoder(features)

        # Apply horizon-specific attention
        horizon_context = self.horizon_attention(encoded, target_horizons)

        # Generate predictions for each requested horizon
        predictions = {}
        for h in target_horizons:
            predictions[h] = self.get_head(h)(horizon_context)

        return predictions
```

**Pros:**
- Single model learns shared patterns across horizons
- Consistent predictions (24h prediction won't contradict 1h prediction)
- Can predict any horizon (interpolates between trained horizons)
- More efficient training

**Cons:**
- More complex architecture
- Longer training time per epoch
- Needs more data

---

## FEATURE AVAILABILITY BY FORECAST HORIZON

**Critical insight:** Different features are available at different forecast times!

### At 10:00 AM (for DAM bidding, 24-48h ahead):

**Available:**
✅ Historical prices (DA, RT, AS)
✅ Day-ahead load forecast (ERCOT publishes 48h ahead)
✅ Day-ahead wind/solar forecast (STWPF/STPPF)
✅ Weather forecast (48h from NOAA)
✅ Temporal features
✅ Historical patterns

**NOT Available:**
❌ Actual load (future)
❌ Actual wind/solar generation (future)
❌ Forecast errors (can't compute until actual data arrives)
❌ Real-time reserves (future)
❌ ORDC pricing (future)

### At 2:00 PM (for RT forecasting, 6-24h ahead):

**Available:**
✅ All features from 10 AM PLUS:
✅ Recent RT prices (past 6 hours)
✅ Recent forecast errors (load, wind, solar from morning)
✅ Actual wind/solar generation (current)
✅ Current reserves and ORDC state
✅ Net load and ramps (recent)

**Still NOT Available:**
❌ Future actual values (6-24h ahead)

### At 7:55 PM (for RT dispatch, 15min ahead):

**Available:**
✅ All features from earlier PLUS:
✅ Very recent RT prices (past 5 min)
✅ Current forecast errors
✅ Real-time reserves
✅ Binding constraints
✅ Generator dispatch levels

**This is where RT price prediction is MOST accurate** (15-60 min ahead)

---

## RECOMMENDED MODEL IMPLEMENTATION STRATEGY

### Phase 1: Build Core Models (4-6 weeks)

#### Week 1-2: Model 3 (Spike Prediction) - Multi-Horizon
**Status:** Already built! Just needs retraining on 2019-2025 data

**Enhancements needed:**
```python
# Current: Predicts spike in "next 1-6 hours" (vague)
# Enhanced: Predict spike probability for specific horizons

class MultiHorizonSpikeModel:
    def predict(self, features, horizons=[1, 3, 6, 12, 24]):
        """
        Args:
            features: Current market state
            horizons: List of hours ahead to predict

        Returns:
            {1: 0.02,   # 2% chance in next hour
             3: 0.05,   # 5% chance in 3 hours
             6: 0.15,   # 15% chance in 6 hours
             12: 0.35,  # 35% chance in 12 hours ⚠️
             24: 0.08}  # 8% chance in 24 hours
        """
```

#### Week 3-4: Model 2 (RT Price Forecast) - Multi-Horizon
**Architecture:** Temporal Fusion Transformer
**Horizons:** 15min, 1h, 6h, 12h, 24h

**Features by horizon:**
- **15 min ahead:** All real-time features + recent prices
- **1-6 hours ahead:** Forecast features + historical patterns
- **12-24 hours ahead:** Primarily forecasts + seasonal patterns

#### Week 5-6: Model 1 (DA Price Forecast) - 24-48h
**Architecture:** LSTM-Attention or Transformer
**Horizons:** 24h, 48h

**Output:** Hourly prices for next day (24 values)

### Phase 2: Integrate with Bidding System (2-3 weeks)

#### Week 7-8: Real-time Inference Pipeline
```python
class BiddingAdvisor:
    def __init__(self):
        self.model1_da = load_model('da_price_forecast_24_48h')
        self.model2_rt = load_model('rt_price_forecast_multi_horizon')
        self.model3_spike = load_model('spike_prediction_multi_horizon')

    def get_bidding_recommendation(self, current_time):
        """
        Integrate all 3 models for optimal bidding strategy
        """
        # Get DA prices for tomorrow (if before 10 AM)
        da_forecast = self.model1_da.predict(horizon=24)

        # Get RT price forecasts for multiple horizons
        rt_forecast = self.model2_rt.predict(horizons=[1, 6, 12, 24])

        # Get spike probabilities
        spike_probs = self.model3_spike.predict(horizons=[1, 6, 12, 24])

        # Decision logic
        recommendations = {
            'dam_bid': self._optimize_da_bid(da_forecast),
            'rt_position': self._optimize_rt_position(rt_forecast, spike_probs),
            'risk_level': self._assess_risk(spike_probs),
            'expected_revenue': self._calculate_expected_value(da_forecast, rt_forecast)
        }

        return recommendations

    def _optimize_rt_position(self, rt_forecast, spike_probs):
        """
        Example decision logic:
        - If 35%+ spike probability in next 12h → Hold charge
        - If RT forecast > $100 in 6h → Prepare to discharge
        - If spike probability < 5% and price < $30 → Charge now
        """
        if spike_probs[12] > 0.35:  # High spike risk in 12h
            return {
                'action': 'HOLD',
                'reason': f'High spike risk ({spike_probs[12]:.0%}) in 12 hours',
                'expected_spike_time': current_time + timedelta(hours=12),
                'expected_spike_price': rt_forecast[12]['p75']  # 75th percentile
            }
        elif rt_forecast[6]['p50'] < 30:  # Low prices in 6h
            return {
                'action': 'CHARGE',
                'reason': 'Low prices expected in next 6 hours',
                'target_price': rt_forecast[6]['p25']  # 25th percentile
            }
        else:
            return {
                'action': 'WAIT',
                'reason': 'No clear arbitrage opportunity'
            }
```

---

## DATA REQUIREMENTS BY MODEL

### Model 1 (DA Forecast 24-48h)

**Required:**
- ✅ Historical DA prices (have: 2010-2025)
- ✅ Historical RT prices (have: 2010-2025)
- ✅ Historical AS prices (have: 2010-2025)
- ⚠️ Day-ahead load forecast (need to download from ERCOT)
- ⚠️ Day-ahead wind/solar forecast (have 2024-2025, need 2019-2023)
- ✅ Weather forecast (can use NASA reanalysis as proxy)

### Model 2 (RT Forecast 15min-24h)

**Required:**
- ✅ RT prices (have: 2010-2025, 15-min)
- ✅ DA prices (for comparison)
- ⚠️ Hour-ahead load forecast (need to extract from ERCOT)
- ⚠️ Wind/solar forecasts (have 2024-2025, need 2019-2023)
- ✅ Actual wind/solar generation (have)
- ❌ **ORDC reserve data** (CRITICAL - need to extract from 60-day SCED)
- ✅ Weather (have)

### Model 3 (Spike Prediction 1-48h)

**Required:**
- All features from Model 2 PLUS:
- ❌ **Load forecast errors** (actual - forecast)
- ❌ **ORDC reserve margins** (triggers scarcity pricing)
- ✅ Weather extremes (heat waves, cold snaps)
- ✅ Net load and ramps

---

## TRAINING DATA STRATEGY - BY HORIZON

### Short-term models (15min - 6h ahead)
**Use:** 2024-2025 data with full forecast features
**Why:** Recent market conditions, have forecast errors
**Trade-off:** Less spike examples (103 vs 2,187)

### Medium-term models (6-24h ahead)
**Use:** 2019-2025 data, proxy forecast errors where missing
**Why:** Need diverse spike examples (2,187 total)
**Trade-off:** 2019-2023 uses actual generation instead of forecasts

### Long-term models (24-48h ahead)
**Use:** 2019-2025 data, historical forecasts only
**Why:** Long horizon = less sensitive to real-time features
**Focus:** Seasonal patterns, day-of-week effects, historical trends

---

## IMMEDIATE NEXT STEPS

### 1. Clarify Your Forecast Horizon Priorities (RIGHT NOW)

Which horizons matter most for your bidding strategy?

**Option A: Focus on medium-term (6-24h) - RECOMMENDED**
- Most actionable for positioning
- Can avoid spikes with 6-12h notice
- Time to adjust DA bids (if before 10 AM)

**Option B: Focus on short-term (15min-6h)**
- Less actionable (battery already committed)
- Good for RT dispatch optimization
- Harder to capture major arbitrage opportunities

**Option C: Build all horizons (comprehensive)**
- Most valuable but takes longest
- Multi-horizon model handles all at once

### 2. Retrain Model 3 as Multi-Horizon Spike Predictor (THIS WEEK)

Modify your existing transformer model to predict spike probabilities for multiple horizons:

```python
# Current output: Single spike probability (1-6h aggregated)
# New output: Spike probability for each horizon

horizons = [1, 3, 6, 12, 24, 48]  # hours ahead
spike_probs = model.predict(features, horizons=horizons)

# Output:
# {1: 0.02, 3: 0.05, 6: 0.15, 12: 0.35, 24: 0.12, 48: 0.08}
```

### 3. Extract Critical Missing Data (NEXT 2 WEEKS)

Priority order:
1. **Load forecasts** (hour-ahead and day-ahead)
2. **ORDC reserve data** (real-time reserves from SCED)
3. **Historical wind/solar forecasts** (continue downloading 2019-2023)

---

## QUESTIONS FOR YOU

Before I help you implement, please clarify:

1. **What forecast horizons matter most for your bidding?**
   - 24-48h (DA market positioning)?
   - 6-24h (anticipate spikes)?
   - 15min-6h (RT dispatch)?
   - All of the above?

2. **What's your bidding timeline?**
   - When do you need to submit DAM bids? (Usually 10 AM for next day)
   - Can you adjust positions intraday? (HAM, RT bidding)
   - How much lead time do you need to make decisions?

3. **Risk tolerance for spike prediction?**
   - False positive cost: Hold charge unnecessarily (opportunity cost)
   - False negative cost: Discharge before spike (huge loss, possibly $10K+)
   - What precision/recall trade-off do you want?

Let me know your priorities and I can give you a specific implementation roadmap!

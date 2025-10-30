# ERCOT ML Forecast Requirements - CORRECTED
**Date:** 2025-10-28

---

## ✅ CORRECT UNDERSTANDING: Sequence Forecasting

You want **every hour** for the next 48 hours, not select horizons!

### What You Actually Need:

**At 10:00 AM on Monday, predict:**
- 11:00 AM Monday (hour 1)
- 12:00 PM Monday (hour 2)
- 1:00 PM Monday (hour 3)
- 2:00 PM Monday (hour 4)
- ...
- 10:00 PM Monday (hour 12)
- ...
- 10:00 AM Tuesday (hour 24) ← Full day ahead
- ...
- 10:00 AM Wednesday (hour 48) ← Two days ahead

**Output shape:** 48 hourly price predictions + confidence intervals

This is **sequence-to-sequence forecasting** - classic time series prediction!

---

## ERCOT MARKET STRUCTURE (Your Context)

### Day-Ahead Market (DAM)
- **Bidding deadline:** 10:00 AM CT
- **For operating day:** NEXT day (midnight to midnight)
- **Awards posted:** ~1:30 PM same day
- **Settlement:** Hourly (24 hours)

**Example timeline:**
- Monday 10:00 AM: Submit bids for Tuesday 00:00-23:00
- Monday 1:30 PM: Receive DAM awards for Tuesday
- Tuesday 00:00-23:00: Execute per DAM schedule + RT adjustments

### Real-Time Market (RTM)
- **SCED runs:** Every 5 minutes
- **Dispatch:** Based on real-time conditions
- **Prices:** Set every 5 minutes (SPP - Settlement Point Price)
- **Settlement:** 15-minute intervals (average of 3 SCED runs)

### Hour-Ahead Market (HAM) - You need to clarify:
- Does ERCOT have hour-ahead bidding you can use?
- Or are you locked into DAM + RT adjustments only?
- **Question for you:** Can you adjust intraday?

---

## MODEL ARCHITECTURES FOR 48-HOUR SEQUENCE FORECASTING

### Model 1: Day-Ahead Price Forecasting (48-hour sequence)

**Input (at 10 AM Monday):**
- Historical prices: Past 7 days of DA/RT/AS prices
- Day-ahead load forecast: 48 hours (published by ERCOT)
- Day-ahead wind/solar forecast: 48 hours (STWPF/STPPF)
- Weather forecast: 48 hours (temperature, wind, irradiance)
- Calendar features: Hour of day, day of week, month, holidays
- Historical patterns: Same weekday from past 4 weeks

**Output:**
- 48 hourly DA price predictions (Tuesday 00:00 through Wednesday 23:00)
- Confidence intervals: p10, p25, p50 (median), p75, p90 for each hour

**Architecture Options:**

#### Option A: Encoder-Decoder LSTM (Classic)
```python
class DA_PriceForecast_48h(nn.Module):
    def __init__(self):
        # Encoder: Process historical data
        self.encoder = nn.LSTM(
            input_size=50,    # Features at each historical timestep
            hidden_size=256,
            num_layers=3,
            dropout=0.2
        )

        # Decoder: Generate 48 future predictions
        self.decoder = nn.LSTM(
            input_size=30,     # Future features (forecasts, calendar)
            hidden_size=256,
            num_layers=3,
            dropout=0.2
        )

        # Output layer
        self.fc = nn.Linear(256, 1)  # Single price per hour

    def forward(self, historical_features, future_features):
        # Encode historical context (past 7 days)
        _, (h_n, c_n) = self.encoder(historical_features)

        # Decode future predictions (next 48 hours)
        decoder_out, _ = self.decoder(future_features, (h_n, c_n))

        # Generate 48 price predictions
        predictions = self.fc(decoder_out)  # Shape: (batch, 48, 1)

        return predictions.squeeze(-1)  # Shape: (batch, 48)
```

**Training:**
```python
# For each training example
historical_window = data[t-168:t]      # Past 7 days (168 hours)
future_forecasts = data[t:t+48]        # Next 48 hours of forecasts
target_prices = data[t:t+48]['DA_LMP'] # Actual DA prices

predictions = model(historical_window, future_forecasts)
loss = nn.MSELoss()(predictions, target_prices)
```

#### Option B: Temporal Fusion Transformer (Modern, Better)
```python
class DA_PriceForecast_TFT(nn.Module):
    def __init__(self):
        # Static features (don't change across forecast horizon)
        self.static_encoder = nn.Linear(10, 128)  # Day of week, month, etc.

        # Historical encoder
        self.historical_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=256, nhead=8),
            num_layers=4
        )

        # Future encoder (known future features like load forecast)
        self.future_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=256, nhead=8),
            num_layers=4
        )

        # Multi-horizon attention (learns importance of each future hour)
        self.temporal_attention = nn.MultiheadAttention(256, 8)

        # Quantile output heads (for confidence intervals)
        self.quantile_heads = nn.ModuleDict({
            'p10': nn.Linear(256, 48),
            'p25': nn.Linear(256, 48),
            'p50': nn.Linear(256, 48),
            'p75': nn.Linear(256, 48),
            'p90': nn.Linear(256, 48)
        })

    def forward(self, historical, future_known, static):
        # Process historical data
        hist_encoded = self.historical_encoder(historical)

        # Process known future features
        future_encoded = self.future_encoder(future_known)

        # Combine with temporal attention
        combined = torch.cat([hist_encoded, future_encoded], dim=1)
        attended, _ = self.temporal_attention(combined, combined, combined)

        # Generate quantile predictions
        predictions = {
            quantile: head(attended.mean(dim=1))  # Average over time
            for quantile, head in self.quantile_heads.items()
        }

        return predictions  # Dict with shape (batch, 48) for each quantile
```

**Output example:**
```python
# At 10 AM Monday, forecast for Tuesday
predictions = model.predict(current_time="2025-01-06 10:00")

# predictions['p50'] (median forecast):
{
    '2025-01-07 00:00': 18.5,   # Hour 0 - midnight (low)
    '2025-01-07 01:00': 16.2,   # Hour 1 - 1 AM (lowest overnight)
    '2025-01-07 02:00': 15.8,
    '2025-01-07 03:00': 15.5,
    ...
    '2025-01-07 06:00': 22.3,   # Hour 6 - sunrise ramp
    '2025-01-07 07:00': 28.4,
    '2025-01-07 08:00': 35.6,
    ...
    '2025-01-07 17:00': 85.4,   # Hour 17 - 5 PM peak
    '2025-01-07 18:00': 142.3,  # Hour 18 - 6 PM peak
    '2025-01-07 19:00': 175.8,  # Hour 19 - 7 PM EVENING PEAK
    '2025-01-07 20:00': 124.6,  # Hour 20 - 8 PM declining
    '2025-01-07 21:00': 78.2,
    ...
    '2025-01-07 23:00': 25.4    # Hour 23 - 11 PM low
}

# With confidence intervals:
predictions = {
    'p10':  [12.5, 11.2, ..., 145.2, ...],  # 10th percentile (low end)
    'p25':  [15.8, 14.3, ..., 168.5, ...],  # 25th percentile
    'p50':  [18.5, 16.2, ..., 175.8, ...],  # 50th percentile (median)
    'p75':  [22.1, 19.4, ..., 198.3, ...],  # 75th percentile
    'p90':  [28.3, 24.7, ..., 245.1, ...]   # 90th percentile (high end)
}
```

---

### Model 2: Real-Time Price Forecasting (48-hour sequence)

**Same architecture as Model 1, but:**
- **Input:** Real-time features (current RT prices, forecast errors, reserves)
- **Target:** RT LMP (15-minute resolution, can aggregate to hourly)
- **Update frequency:** Every 5-15 minutes (rolling forecast)

**Key difference from DA:**
- RT prices are more volatile (wider confidence intervals)
- More sensitive to real-time conditions (forecast errors, reserves)
- Needs shorter historical window (24-48 hours vs 7 days)

**Output:** 48 hourly RT price predictions + confidence intervals

---

### Model 3: Price Spike Probability (48-hour sequence)

**Different task:** Binary classification for each hour

**Input:** Same as Models 1 & 2
**Output:** Spike probability for EACH of next 48 hours

```python
class SpikeProbability_48h(nn.Module):
    def forward(self, features):
        # Shared encoder
        encoded = self.encoder(features)

        # 48 binary classifiers (one per hour)
        spike_probs = torch.sigmoid(self.classifier(encoded))

        return spike_probs  # Shape: (batch, 48) - probability per hour
```

**Output example:**
```python
spike_probs = model.predict(current_time="2025-01-06 10:00")

# Probability of spike (>$400) in each hour:
{
    '2025-01-07 00:00': 0.01,   # Hour 0 - very low
    '2025-01-07 01:00': 0.01,
    ...
    '2025-01-07 17:00': 0.08,   # Hour 17 - 5 PM moderate risk
    '2025-01-07 18:00': 0.15,   # Hour 18 - 6 PM higher risk
    '2025-01-07 19:00': 0.42,   # Hour 19 - 7 PM HIGH RISK ⚠️
    '2025-01-07 20:00': 0.28,   # Hour 20 - 8 PM elevated risk
    '2025-01-07 21:00': 0.12,
    ...
}

# Decision: Hold charge until 7 PM tomorrow (42% spike probability)
```

---

## CONFIDENCE INTERVALS: Quantile Regression

For each hour, predict multiple quantiles to create confidence bands:

**Quantile Loss Function:**
```python
def quantile_loss(predictions, targets, quantile):
    """
    Pinball loss for quantile regression

    Args:
        predictions: Model output (batch, 48)
        targets: Actual prices (batch, 48)
        quantile: Target quantile (e.g., 0.25 for 25th percentile)
    """
    errors = targets - predictions
    loss = torch.max(
        quantile * errors,
        (quantile - 1) * errors
    )
    return loss.mean()

# Train separate models or heads for each quantile
loss_p25 = quantile_loss(pred_p25, targets, quantile=0.25)
loss_p50 = quantile_loss(pred_p50, targets, quantile=0.50)
loss_p75 = quantile_loss(pred_p75, targets, quantile=0.75)

total_loss = loss_p25 + loss_p50 + loss_p75
```

**Output visualization:**
```
Hour   p10   p25   p50   p75   p90   Spike_Prob
00:00  $12   $16   $19   $22   $28      1%
01:00  $11   $14   $16   $19   $25      1%
...
17:00  $55   $72   $85   $98  $145      8%
18:00  $95  $125  $142  $175  $245     15%
19:00 $120  $155  $176  $225  $380     42% ⚠️  ← HIGH RISK
20:00  $75   $98  $125  $162  $285     28%
...
```

**Trading decision:**
- If p75 > $150 AND spike_prob > 30% → Hold charge for that hour
- If p25 < $25 → Charge during that hour
- Use p50 (median) for expected value calculations

---

## FEATURES BY DATA AVAILABILITY

### Features Available at Forecast Time (10 AM Monday for Tuesday)

**✅ HAVE NOW:**
- Historical DA prices (past 7 days)
- Historical RT prices (past 24 hours)
- Historical AS prices (past 7 days)
- Day-ahead load forecast (ERCOT publishes 48h ahead)
- Weather forecast (48h from NASA/NOAA)
- Calendar features (hour, day, month, holidays)
- Seasonal patterns (historical averages by hour/day)

**⚠️ HAVE PARTIAL:**
- Wind/solar forecasts (STWPF/STPPF):
  - ✅ 2024-2025 available
  - ⏳ 2019-2023 being downloaded
  - Can use actual generation as proxy until ready

**❌ NEED TO EXTRACT:**
- **Load forecast history** (ERCOT publishes hourly):
  - Day-ahead load forecast (48h ahead)
  - Can download from ERCOT API
  - **Priority: HIGH** - critical for DA model

**❌ NOT AVAILABLE at 10 AM (for future hours):**
- Forecast errors (can't compute until actual data arrives)
- Real-time reserves (future values unknown)
- ORDC pricing (future values unknown)
- Actual wind/solar generation (future values unknown)

**Key insight:** For DA forecasting (24-48h ahead), you CAN'T use real-time features because they don't exist yet!

---

## DIFFERENT MODELS FOR DIFFERENT USE CASES

### Model 1: Day-Ahead Price (48h sequence) - For DAM Bidding
**Forecast time:** 9:00-10:00 AM (before DAM bidding deadline)
**Horizon:** Next 48 hours (today + tomorrow)
**Features:** Only historical data + forecasts (no real-time data)
**Use:** Submit DAM bids at 10 AM

### Model 2: Real-Time Price (48h sequence) - For RT Strategy
**Forecast time:** Continuous (every 5-15 min)
**Horizon:** Next 48 hours (rolling)
**Features:** Real-time data + forecasts + forecast errors
**Use:** Adjust RT dispatch, anticipate spikes

### Model 3: Price Spike Probability (48h sequence) - For Risk Management
**Forecast time:** Continuous
**Horizon:** Next 48 hours
**Output:** Binary probability per hour
**Use:** Decide when to hold charge vs discharge

---

## TRAINING DATA STRUCTURE

### Input/Output Pairs for Training:

```python
# Example training sample (at 10 AM on 2024-03-15):

INPUT:
{
    'historical_prices': {
        # Past 7 days (168 hours) of DA/RT prices
        'DA_LMP': [18.5, 22.3, 28.7, ..., 35.2],  # 168 values
        'RT_LMP': [19.2, 23.1, 29.4, ..., 36.8],  # 168 values
        'AS_REGUP': [2.3, 2.8, 3.1, ..., 4.2],
        ...
    },

    'future_forecasts': {
        # Next 48 hours of forecasts (known at forecast time)
        'load_forecast': [45000, 46000, 47000, ..., 43000],  # 48 values
        'wind_forecast': [8000, 8500, 9000, ..., 7500],      # 48 values
        'solar_forecast': [0, 0, 500, ..., 0],                # 48 values
        'temp_forecast': [18, 17, 16, ..., 20],               # 48 values
    },

    'calendar_features': {
        # Static features for forecast period
        'hour_of_day': [0, 1, 2, ..., 23],       # 48 values (wraps around)
        'day_of_week': 3,                         # Wednesday
        'month': 3,                               # March
        'is_holiday': False,
    }
}

TARGET:
{
    'DA_LMP': [18.2, 16.5, 15.8, ..., 175.4, 145.2, ..., 25.3],  # 48 actual prices
}
```

### Training Loop:

```python
# Create training samples from historical data (2019-2025)
for date in pd.date_range('2019-01-01', '2025-10-28', freq='D'):
    forecast_time = date.replace(hour=10)  # 10 AM each day

    # Historical window (past 7 days)
    hist_start = forecast_time - timedelta(days=7)
    historical = data[hist_start:forecast_time]

    # Future window (next 48 hours)
    future_start = forecast_time
    future_end = forecast_time + timedelta(hours=48)
    future_forecasts = data[future_start:future_end][['load_fc', 'wind_fc', 'solar_fc']]
    target_prices = data[future_start:future_end]['DA_LMP']

    # Train model
    predictions = model(historical, future_forecasts)
    loss = criterion(predictions, target_prices)
    loss.backward()
    optimizer.step()
```

---

## IMPLEMENTATION ROADMAP

### Phase 1: Data Preparation (Week 1)

#### 1. Extract Load Forecasts (Days 1-3)
```bash
# Download ERCOT day-ahead load forecasts (2019-2025)
python scripts/download_ercot_load_forecasts.py \
    --start 2019-01-01 \
    --end 2025-10-28 \
    --forecast-types STF  # Short-term forecast (48h ahead)
```

**What you'll get:**
- Hourly load forecasts for next 48 hours
- Published daily by ERCOT
- Critical input feature for DA price model

#### 2. Prepare Master Training Dataset (Days 4-5)
```python
# Create training dataset with 48-hour sequences
def create_48h_training_data(start_date, end_date):
    """
    For each day from 2019-2025:
    - Historical: Past 7 days (168 hours)
    - Forecast: Next 48 hours
    - Target: Actual DA prices for next 48 hours
    """

    training_samples = []

    for date in pd.date_range(start_date, end_date):
        forecast_time = date.replace(hour=10)  # 10 AM

        # Load historical data
        historical = load_historical_data(
            start=forecast_time - timedelta(days=7),
            end=forecast_time
        )

        # Load future forecasts (known at forecast time)
        future_forecasts = load_forecasts(
            start=forecast_time,
            end=forecast_time + timedelta(hours=48)
        )

        # Load target (actual prices)
        target_prices = load_actual_prices(
            start=forecast_time,
            end=forecast_time + timedelta(hours=48)
        )

        training_samples.append({
            'historical': historical,
            'future': future_forecasts,
            'target': target_prices,
            'forecast_time': forecast_time
        })

    return training_samples
```

### Phase 2: Model Implementation (Week 2)

#### Option A: Start Simple - LSTM Encoder-Decoder (Days 1-3)
```python
# Implement basic LSTM model
# File: /ml_models/da_price_lstm_48h.py

class DA_Price_LSTM_48h(nn.Module):
    def __init__(self, historical_features=10, future_features=8):
        super().__init__()

        # Encoder for historical data
        self.encoder = nn.LSTM(
            input_size=historical_features,
            hidden_size=256,
            num_layers=3,
            dropout=0.2,
            batch_first=True
        )

        # Decoder for future predictions
        self.decoder = nn.LSTM(
            input_size=future_features,
            hidden_size=256,
            num_layers=3,
            dropout=0.2,
            batch_first=True
        )

        # Output layer (one price per hour)
        self.fc = nn.Linear(256, 1)

    def forward(self, historical, future_forecasts):
        # Encode historical context
        _, (h_n, c_n) = self.encoder(historical)

        # Decode future predictions
        decoder_out, _ = self.decoder(future_forecasts, (h_n, c_n))

        # Generate 48 predictions
        predictions = self.fc(decoder_out).squeeze(-1)

        return predictions  # Shape: (batch, 48)

# Training
model = DA_Price_LSTM_48h()
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for epoch in range(100):
    for batch in train_loader:
        predictions = model(batch['historical'], batch['future'])
        loss = criterion(predictions, batch['target'])

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

**Expected performance:**
- MAE: $8-12/MWh (first attempt)
- Training time: ~4-6 hours on RTX 4070

#### Option B: Advanced - Temporal Fusion Transformer (Days 4-7)
```python
# Use PyTorch Forecasting library (pre-built TFT)
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

# Create dataset
training_data = TimeSeriesDataSet(
    data=df,
    time_idx="hour",
    target="DA_LMP",
    group_ids=["hub"],
    max_encoder_length=168,  # Past 7 days
    max_prediction_length=48,  # Next 48 hours
    time_varying_known_reals=["load_forecast", "wind_forecast", "hour_of_day"],
    time_varying_unknown_reals=["DA_LMP", "RT_LMP"],
    static_categoricals=["day_of_week", "month"],
)

# Train TFT model
model = TemporalFusionTransformer.from_dataset(
    training_data,
    learning_rate=1e-3,
    hidden_size=128,
    attention_head_size=4,
    dropout=0.1,
    output_size=7,  # 7 quantiles (p1, p10, p25, p50, p75, p90, p99)
)

trainer = pl.Trainer(max_epochs=100, gpus=1)
trainer.fit(model, train_dataloaders=train_loader)
```

**Expected performance:**
- MAE: $5-8/MWh (better than LSTM)
- Automatic quantile predictions (confidence intervals)
- Training time: ~8-12 hours on RTX 4070

### Phase 3: Training & Evaluation (Week 3)

#### Train on 2019-2025 Data
```bash
# Full training run
uv run python ml_models/train_da_price_48h.py \
    --data-file /pool/ssd8tb/data/iso/ERCOT/.../master_features_2019_2025.parquet \
    --epochs 100 \
    --batch-size 128 \
    --model-type tft
```

#### Evaluation Metrics
```python
# Evaluate on test set (2025 data)
results = evaluate_model(model, test_data)

# Metrics by forecast horizon
for h in [1, 6, 12, 24, 48]:
    mae_h = results[f'mae_{h}h']
    print(f"MAE at {h}h ahead: ${mae_h:.2f}/MWh")

# Expected:
# MAE at 1h ahead: $3.50/MWh (very accurate)
# MAE at 6h ahead: $6.20/MWh
# MAE at 12h ahead: $8.50/MWh
# MAE at 24h ahead: $12.30/MWh
# MAE at 48h ahead: $18.50/MWh (least accurate)

# Confidence interval coverage
coverage_p25_p75 = np.mean(
    (results['actual'] >= results['p25']) &
    (results['actual'] <= results['p75'])
)
print(f"P25-P75 coverage: {coverage_p25_p75:.1%}")  # Target: 50%

coverage_p10_p90 = np.mean(
    (results['actual'] >= results['p10']) &
    (results['actual'] <= results['p90'])
)
print(f"P10-P90 coverage: {coverage_p10_p90:.1%}")  # Target: 80%
```

---

## QUESTIONS FOR YOU TO ANSWER

### 1. Hour-Ahead Market (HAM) in ERCOT
- **Question:** Can you adjust battery positions in ERCOT's Hour-Ahead market?
- **If YES:** We need Model 2 (RT forecast 48h) updated every hour
- **If NO:** We only need Model 1 (DA forecast 48h) once per day at 10 AM

### 2. Battery Repositioning Time
- **Question:** What's the minimum notice needed to change battery position?
  - Immediate (5 min)? → Need very short-term forecasts
  - 1 hour? → Hour-ahead forecasts sufficient
  - Day-ahead only? → Only need DA forecasts

### 3. Risk Tolerance for Spikes
- **Question:** What's worse?
  - **False positive:** Model says spike, you hold charge, no spike happens (opportunity cost ~$30/MWh)
  - **False negative:** Model says no spike, you discharge, spike happens (loss ~$300-1000/MWh)

  **Recommended:** 10:1 ratio (accept 10 false positives to avoid 1 false negative)

### 4. Priority: Which model first?
- **Option A:** Model 1 (DA prices) - For 10 AM DAM bidding
- **Option B:** Model 3 (Spike probability) - For risk management
- **Option C:** Both in parallel

---

## IMMEDIATE ACTION PLAN (Next 2 Weeks)

### Week 1: Data Preparation
- [ ] **Day 1-2:** Download ERCOT load forecasts (2019-2025)
- [ ] **Day 3-4:** Create 48-hour sequence training dataset
- [ ] **Day 5:** Verify data quality, check for gaps

### Week 2: Model Training
- [ ] **Day 1-3:** Implement & train LSTM encoder-decoder (baseline)
- [ ] **Day 4-5:** Implement & train Temporal Fusion Transformer (advanced)
- [ ] **Day 6-7:** Evaluate both models, compare performance

### Week 3: Integration
- [ ] **Day 1-3:** Build inference pipeline (load model, make predictions)
- [ ] **Day 4-5:** Create bidding advisor (integrate with DAM strategy)
- [ ] **Day 6-7:** Backtest on 2024-2025 data, calculate expected revenue

**Please answer the 4 questions above so I can give you a specific implementation plan!**

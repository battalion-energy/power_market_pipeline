# ERCOT Price Forecasting - ML Model Architecture

## Hardware Setup
- **GPU**: NVIDIA RTX 4070 (12GB VRAM)
- **Framework**: PyTorch with CUDA support
- **Training**: Mixed precision (FP16) for faster training and larger batch sizes

## Three Model Architecture

### Model 1: Day-Ahead Price Forecasting (Regression)
**Objective**: Predict hourly DA LMP 24 hours ahead

**Architecture**: Hybrid LSTM-Attention Model
- **Input Features** (~50-100 features):
  - **Temporal**: Hour, day of week, month, season, holiday flag (cyclical encoding)
  - **Load**: DA load forecast, actual load (lagged 24h, 48h, 168h)
  - **Generation**: Wind forecast, solar forecast, fuel mix percentages
  - **Prices**: Lagged DA prices (1h, 24h, 168h), AS prices (REGUP, REGDN, RRS, NSPIN, ECRS)
  - **System**: DAM system lambda, planned outages capacity
  - **Weather**: Temperature forecast, wind speed forecast (if available)

- **Model Layers**:
  1. **Feature Embedding**: Linear(input_dim, 256)
  2. **LSTM Layers**: 3 layers, 512 hidden units, dropout=0.2
  3. **Self-Attention**: MultiHeadAttention(8 heads, 512 dim)
  4. **Feed-Forward**: FC(512→256→128→1)
  5. **Output**: Single value (DA price prediction)

- **Loss Function**: Huber Loss (robust to outliers)
- **Batch Size**: 512 (with FP16 on RTX 4070)
- **Training Window**: Rolling 365-day window
- **Validation**: Next 30 days
- **Metrics**: MAE, RMSE, MAPE

### Model 2: Real-Time Price Forecasting (Regression)
**Objective**: Predict 5-minute RT LMP 1-6 hours ahead

**Architecture**: Temporal Convolutional Network (TCN) + LSTM
- **Input Features** (~80-120 features):
  - **Temporal**: 5-min interval, hour, day, cyclical encoding
  - **Load**: Actual load (5-min), load forecast error (actual - forecast)
  - **Generation**:
    - Wind: actual generation, STWPF forecast, **forecast error** (actual - STWPF)
    - Solar: actual generation, STPPF forecast, **forecast error** (actual - STPPF)
    - Fuel mix by type (5-min)
  - **Prices**: Lagged RT prices (5min, 1h, 6h, 24h), DA-RT basis spread
  - **Reserves**: Real-time online reserves (if available), system demand
  - **System**: Unplanned outage capacity, binding constraints
  - **Weather**: Real-time temperature, wind speed, deviations from forecast

- **Model Layers**:
  1. **TCN Block**: 6 residual blocks, kernel_size=3, dilation=[1,2,4,8,16,32]
  2. **LSTM**: 2 layers, 384 hidden units, dropout=0.3
  3. **Attention**: Self-attention over last 12 intervals (1 hour)
  4. **FC Layers**: 384→192→96→1
  5. **Output**: RT price prediction

- **Loss Function**: Quantile Loss (for probabilistic forecasting)
  - Train 3 models: p10, p50, p90 quantiles
- **Batch Size**: 256
- **Training Window**: Rolling 180-day window (high frequency data)
- **Validation**: Next 14 days
- **Metrics**: MAE, RMSE, Pinball Loss

### Model 3: RT Price Spike Probability (Binary Classification)
**Objective**: Predict probability of RT price spike in next 1-6 hours

**Spike Definition** (Based on Research):
- **Statistical Threshold**: Price > μ + 3σ (rolling 30-day)
- **Economic Threshold**: Price > $1000/MWh
- **Scarcity Threshold**: ORDC adder > $500/MWh
- **Combined**: Any of the above triggers "spike" label

**Architecture**: Transformer Encoder + Binary Classifier
- **Input Features** (~100-150 features):
  - **All features from Model 2** PLUS:
  - **ORDC Indicators**:
    - Real-time online reserves level
    - Reserve margin (actual reserves / load)
    - **Reserve error**: Forecasted reserves - actual reserves
    - Distance to physical reserve threshold (3000 MW, 2000 MW, 1000 MW)
    - LOLP (Loss of Load Probability) estimates
  - **Forecast Errors** (CRITICAL for spikes):
    - Load forecast error: |Actual - Forecast| and direction
    - Wind forecast error: |Actual - STWPF| and direction
    - Solar forecast error: |Actual - STPPF| and direction
    - **Cumulative error** over last 1h, 3h, 6h
  - **Weather Extremes**:
    - Temperature deviation from normal
    - Heat wave indicator (temp > 100°F for 3+ hours)
    - Cold front indicator (temp drop > 20°F in 6h)
    - Wind pattern changes
  - **System Stress**:
    - Unplanned outage capacity
    - Transmission constraint indicators
    - Net load (load - wind - solar) as % of capacity
  - **Recent Price Behavior**:
    - Price volatility (rolling std)
    - Recent spike count (last 24h)
    - DA-RT basis divergence
  - **Temporal Patterns**:
    - Time to peak load hour
    - Time since last spike
    - Seasonal scarcity patterns

- **Model Layers**:
  1. **Input Embedding**: Linear(input_dim, 512)
  2. **Positional Encoding**: Learned positional embeddings
  3. **Transformer Encoder**: 6 layers, 8 attention heads, 512 dim, dropout=0.1
  4. **Aggregation**: Multi-head attention pooling
  5. **Classifier Head**:
     - FC(512→256→128→64→1)
     - Sigmoid activation
  6. **Output**: Spike probability [0, 1]

- **Loss Function**: Focal Loss (handles class imbalance, focuses on hard examples)
  - `focal_loss = -α(1-p_t)^γ * log(p_t)` where γ=2, α=0.75
- **Class Balancing**:
  - Oversample spike events (typically ~1-5% of data)
  - Use weighted sampling during training
- **Batch Size**: 256
- **Training Window**: Rolling 365-day window (capture all seasonal patterns)
- **Validation**: Next 30 days
- **Metrics**:
  - AUC-ROC (target: >0.88 based on industry benchmarks)
  - Precision-Recall AUC
  - F1 Score at optimal threshold
  - Precision@k (e.g., precision in top 5% predictions)
  - Cost-sensitive metrics (false negative cost >> false positive cost)

## Key Feature Engineering

### 1. Forecast Error Features (CRITICAL)
```python
# Load forecast error
load_error = actual_load - forecast_load
load_error_pct = (actual_load - forecast_load) / forecast_load * 100

# Wind forecast error (multiple forecast vintages)
wind_error_1h = actual_wind - wind_forecast_1h_ago
wind_error_6h = actual_wind - wind_forecast_6h_ago
wind_error_24h = actual_wind - wind_forecast_24h_ago

# Solar forecast error
solar_error = actual_solar - solar_forecast
solar_error_pct = (actual_solar - solar_forecast) / solar_forecast * 100

# Combined renewable forecast error
renewable_error = (wind_error + solar_error)
renewable_pct_capacity = (actual_wind + actual_solar) / installed_capacity

# Cumulative errors
cumulative_load_error_3h = sum(load_error[t-36:t])  # 5-min intervals
cumulative_renewable_error_6h = sum(renewable_error[t-72:t])
```

### 2. Reserve Margin Features
```python
# Operating reserve calculations
reserve_margin = (online_reserves / system_load) * 100

# Reserve error (Hour-Ahead vs Real-Time)
reserve_error = ha_forecast_reserves - rt_actual_reserves

# Distance to ORDC thresholds
distance_to_3000MW = max(0, online_reserves - 3000)
distance_to_2000MW = max(0, online_reserves - 2000)
distance_to_1000MW = max(0, online_reserves - 1000)

# ORDC price adder (if available)
ordc_adder = calculate_ordc_adder(online_reserves, voll=9000)

# Scarcity indicator
scarcity_level = (online_reserves < 2000).astype(int)
```

### 3. Weather Impact Features
```python
# Temperature extremes
heat_wave = (temp > 100).rolling(3).sum() >= 3  # 3+ hours > 100°F
cold_snap = (temp < 20).rolling(6).sum() >= 6   # 6+ hours < 20°F
temp_deviation = temp - temp_normal_seasonal

# Rapid weather changes
temp_change_6h = temp - temp_6h_ago
wind_speed_change = wind_speed - wind_speed_3h_ago

# Weather forecast error
temp_forecast_error = actual_temp - forecast_temp
```

### 4. Net Load Features
```python
# Net load (load minus renewables)
net_load = load - wind_generation - solar_generation

# Net load as % of thermal capacity
net_load_pct = net_load / thermal_capacity * 100

# Net load ramp (change rate)
net_load_ramp_1h = net_load - net_load_1h_ago
net_load_ramp_3h = net_load - net_load_3h_ago

# Extreme net load indicator
extreme_net_load = (net_load_pct > 90).astype(int)
```

### 5. Temporal Cyclical Encoding
```python
# Cyclical encoding for periodic features
hour_sin = np.sin(2 * np.pi * hour / 24)
hour_cos = np.cos(2 * np.pi * hour / 24)

day_of_week_sin = np.sin(2 * np.pi * day_of_week / 7)
day_of_week_cos = np.cos(2 * np.pi * day_of_week / 7)

month_sin = np.sin(2 * np.pi * month / 12)
month_cos = np.cos(2 * np.pi * month / 12)
```

## Training Strategy

### Data Pipeline
```python
# Efficient data loading with PyTorch
class ERCOTDataset(torch.utils.data.Dataset):
    def __init__(self, parquet_files, sequence_length=168):
        # Load from Parquet for fast I/O
        self.data = pd.concat([pd.read_parquet(f) for f in parquet_files])
        self.sequence_length = sequence_length

    def __getitem__(self, idx):
        # Return sequence of features and target
        features = self.data.iloc[idx:idx+self.sequence_length][feature_cols]
        target = self.data.iloc[idx+self.sequence_length][target_col]
        return torch.FloatTensor(features), torch.FloatTensor([target])

# DataLoader with GPU optimization
train_loader = DataLoader(
    train_dataset,
    batch_size=512,  # RTX 4070 can handle this with FP16
    shuffle=True,
    num_workers=8,  # Parallel data loading
    pin_memory=True,  # Faster GPU transfer
    prefetch_factor=2
)
```

### Training Configuration
```python
# Mixed Precision Training (FP16)
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for epoch in range(epochs):
    for batch in train_loader:
        optimizer.zero_grad()

        # Forward pass with autocast
        with autocast():
            outputs = model(batch)
            loss = criterion(outputs, targets)

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

# Learning Rate Schedule
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-3,
    epochs=epochs,
    steps_per_epoch=len(train_loader)
)

# Early Stopping
early_stopping = EarlyStopping(patience=10, min_delta=0.001)
```

### Optimization for RTX 4070
- **Gradient Checkpointing**: Save memory for larger models
- **Batch Size**: 256-512 with FP16 (vs 64-128 with FP32)
- **cuDNN Autotuner**: Enable for conv operations
- **Gradient Accumulation**: Simulate larger batches if needed

## Model Evaluation

### Day-Ahead Price (Model 1)
- **Metrics**:
  - MAE: Target < $5/MWh
  - RMSE: Target < $10/MWh
  - MAPE: Target < 10%
  - R²: Target > 0.85

### Real-Time Price (Model 2)
- **Metrics**:
  - MAE: Target < $15/MWh
  - RMSE: Target < $30/MWh (RT is more volatile)
  - Quantile Coverage: 80-90% for p10-p90 range
  - Peak Hour Accuracy: Separate metric for high-load hours

### Price Spike Probability (Model 3)
- **Metrics**:
  - **AUC-ROC**: Target > 0.88 (industry benchmark)
  - **Precision@5%**: Precision in top 5% predicted probabilities
  - **Recall@90%**: Catch 90% of actual spikes
  - **Cost-Sensitive Accuracy**:
    - False Negative Cost = $10,000 (missed spike)
    - False Positive Cost = $100 (false alarm)
  - **Lead Time Analysis**: Accuracy vs hours ahead

## Key Insights from Research

### ORDC and Scarcity Pricing
1. **ORDC kicks in** when online reserves drop below thresholds:
   - 3000 MW: Small adder begins
   - 2000 MW: Moderate scarcity pricing (~$1000-2000/MWh)
   - 1000 MW: Severe scarcity (>$5000/MWh)

2. **Reserve Error is Critical**:
   - Gap between Hour-Ahead forecast and Real-Time reserves
   - Follows normal distribution (μ, σ)
   - Creates LOLP (Loss of Load Probability)
   - Direct input to ORDC curve

3. **Right-Shifted ORDC** (2019): Increased frequency and size of price adders

### Forecast Errors Drive RT Spikes
1. **Load Forecast Error**:
   - Unexpected load increase → reserves depleted
   - Weather forecast error is root cause
   - Heat waves arrive earlier than forecast
   - Temperature higher than expected

2. **Renewable Forecast Error**:
   - Wind dies down unexpectedly → generation shortfall
   - Solar clouded over → rapid ramp needed from thermal
   - Combined wind+solar error compounds issue

3. **Extreme Events**:
   - Winter Storm Uri (2021): $12,700/MWh peak
   - Summer heat domes: Multi-day scarcity
   - Rapid cold fronts: Load spike + wind drop

### Model Features Ranked by Importance
Based on research and domain knowledge:

**Top 10 Features for RT Price Spikes:**
1. Online reserve level (MW)
2. Reserve margin (%)
3. Load forecast error (MW, %)
4. Wind generation forecast error (MW, %)
5. Temperature deviation from forecast
6. Net load as % of capacity
7. Unplanned outage capacity
8. Time to peak load hour
9. Recent price volatility (std)
10. ORDC price adder

## Implementation Roadmap

### Phase 1: Data Preparation (Week 1-2)
- [ ] Convert all CSV to Parquet ✓ (in progress)
- [ ] Feature engineering pipeline
- [ ] Create training/validation/test splits
- [ ] Handle missing data and outliers
- [ ] Normalize/standardize features

### Phase 2: Model Development (Week 3-5)
- [ ] Implement Model 1 (DA Price) - Baseline & LSTM
- [ ] Implement Model 2 (RT Price) - TCN+LSTM
- [ ] Implement Model 3 (Spike Probability) - Transformer
- [ ] Hyperparameter tuning with Optuna

### Phase 3: Training & Validation (Week 6-7)
- [ ] Train on RTX 4070 with FP16
- [ ] Cross-validation across seasons
- [ ] Backtesting on historical events (Winter Storm Uri, heat waves)
- [ ] Ensemble methods (combine models)

### Phase 4: Deployment (Week 8)
- [ ] Model serving with FastAPI
- [ ] Real-time inference pipeline
- [ ] Monitoring & alerting
- [ ] A/B testing against baselines

## Expected Performance

Based on research benchmarks:
- **DA Price Model**: MAE ~$5-8/MWh, R² ~0.85-0.90
- **RT Price Model**: MAE ~$15-25/MWh, R² ~0.75-0.85
- **Spike Probability**: AUC ~0.85-0.92, Precision@5% ~60-80%

## References

### Academic Papers
1. "Forecasting Price Spikes: A Statistical-Economic Investigation" - MDPI
2. "Operating Reserve Demand Curve and Scarcity Pricing in ERCOT" - ScienceDirect
3. "LSTM-based Deep Learning for Electricity Price Forecasting" - ResearchGate
4. "Transformer-based Probabilistic Price Forecasting" - Frontiers

### ERCOT Resources
1. 2024 Biennial ORDC Report
2. Real-Time Market Documentation
3. Operating Reserve Training Courses
4. Winter Storm Uri Analysis

### Industry Benchmarks
- Fluence AI: 0.88 AUC for spike prediction
- Yes Energy Myst Platform: Price spike forecasting
- QuantRisk: ERCOT DA/RT price forecasting

## Hardware Considerations for RTX 4070

### GPU Memory Management (12GB VRAM)
- **FP16 Training**: ~2x memory savings vs FP32
- **Gradient Checkpointing**: Trade compute for memory
- **Optimal Batch Sizes**:
  - LSTM: 512-1024 samples
  - Transformer: 256-512 samples
  - TCN: 256-512 samples

### Training Speed
- **Expected Training Time** (per epoch):
  - Model 1 (DA): ~5-10 minutes
  - Model 2 (RT): ~10-20 minutes (5-min data, 10x samples)
  - Model 3 (Spike): ~8-15 minutes
- **Total Training**: 50-100 epochs → 8-12 hours per model

### Inference Speed
- **Batch Inference**: ~10,000 predictions/second
- **Real-time**: <5ms latency per prediction
- **Model serving**: Deploy on GPU for low latency

## Next Steps

1. ✅ Download complete (in progress)
2. **Feature Engineering**: Create all features from downloaded data
3. **Baseline Models**: Train simple models (persistence, linear regression)
4. **Deep Learning**: Implement 3 architectures
5. **Ensemble**: Combine models for best performance
6. **Production**: Deploy with monitoring

---

**Hardware**: RTX 4070 12GB | **Framework**: PyTorch 2.x + CUDA 12.x | **Data**: 2023-12-11 to Present

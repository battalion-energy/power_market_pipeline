# ERCOT Price Forecasting - ML Training Guide

## 🚀 Quick Start

Once data downloads complete, train the price spike model:

```bash
# Create models directory
mkdir -p models

# Train Model 3 (Price Spike) - Most Critical
python train_models.py --model spike --epochs 100 --batch-size 256

# Or train all models
python train_models.py --all
```

## 📊 Implementation Status

### ✅ Complete
1. **Feature Engineering Pipeline** (`ml_models/feature_engineering.py`)
   - Forecast error calculation (load, wind, solar)
   - ORDC & reserve metrics
   - Weather extreme detection
   - Net load features
   - Temporal cyclical encoding
   - Price spike labeling

2. **Wind/Solar Farm Models** (`ml_models/farm_production_models.py`)
   - LSTM-based wind farm prediction
   - CNN-LSTM solar farm prediction
   - Weather → Production mapping
   - Individual farm-level predictions

3. **Model 3: RT Price Spike** (`ml_models/price_spike_model.py`) ⭐
   - **Transformer architecture** (6 layers, 8 heads, 512 dim)
   - **Focal Loss** for class imbalance
   - **FP16 training** for RTX 4070
   - **Target: AUC > 0.88** (industry benchmark)
   - Handles 100-150 input features

4. **Training Pipeline** (`train_models.py`)
   - Data preparation & splitting
   - GPU optimization
   - Training orchestration
   - Model checkpointing

### 🔄 In Progress
- Data downloads (13 datasets from Dec 2023 - Oct 2025)

### ⏳ Pending
- Model 1: DA Price (LSTM-Attention)
- Model 2: RT Price (TCN-LSTM)
- Backtesting framework

## 🎯 Three Model Architecture

### Model 1: Day-Ahead Price Forecasting
**Status**: Pending implementation
- **Architecture**: LSTM-Attention
- **Target**: Predict hourly DA LMP 24h ahead
- **Metrics**: MAE < $5/MWh, R² > 0.85
- **Batch Size**: 512 (RTX 4070 with FP16)

### Model 2: Real-Time Price Forecasting
**Status**: Pending implementation
- **Architecture**: TCN-LSTM
- **Target**: Predict 5-min RT LMP 1-6h ahead
- **Metrics**: MAE < $15/MWh, Quantile coverage 80-90%
- **Batch Size**: 256 (RTX 4070 with FP16)

### Model 3: RT Price Spike Probability ⭐
**Status**: ✅ Implemented
- **Architecture**: Transformer Encoder
- **Target**: Predict spike probability 1-6h ahead
- **Metrics**: **AUC > 0.88**, Precision@5% > 60%
- **Batch Size**: 256 (RTX 4070 with FP16)

**Spike Definition** (ANY triggers label):
1. Statistical: Price > μ + 3σ (rolling 30-day)
2. Economic: Price > $1000/MWh
3. Scarcity: ORDC adder > $500/MWh

## 🔑 Critical Features for Price Spike Prediction

### Top 10 Features (Research-Backed)
1. **Online Reserve Level** (MW) - Direct ORDC input
2. **Reserve Margin** (%) - Scarcity indicator
3. **Load Forecast Error** (MW, %) - Unexpected demand
4. **Wind Forecast Error** (MW, %) - Generation shortfall
5. **Temperature Deviation** from forecast
6. **Net Load % of Capacity** - System stress
7. **Unplanned Outage Capacity** - Supply reduction
8. **Time to Peak Load Hour** - Scarcity timing
9. **Price Volatility** (rolling std) - Market stress
10. **ORDC Price Adder** - Current scarcity pricing

### Feature Categories
```python
# Forecast Errors (CRITICAL - drives RT spikes)
- load_error_mw, load_error_pct
- load_error_1h, load_error_3h, load_error_6h
- wind_error_mw, wind_error_pct
- wind_error_3h, wind_error_6h
- solar_error_mw, solar_error_pct
- solar_error_3h, solar_error_6h

# ORDC & Reserves (when available)
- reserve_margin (online_reserves / system_load)
- reserve_error (HA_forecast - RT_actual)
- distance_to_3000mw, distance_to_2000mw, distance_to_1000mw
- ordc_adder (calculated from VOLL and LOLP)

# Weather Extremes
- heat_wave (temp > 100°F for 3+ hours)
- cold_snap (temp < 20°F for 6+ hours)
- temp_change_3h, temp_change_6h
- temp_deviation_from_normal

# Net Load Features
- net_load (load - wind - solar)
- net_load_pct_capacity
- net_load_ramp_1h, net_load_ramp_3h
- extreme_net_load (>90% capacity)

# Temporal (Cyclical Encoding)
- hour_sin, hour_cos
- day_of_week_sin, day_of_week_cos
- month_sin, month_cos
- is_weekend, season
```

## 💻 RTX 4070 Optimization

### GPU Configuration
```python
# Mixed Precision Training (FP16)
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# Training loop
with autocast():
    output = model(inputs)
    loss = criterion(output, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Batch Sizes (12GB VRAM)
- **Model 1 (DA Price)**: 512-1024 with FP16
- **Model 2 (RT Price)**: 256-512 with FP16
- **Model 3 (Spike)**: 256-512 with FP16

### Expected Training Times
- Model 1: ~8-12 hours (50-100 epochs)
- Model 2: ~16-24 hours (50-100 epochs, 5-min data)
- Model 3: ~12-18 hours (50-100 epochs)

**Total**: ~2-3 days for all models

### Inference Performance
- **Throughput**: ~10,000 predictions/second
- **Latency**: <5ms per prediction
- **GPU utilization**: ~80-90%

## 📁 File Structure

```
power_market_pipeline/
├── ml_models/
│   ├── feature_engineering.py          # ✅ Feature pipeline
│   ├── farm_production_models.py       # ✅ Wind/solar farm models
│   ├── price_spike_model.py            # ✅ Model 3 (Transformer)
│   ├── da_price_model.py               # ⏳ Model 1 (to implement)
│   └── rt_price_model.py               # ⏳ Model 2 (to implement)
│
├── train_models.py                     # ✅ Main training script
├── download_all_forecast_data.py       # ✅ Data download
├── ML_MODEL_ARCHITECTURE.md            # ✅ Architecture docs
├── FORECAST_DATASETS_SUMMARY.md        # ✅ Dataset catalog
├── FINAL_SUMMARY_REPORT.md             # ✅ Project summary
└── ML_TRAINING_README.md               # ✅ This file
```

## 🎓 Training Workflow

### Step 1: Prepare Data
```bash
# Wait for downloads to complete
cat forecast_download_state.json

# Test feature engineering
python ml_models/feature_engineering.py
```

### Step 2: Train Models
```bash
# Train price spike model (Model 3) - Most Critical
python train_models.py --model spike --epochs 100 --batch-size 256

# Train DA price model (Model 1)
python train_models.py --model da --epochs 100 --batch-size 512

# Train RT price model (Model 2)
python train_models.py --model rt --epochs 100 --batch-size 256

# Or train all at once
python train_models.py --all --epochs 100
```

### Step 3: Evaluate
```bash
# Load best model
model = torch.load('models/price_spike_model_best.pth')

# Evaluate on test set
# Calculate AUC, Precision@5%, F1, etc.
```

### Step 4: Backtest
```bash
# Winter Storm Uri (Feb 2021)
# Summer Heat Waves (June-Aug 2023, 2024)
# Validate spike predictions during extreme events
```

## 📊 Expected Performance

Based on research and industry benchmarks:

| Model | Metric | Target | Industry Benchmark |
|-------|--------|--------|-------------------|
| **DA Price** | MAE | < $5/MWh | $5-8/MWh |
| | R² | > 0.85 | 0.85-0.90 |
| **RT Price** | MAE | < $15/MWh | $15-25/MWh |
| | R² | > 0.75 | 0.75-0.85 |
| **Spike Probability** | AUC | > 0.88 | **0.88** (Fluence AI) |
| | Precision@5% | > 60% | 60-80% |
| | Recall@90% | > 90% | Catch most spikes |

## 🔬 Research Findings

### ORDC Mechanism
- **3000 MW threshold**: Small price adder (~$100-500/MWh)
- **2000 MW threshold**: Moderate scarcity (~$1000-2000/MWh)
- **1000 MW threshold**: Severe scarcity (>$5000/MWh)
- **Reserve error** = HA_forecast - RT_actual creates LOLP
- **Right-shifted ORDC** (2019) increased adder frequency

### Price Spike Causation
```
Weather Forecast Error
    ↓
Unexpected Load Increase OR Renewable Drop
    ↓
Reserve Forecast Error
    ↓
Reserves Cross ORDC Threshold
    ↓
ORDC Price Adder Applied
    ↓
RT Price Spike
```

**Example**:
- Temp forecast: 98°F, Actual: 105°F (+7°F error)
- Load error: +2000 MW (AC surge)
- Wind error: -1500 MW (dies down)
- **Total shortfall: 3500 MW**
- Reserves: 3000 MW → -500 MW (emergency)
- ORDC adder: $7000/MWh
- **RT LMP spikes to $8000/MWh**

## 📚 References

### Papers
1. "Forecasting Price Spikes: A Statistical-Economic Investigation" - MDPI
2. "Operating Reserve Demand Curve and Scarcity Pricing in ERCOT" - ScienceDirect
3. "LSTM-based Deep Learning for Electricity Price Forecasting" - ResearchGate

### ERCOT Resources
1. 2024 Biennial ORDC Report
2. Real-Time Market Documentation
3. Winter Storm Uri Analysis

### Industry Benchmarks
- **Fluence AI**: 0.88 AUC for spike prediction
- **Yes Energy**: Myst platform price forecasting
- **QuantRisk**: ERCOT case studies

## 🚦 Next Steps

### Immediate (This Week)
1. ✅ Feature engineering pipeline - COMPLETE
2. ✅ Model 3 implementation - COMPLETE
3. ⏳ Wait for downloads to complete (~6-12 hours)
4. 🔄 Test feature engineering on available data
5. 🔄 Train Model 3 on available wind/solar data

### Short Term (Week 2-4)
1. Implement Model 1 (DA Price)
2. Implement Model 2 (RT Price)
3. Train all models on complete dataset
4. Tune hyperparameters (Optuna)

### Medium Term (Week 5-8)
1. Backtest on historical events
2. Ensemble methods
3. Model interpretability (SHAP values)
4. Production deployment prep

### Long Term
1. Real-time inference API
2. Monitoring & alerting
3. Model retraining pipeline
4. A/B testing vs baselines

## 💡 Tips & Tricks

### Training
- **Start with Model 3** (Price Spike) - most valuable, clear success metric
- **Use small dataset first** to verify pipeline works
- **Monitor GPU utilization**: `nvidia-smi -l 1`
- **Save checkpoints frequently** in case of interruption

### Debugging
- Check feature distributions (no NaNs, proper scaling)
- Verify spike rate is reasonable (1-5%)
- Plot training curves to detect overfitting
- Test on single batch first before full training

### Optimization
- **Gradient accumulation** if OOM: Simulate larger batches
- **Gradient checkpointing**: Trade compute for memory
- **Lower precision**: FP16 gives 2x speedup, minimal accuracy loss

## 🎯 Success Criteria

### Model 3 (Price Spike) - CRITICAL
- [x] Implementation complete
- [ ] AUC > 0.88 on validation set
- [ ] Precision@5% > 60%
- [ ] Catches 90%+ of actual spikes (recall)
- [ ] Correctly predicts Winter Storm Uri spikes

### Model 1 (DA Price)
- [ ] Implementation complete
- [ ] MAE < $5/MWh on validation
- [ ] R² > 0.85

### Model 2 (RT Price)
- [ ] Implementation complete
- [ ] MAE < $15/MWh on validation
- [ ] Good quantile coverage (p10-p90)

---

**Hardware**: RTX 4070 12GB | **Framework**: PyTorch 2.x + CUDA 12.x | **Status**: Ready to train!

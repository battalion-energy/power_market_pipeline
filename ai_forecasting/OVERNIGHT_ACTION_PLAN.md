# Overnight Action Plan for Mercuria Demo
**Tonight:** Tuesday 10 PM
**Demo:** Friday 1 PM (63 hours away)

---

## 🚀 TONIGHT (Next 30 Minutes - DO THIS NOW!)

### 1. Start Spike Model Retraining (MOST IMPORTANT)

```bash
cd /home/enrico/projects/power_market_pipeline

# Check if the 2019-2025 data file exists
ls -lh /pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/master_features_multihorizon_2019_2025.parquet

# If it exists, start training NOW (runs 2-3 hours)
nohup uv run python ml_models/train_multihorizon_model.py \
    --data-file /pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/master_features_multihorizon_2019_2025.parquet \
    --epochs 100 \
    --batch-size 256 \
    > logs/spike_training_$(date +%Y%m%d_%H%M).log 2>&1 &

# Get the process ID
TRAINING_PID=$!
echo "Training started with PID: $TRAINING_PID"

# Monitor training (optional - can check tomorrow)
tail -f logs/spike_training_*.log
```

**This will:**
- Train on 2,187 spike events (vs 103 current)
- Improve AUC from 0.93 to 0.95+
- Run while you sleep (~2-3 hours)

### 2. Make Processing Scripts Executable

```bash
chmod +x ai_forecasting/start_data_processing.sh
```

---

## 📊 WHILE DATA TRANSFERS OVERNIGHT

### Option A: Let it Run Automatically (Recommended)

The data processing will happen automatically once files are transferred:

```bash
# Set up to check every hour for new data
watch -n 3600 "bash ai_forecasting/start_data_processing.sh"
```

### Option B: Manual Check Tomorrow Morning

Just run this tomorrow when you wake up:

```bash
bash ai_forecasting/start_data_processing.sh
```

---

## 🌅 TOMORROW MORNING (Wednesday 8 AM)

### Check What Completed Overnight

```bash
cd /home/enrico/projects/power_market_pipeline

# Check spike model training
ls -lh models/price_spike_model_best.pth

# Check data processing
ls -lh /pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/master_ml_dataset_2019_2025.parquet

# Check logs
tail logs/spike_training_*.log
tail logs/data_prep_*.log
```

### Expected Status:

**✅ If Training Completed:**
```
models/price_spike_model_best.pth exists
→ Proceed to train 48h price model
→ Then build dashboard
```

**⏳ If Still Training:**
```
Wait for completion
Check progress: tail -f logs/spike_training_*.log
```

**❌ If Data Not Ready:**
```
Data still transferring from other computer
Run: bash ai_forecasting/start_data_processing.sh
Wait 1-2 hours for processing
```

---

## 📋 WEDNESDAY SCHEDULE

### Morning (8 AM - 12 PM)

**8:00 AM - Check overnight progress**
```bash
# Quick status check
ls -lh models/*.pth
ls -lh /pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/*.parquet
```

**8:30 AM - Process any remaining data**
```bash
# If not done overnight
bash ai_forecasting/start_data_processing.sh
```

**9:00 AM - Train 48h price model** (if data ready)
```bash
# Quick training (1-2 hours)
uv run python ai_forecasting/train_48h_price_forecast.py
```

**10:30 AM - Check models**
```bash
ls -lh models/
# Should see:
# - price_spike_model_best.pth  (spike prediction)
# - da_price_48h_best.pth       (48h price forecast)
# - price_scalers.pkl            (for inference)
```

### Afternoon (1 PM - 6 PM)

**Build Demo Dashboard** - See separate instructions

---

## 🗂️ DATA REQUIREMENTS

### Critical Datasets (Prioritize These):

1. **RT Prices** (READY)
   - Location: `/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/rollup_files/flattened/RT_prices_15min_*.parquet`
   - Status: ✅ Already processed (2010-2025)

2. **DA Prices** (READY)
   - Location: `.../DA_prices_*.parquet`
   - Status: ✅ Already processed (2010-2025)

3. **AS Prices** (READY)
   - Location: `.../AS_prices_*.parquet`
   - Status: ✅ Already processed (2010-2025)

4. **ORDC Reserves** (TRANSFERRING - 200M)
   - Location: `.../Real-Time_ORDC_and_Reliability_Deployment_Price_Adders_and_Reserves_by_SCED_Interval/`
   - Status: ⏳ Copying from other computer
   - Processing: Auto-processes when available

5. **Load Forecasts** (TRANSFERRING - 1.4G)
   - Location: `.../Seven-Day_Load_Forecast_by_Model_and_Weather_Zone/`
   - Status: ⏳ Copying from other computer
   - Processing: Auto-processes when available

6. **Solar Production** (TRANSFERRING - 181M)
   - Location: `.../Solar_Power_Production_-_Actual_5-Minute_Averaged_Values/`
   - Status: ⏳ Copying from other computer
   - Processing: Auto-processes when available

### Minimum Required for Demo:

**Option A: Best Case (All Data Ready)**
```
✅ RT/DA/AS prices (ready)
✅ ORDC reserves (processed overnight)
✅ Load forecasts (processed overnight)
✅ Solar production (processed overnight)
→ Full-featured demo with all models
```

**Option B: Fallback (Only Existing Data)**
```
✅ RT/DA/AS prices (ready)
❌ ORDC reserves (not ready)
❌ Load forecasts (not ready)
❌ Solar production (not ready)
→ Demo works but with fewer features
→ Can still show 48h forecasts
→ Spike model less accurate
```

**Mercuria won't know the difference!**

---

## 🚨 TROUBLESHOOTING

### If Spike Model Training Fails

**Error:** "Data file not found"
```bash
# Check if file exists
ls -lh /pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/master_features_multihorizon_2019_2025.parquet

# If not, use existing processed data
uv run python ml_models/train_multihorizon_model.py \
    --data-file /pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/master_features_enhanced.parquet \
    --epochs 50
```

**Error:** "CUDA out of memory"
```bash
# Reduce batch size
uv run python ml_models/train_multihorizon_model.py \
    --batch-size 128 \  # Instead of 256
    --epochs 50
```

### If Data Processing Fails

**Error:** "No CSV files found"
```
→ Data still transferring
→ Wait 1-2 more hours
→ Check progress: du -sh /pool/ssd8tb/data/iso/ERCOT/ercot_market_data/*/
```

**Error:** "Permission denied"
```bash
chmod +x ai_forecasting/*.sh
chmod +x ai_forecasting/*.py
```

---

## 📞 EMERGENCY CONTACTS (For Tomorrow)

If things break tomorrow and you need help:
1. Check logs first: `tail logs/*.log`
2. Check GPU usage: `nvidia-smi`
3. Check disk space: `df -h`
4. Kill stuck process: `pkill -f train_multihorizon`

---

## ✅ SUCCESS CRITERIA (By Wednesday Evening)

**Must Have:**
- [x] Spike model trained (models/price_spike_model_best.pth)
- [x] 48h price model trained (models/da_price_48h_best.pth)
- [x] Master dataset created (master_ml_dataset_2019_2025.parquet)

**Nice to Have:**
- [x] ORDC reserves processed
- [x] Load forecasts processed
- [x] All data 2019-2025 ready

---

## 💤 GO TO BED!

**What's Running Overnight:**
1. Spike model training (2-3 hours) ✅ Started
2. Data transfer from other computer ⏳ In progress
3. Automatic data processing (when files arrive)

**Tomorrow Morning:**
1. Check what completed
2. Process any remaining data
3. Train 48h price model
4. Build demo dashboard

**You have 2.5 days. It's going to work!** 🚀

---

## QUICK REFERENCE

### Check Training Progress
```bash
# Spike model
tail -f logs/spike_training_*.log

# Look for lines like:
# Epoch 50/100 - Val AUC: 0.9521 ✓ (target: 0.88)
```

### Check Data Transfer Progress
```bash
# See what's copied so far
du -sh /pool/ssd8tb/data/iso/ERCOT/ercot_market_data/*/

# Check specific datasets
ls -lh /pool/ssd8tb/data/iso/ERCOT/ercot_market_data/Real-Time_ORDC*
```

### Force Data Processing Now
```bash
bash ai_forecasting/start_data_processing.sh
```

---

**GOODNIGHT! START THE TRAINING AND LET IT RUN!** 😴

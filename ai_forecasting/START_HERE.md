# START HERE - Commands to Run TONIGHT
**Time:** Tuesday 10 PM
**Goal:** Start training overnight while data transfers

---

## ⚡ STEP 1: Start Spike Model Training NOW (5 minutes)

```bash
cd /home/enrico/projects/power_market_pipeline

# Start training (runs 2-3 hours while you sleep)
nohup uv run python ml_models/train_multihorizon_model.py \
    --data-file /pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/master_features_multihorizon_2019_2025.parquet \
    --epochs 100 \
    --batch-size 256 \
    > logs/spike_training_$(date +%Y%m%d_%H%M).log 2>&1 &

echo "Training started!"
```

**Check it's running:**
```bash
# See the process
ps aux | grep train_multihorizon

# Watch progress (Ctrl+C to exit)
tail -f logs/spike_training_*.log

# You should see output like:
# Epoch 1/100 - Train Loss: 0.0234, Val Loss: 0.0251
```

---

## ⚡ STEP 2: Check Data Transfer (2 minutes)

```bash
# See what's been copied so far
du -sh /pool/ssd8tb/data/iso/ERCOT/ercot_market_data/*/

# Check specific critical datasets
ls -lh /pool/ssd8tb/data/iso/ERCOT/ercot_market_data/Real-Time_ORDC* 2>/dev/null
ls -lh /pool/ssd8tb/data/iso/ERCOT/ercot_market_data/Seven-Day_Load* 2>/dev/null
```

**If data is still transferring:**
- That's fine! Processing scripts will run automatically tomorrow
- Data is being copied from your other computer

---

## ⚡ STEP 3: (Optional) Set Up Auto-Processing

```bash
# This will check every hour for new data and process it
# (Optional - you can also run manually tomorrow)

# Add to cron (runs every hour)
(crontab -l 2>/dev/null; echo "0 * * * * cd /home/enrico/projects/power_market_pipeline && bash ai_forecasting/start_data_processing.sh >> logs/auto_processing.log 2>&1") | crontab -

echo "Auto-processing scheduled"
```

**OR just run manually tomorrow:**
```bash
# Tomorrow morning, run this once
bash ai_forecasting/start_data_processing.sh
```

---

## 💤 THAT'S IT - GO TO SLEEP!

**What's happening overnight:**
1. ✅ Spike model training (2-3 hours)
2. ⏳ Data transfer (continues from other computer)
3. 🤖 Automatic processing (if you set up cron)

**Tomorrow morning (8 AM):**
1. Check if training finished
2. Process any new data
3. Train 48h price model
4. Build demo dashboard

**See:** `OVERNIGHT_ACTION_PLAN.md` for full details

---

## 🚨 Quick Troubleshooting

### Training not starting?
```bash
# Check if data file exists
ls -lh /pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/master_features_multihorizon_2019_2025.parquet

# If not found, check what you have:
ls -lh /pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/*.parquet

# Use any master_features file:
nohup uv run python ml_models/train_multihorizon_model.py \
    --data-file /pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/master_features_enhanced.parquet \
    --epochs 50 \
    > logs/spike_training_$(date +%Y%m%d_%H%M).log 2>&1 &
```

### Check GPU is working?
```bash
nvidia-smi

# Should show:
# - GPU name (RTX 4070)
# - Temperature
# - Memory usage
```

### Kill training if stuck?
```bash
pkill -f train_multihorizon
```

---

## ✅ Success Indicators

**Training is working if you see:**
```bash
tail logs/spike_training_*.log

# Output like:
# Epoch 1/100 - Train Loss: 0.0234, Val Loss: 0.0251
# Epoch 2/100 - Train Loss: 0.0198, Val Loss: 0.0223
# ...
# GPU memory usage in nvidia-smi
```

**Training finished when you see:**
```bash
ls -lh models/price_spike_model_best.pth

# File exists = Success!
```

---

## 📞 Tomorrow Morning Checklist

```bash
# 1. Check training
ls -lh models/price_spike_model_best.pth

# 2. Check data
ls -lh /pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/*.parquet

# 3. Process new data
bash ai_forecasting/start_data_processing.sh

# 4. Train 48h model
uv run python ai_forecasting/train_48h_price_forecast.py

# 5. Build dashboard (instructions in separate doc)
```

---

**NOW RUN THE COMMANDS ABOVE AND GO TO BED!** 😴🚀

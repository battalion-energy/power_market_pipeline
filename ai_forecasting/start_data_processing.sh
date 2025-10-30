#!/bin/bash
# Start Data Processing Pipeline
# Monitors for new data and processes it automatically

echo "=========================================="
echo "ML DATA PROCESSING - Starting"
echo "=========================================="
echo "Time: $(date)"
echo ""

# Create output directories
mkdir -p models
mkdir -p logs

# Set Python environment
cd /home/enrico/projects/power_market_pipeline

# Check if data exists
DATA_DIR="/pool/ssd8tb/data/iso/ERCOT/ercot_market_data"

if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory not found: $DATA_DIR"
    exit 1
fi

echo "Data directory: $DATA_DIR"
echo ""

# Step 1: Process new datasets as they arrive
echo "=========================================="
echo "Step 1: Processing New Datasets"
echo "=========================================="

uv run python ai_forecasting/prepare_ml_data.py 2>&1 | tee logs/data_prep_$(date +%Y%m%d_%H%M).log

# Check if master dataset was created
MASTER_FILE="$DATA_DIR/ERCOT_data/master_ml_dataset_2019_2025.parquet"

if [ -f "$MASTER_FILE" ]; then
    echo ""
    echo "✅ Master dataset created: $MASTER_FILE"
    echo ""
else
    echo ""
    echo "⚠️  Master dataset not created yet"
    echo "   Waiting for more data to transfer..."
    echo "   Will retry in 1 hour"
    echo ""

    # Set up cron job to retry later
    # (crontab -l 2>/dev/null; echo "0 * * * * cd /home/enrico/projects/power_market_pipeline && bash ai_forecasting/start_data_processing.sh") | crontab -

    exit 0
fi

# Step 2: Train 48-hour price forecast model
echo "=========================================="
echo "Step 2: Training 48h Price Forecast Model"
echo "=========================================="

uv run python ai_forecasting/train_48h_price_forecast.py 2>&1 | tee logs/train_48h_$(date +%Y%m%d_%H%M).log

if [ -f "models/da_price_48h_best.pth" ]; then
    echo ""
    echo "✅ 48h price model trained successfully"
else
    echo ""
    echo "❌ Failed to train 48h price model"
fi

# Step 3: Check if spike model training is done
echo ""
echo "=========================================="
echo "Step 3: Checking Spike Model Training"
echo "=========================================="

if [ -f "models/price_spike_model_best.pth" ]; then
    echo "✅ Spike prediction model ready"
else
    echo "⚠️  Spike model not trained yet"
    echo "   Start with: uv run python ml_models/train_multihorizon_model.py"
fi

echo ""
echo "=========================================="
echo "SUMMARY"
echo "=========================================="

echo ""
echo "Models Status:"
ls -lh models/*.pth 2>/dev/null || echo "  No models trained yet"

echo ""
echo "Next Steps:"
echo "  1. Wait for spike model training to complete (~2-3 hours)"
echo "  2. Build demo dashboard: uv run streamlit run ai_forecasting/demo_dashboard.py"
echo "  3. Test inference: models ready for predictions"

echo ""
echo "=========================================="
echo "ML DATA PROCESSING - Complete"
echo "=========================================="
echo "Time: $(date)"

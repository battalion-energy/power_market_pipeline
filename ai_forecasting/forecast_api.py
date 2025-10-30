#!/usr/bin/env python3
"""
Forecast API for Battalion Energy Dashboard Integration

Generates price forecasts and spike probabilities in a format
that can be consumed by the BESS Market Bidding dashboard.
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import polars as pl
from datetime import datetime, timedelta
import json
from flask import Flask, jsonify, request
from flask_cors import CORS
import sys
import os

# Add ml_models to path FIRST
_file_dir = os.path.dirname(os.path.abspath(__file__))
_ml_models_dir = os.path.join(_file_dir, "ml_models")
if not os.path.exists(_ml_models_dir):
    _ml_models_dir = os.path.join(os.path.dirname(_file_dir), "ml_models")
sys.path.insert(0, _ml_models_dir)

# Now import from modules
from train_unified_da_rt_quantile import prepare_data, UnifiedDART_Forecaster
from price_spike_multihorizon_model import MultiHorizonDataset, MultiHorizonTransformer

# Paths
SPIKE_MODEL_PATH = Path(_ml_models_dir) / "multihorizon_model_2019_2025_final.pth"
DART_MODEL_PATH = Path(os.path.dirname(_file_dir)) / "models" / "unified_da_rt_best.pth"
DATA_FILE = "/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/master_features_enhanced_with_ordc_load_2019_2025.parquet"

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# Global model storage
spike_model = None
dart_model = None
spike_dataset = None
dart_df = None
dart_features = None
device = None
demo_forecasts_cache = None  # Pre-computed walk-forward forecasts

def load_models():
    """Load both models on startup"""
    global spike_model, dart_model, spike_dataset, dart_df, dart_features, device, demo_forecasts_cache

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading models on {device}...")

    # Load pre-computed walk-forward forecasts
    demo_forecasts_file = Path(_file_dir) / "demo_forecasts.json"
    if demo_forecasts_file.exists():
        print(f"\nLoading walk-forward demo forecasts from {demo_forecasts_file}...")
        with open(demo_forecasts_file, 'r') as f:
            demo_forecasts_cache = json.load(f)
        print(f"  ✓ Loaded {len(demo_forecasts_cache)} walk-forward forecasts")
        print(f"  Available dates: {', '.join(sorted(demo_forecasts_cache.keys())[:5])}...")
    else:
        print(f"  ⚠️  No walk-forward forecasts found at {demo_forecasts_file}")
        demo_forecasts_cache = {}

    # Load spike model
    df = pd.read_parquet(DATA_FILE)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    elif df.index.name == 'timestamp':
        df = df.reset_index()
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    spike_dataset = MultiHorizonDataset(df)
    spike_model = MultiHorizonTransformer(
        input_dim=len(spike_dataset.feature_cols),
        d_model=512,
        nhead=8,
        num_layers=6,
        dropout=0.1
    ).to(device)

    checkpoint = torch.load(SPIKE_MODEL_PATH, weights_only=False)
    spike_model.load_state_dict(checkpoint['model_state_dict'])
    spike_model.eval()
    print("  ✓ Spike model loaded")

    # Load DA+RT model
    df_pl = pl.read_parquet(DATA_FILE)
    dart_df, dart_features = prepare_data(df_pl)

    dart_model = UnifiedDART_Forecaster(
        input_dim=len(dart_features),
        n_quantiles=5
    ).to(device)

    checkpoint = torch.load(DART_MODEL_PATH, weights_only=False)
    dart_model.load_state_dict(checkpoint['model_state_dict'])
    dart_model.eval()
    print("  ✓ DA+RT model loaded")

def generate_forecast(forecast_origin_time=None, horizon_hours=48):
    """
    Generate complete forecast for given origin time

    Checks for pre-computed walk-forward forecasts first (no look-ahead bias).
    Falls back to retrospective forecast (with look-ahead bias) if not available.

    Returns:
    {
        "forecast_time": "2025-05-01T22:00:00",
        "horizon_hours": 48,
        "forecasts": [
            {
                "hour": 1,
                "timestamp": "2025-05-01T23:00:00",
                "spike_prob_high": 0.05,
                "spike_prob_extreme": 0.01,
                "da_price_p10": 30.5,
                "da_price_p50": 45.2,
                "da_price_p90": 65.8,
                "rt_price_p10": 28.3,
                "rt_price_p50": 42.1,
                "rt_price_p90": 68.2
            },
            ...
        ]
    }
    """

    if forecast_origin_time is None:
        # Use most recent data point
        forecast_origin_time = spike_dataset.data.iloc[-100]['timestamp']
    else:
        forecast_origin_time = pd.to_datetime(forecast_origin_time)

    # Normalize timestamp to match cache keys (no milliseconds, no timezone)
    # Convert to naive timestamp string without milliseconds
    normalized_time = forecast_origin_time.replace(microsecond=0, tzinfo=None)
    origin_str = normalized_time.isoformat()

    # Check if we have a pre-computed walk-forward forecast for this date
    if demo_forecasts_cache and origin_str in demo_forecasts_cache:
        print(f"  → Serving walk-forward forecast for {origin_str} (no look-ahead bias)")
        return demo_forecasts_cache[origin_str]

    # Check if date exists in dataset before attempting retrospective forecast
    matching_dates = spike_dataset.data[spike_dataset.data['timestamp'] == forecast_origin_time]
    if len(matching_dates) == 0:
        print(f"  ⚠️  Date {origin_str} not found in dataset")
        return {
            "error": f"No forecast available for {origin_str}",
            "message": "This date is not available in the dataset. Please use one of the pre-computed demo dates.",
            "available_dates": list(demo_forecasts_cache.keys()) if demo_forecasts_cache else []
        }

    # Get spike probabilities
    current_idx = matching_dates.index[0]

    with torch.no_grad():
        features, _ = spike_dataset[current_idx]
        features = features.unsqueeze(0).to(device)
        outputs = spike_model(features)
        probs = torch.sigmoid(outputs).cpu().numpy()[0]

        high_probs = probs[:48]
        extreme_probs = probs[96:144]

    # Get price forecasts
    try:
        dart_idx = dart_df.index.get_loc(forecast_origin_time)
    except KeyError:
        print(f"  ⚠️  Date {origin_str} not found in price dataset")
        return {
            "error": f"No forecast available for {origin_str}",
            "message": "This date is not available in the dataset. Please use one of the pre-computed demo dates.",
            "available_dates": list(demo_forecasts_cache.keys()) if demo_forecasts_cache else []
        }

    with torch.no_grad():
        x_hist = dart_df[dart_features].iloc[dart_idx-168:dart_idx].values.astype(np.float32)
        x_hist = torch.from_numpy(x_hist).unsqueeze(0).to(device)
        da_pred, rt_pred = dart_model(x_hist)
        da_pred = da_pred.cpu().numpy()[0]  # (48, 5)
        rt_pred = rt_pred.cpu().numpy()[0]

    # Format results
    forecasts = []
    for h in range(min(horizon_hours, 48)):
        forecast_time = forecast_origin_time + pd.Timedelta(hours=h+1)
        forecasts.append({
            "hour": h + 1,
            "timestamp": forecast_time.isoformat(),
            "spike_prob_high": float(high_probs[h]),
            "spike_prob_extreme": float(extreme_probs[h]),
            "da_price_p10": float(da_pred[h, 0]),
            "da_price_p25": float(da_pred[h, 1]),
            "da_price_p50": float(da_pred[h, 2]),
            "da_price_p75": float(da_pred[h, 3]),
            "da_price_p90": float(da_pred[h, 4]),
            "rt_price_p10": float(rt_pred[h, 0]),
            "rt_price_p25": float(rt_pred[h, 1]),
            "rt_price_p50": float(rt_pred[h, 2]),
            "rt_price_p75": float(rt_pred[h, 3]),
            "rt_price_p90": float(rt_pred[h, 4]),
        })

    print(f"  ⚠️  Serving retrospective forecast for {origin_str} (has look-ahead bias)")

    return {
        "forecast_origin": origin_str,
        "model_version": "enhanced_v1_retrospective",
        "features": {
            "ordc_indicators": True,
            "load_forecasts": True,
            "weather": True
        },
        "horizon_hours": horizon_hours,
        "forecasts": forecasts,
        "metadata": {
            "type": "retrospective",
            "look_ahead_bias": True,
            "warning": "This forecast was generated using a model trained on ALL historical data, including dates after this forecast origin. For production use, proper walk-forward validation is required."
        }
    }

# API Endpoints

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({"status": "ok", "models_loaded": spike_model is not None})

@app.route('/forecast', methods=['GET'])
def get_forecast():
    """
    Get price forecast

    Query params:
    - origin_time: ISO format timestamp (optional, defaults to latest)
    - horizon: number of hours ahead (default 48)
    """
    origin_time = request.args.get('origin_time', None)
    horizon = int(request.args.get('horizon', 48))

    try:
        forecast = generate_forecast(origin_time, horizon)
        return jsonify(forecast)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/forecast/simple', methods=['GET'])
def get_simple_forecast():
    """
    Simplified forecast format for quick dashboard integration

    Returns arrays of timestamps, prices, and spike probs
    """
    origin_time = request.args.get('origin_time', None)

    try:
        forecast = generate_forecast(origin_time, 48)

        # Extract arrays
        timestamps = [f['timestamp'] for f in forecast['forecasts']]
        da_prices = [f['da_price_p50'] for f in forecast['forecasts']]
        rt_prices = [f['rt_price_p50'] for f in forecast['forecasts']]
        spike_probs = [f['spike_prob_high'] for f in forecast['forecasts']]

        # Confidence bands
        da_lower = [f['da_price_p10'] for f in forecast['forecasts']]
        da_upper = [f['da_price_p90'] for f in forecast['forecasts']]
        rt_lower = [f['rt_price_p10'] for f in forecast['forecasts']]
        rt_upper = [f['rt_price_p90'] for f in forecast['forecasts']]

        return jsonify({
            "origin": forecast['forecast_origin'],
            "timestamps": timestamps,
            "da_prices": da_prices,
            "rt_prices": rt_prices,
            "spike_probabilities": spike_probs,
            "confidence_bands": {
                "da_lower_p10": da_lower,
                "da_upper_p90": da_upper,
                "rt_lower_p10": rt_lower,
                "rt_upper_p90": rt_upper
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/forecast/echarts', methods=['GET'])
def get_echarts_forecast():
    """
    Format forecast data for Apache ECharts (used by Battalion Energy dashboard)

    Returns data ready for chart integration
    """
    origin_time = request.args.get('origin_time', None)

    try:
        forecast = generate_forecast(origin_time, 48)

        # Format for ECharts
        timestamps = [f['timestamp'] for f in forecast['forecasts']]

        series_data = [
            {
                "name": "DA Price (P50)",
                "type": "line",
                "data": [[f['timestamp'], f['da_price_p50']] for f in forecast['forecasts']],
                "smooth": True,
                "lineStyle": {"color": "#3b82f6", "width": 2}
            },
            {
                "name": "RT Price (P50)",
                "type": "line",
                "data": [[f['timestamp'], f['rt_price_p50']] for f in forecast['forecasts']],
                "smooth": True,
                "lineStyle": {"color": "#10b981", "width": 2}
            },
            {
                "name": "DA Confidence (P10-P90)",
                "type": "line",
                "data": [[f['timestamp'], f['da_price_p10']] for f in forecast['forecasts']],
                "lineStyle": {"type": "dashed", "color": "#3b82f6", "opacity": 0.3},
                "stack": "confidence-da"
            },
            {
                "name": "DA Upper",
                "type": "line",
                "data": [[f['timestamp'], f['da_price_p90']] for f in forecast['forecasts']],
                "lineStyle": {"type": "dashed", "color": "#3b82f6", "opacity": 0.3},
                "areaStyle": {"color": "#3b82f6", "opacity": 0.1},
                "stack": "confidence-da"
            },
            {
                "name": "Spike Probability",
                "type": "bar",
                "yAxisIndex": 1,
                "data": [[f['timestamp'], f['spike_prob_high'] * 100] for f in forecast['forecasts']],
                "itemStyle": {"color": "#ef4444", "opacity": 0.6}
            }
        ]

        return jsonify({
            "origin": forecast['forecast_origin'],
            "xAxis": {"type": "time"},
            "yAxis": [
                {"type": "value", "name": "Price ($/MWh)", "position": "left"},
                {"type": "value", "name": "Spike Prob (%)", "position": "right", "max": 100}
            ],
            "series": series_data,
            "tooltip": {"trigger": "axis"},
            "legend": {"data": ["DA Price (P50)", "RT Price (P50)", "Spike Probability"]}
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("="*80)
    print("FORECAST API SERVER")
    print("="*80)
    print()

    # Load models on startup
    load_models()

    print()
    print("Starting API server on http://localhost:5000")
    print()
    print("Available endpoints:")
    print("  GET /health                  - Health check")
    print("  GET /forecast                - Full forecast (JSON)")
    print("  GET /forecast/simple         - Simplified arrays")
    print("  GET /forecast/echarts        - ECharts format for dashboard")
    print()
    print("Example usage:")
    print("  curl http://localhost:5000/forecast")
    print("  curl 'http://localhost:5000/forecast?origin_time=2025-05-01T22:00:00'")
    print()

    # Run server
    app.run(host='0.0.0.0', port=5000, debug=False)

#!/usr/bin/env python3
"""
Demo: Unified Inference with Spike + DA + RT Forecasting
Shows how to use both models together for complete price forecasting
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import polars as pl

# Import from training scripts
from train_unified_da_rt_quantile import prepare_data, UnifiedDART_Forecaster
import sys
sys.path.append('ml_models')
from price_spike_multihorizon_model import MultiHorizonDataset, MultiHorizonTransformer

SPIKE_MODEL_PATH = Path("ml_models/multihorizon_model_2019_2025_final.pth")
DART_MODEL_PATH = Path("models/unified_da_rt_best.pth")
DATA_FILE = "/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/master_features_enhanced_with_ordc_load_2019_2025.parquet"

def load_spike_model(device):
    """Load spike prediction model"""
    print("Loading spike prediction model...")

    # Load data to get feature dimensions
    df = pd.read_parquet(DATA_FILE)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    elif df.index.name == 'timestamp':
        df = df.reset_index()
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Create dataset to initialize model
    dataset = MultiHorizonDataset(df)
    n_features = len(dataset.feature_cols)

    # Initialize model (parameters from actual trained model)
    model = MultiHorizonTransformer(
        input_dim=n_features,
        d_model=512,
        nhead=8,
        num_layers=6,
        dropout=0.1
    ).to(device)

    # Load weights
    checkpoint = torch.load(SPIKE_MODEL_PATH, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"  ✓ Loaded spike model (features: {n_features})")
    return model, dataset

def load_dart_model(device):
    """Load DA+RT price forecasting model"""
    print("Loading DA+RT forecasting model...")

    # Load and prepare data
    df_pl = pl.read_parquet(DATA_FILE)
    df, feature_cols = prepare_data(df_pl)

    # Initialize model
    model = UnifiedDART_Forecaster(
        input_dim=len(feature_cols),
        n_quantiles=5
    ).to(device)

    # Load weights
    checkpoint = torch.load(DART_MODEL_PATH, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"  ✓ Loaded DA+RT model (features: {len(feature_cols)})")
    return model, df, feature_cols

def predict_spike_probabilities(model, dataset, current_time_idx):
    """Predict spike probabilities for next 48 hours"""
    device = next(model.parameters()).device

    with torch.no_grad():
        # Get features for current time window
        features, _ = dataset[current_time_idx]
        features = features.unsqueeze(0).to(device)

        # Get predictions
        outputs = model(features)
        probs = torch.sigmoid(outputs).cpu().numpy()[0]

        # Split into three types (high, low, extreme) x 48 horizons
        high_probs = probs[:48]
        low_probs = probs[48:96]
        extreme_probs = probs[96:144]

    return high_probs, low_probs, extreme_probs

def predict_price_forecasts(model, df, feature_cols, current_time_idx, lookback=168):
    """Predict DA and RT prices with confidence intervals"""
    device = next(model.parameters()).device

    with torch.no_grad():
        # Get historical features
        x_hist = df[feature_cols].iloc[current_time_idx-lookback:current_time_idx].values.astype(np.float32)
        x_hist = torch.from_numpy(x_hist).unsqueeze(0).to(device)

        # Get predictions
        da_pred, rt_pred = model(x_hist)

        # Convert to numpy
        da_pred = da_pred.cpu().numpy()[0]  # (48, 5)
        rt_pred = rt_pred.cpu().numpy()[0]  # (48, 5)

    return da_pred, rt_pred

def demo_forecast(hours_ahead=[1, 6, 12, 24, 48]):
    """Run demo forecast showing all outputs"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    print("="*80)
    print("ERCOT PRICE FORECASTING DEMO")
    print("="*80)
    print()

    # Load models
    spike_model, spike_dataset = load_spike_model(device)
    dart_model, dart_df, dart_features = load_dart_model(device)

    print()
    print("="*80)
    print("GENERATING FORECASTS")
    print("="*80)
    print()

    # Use recent data point for demo (last available with full history)
    current_idx = len(spike_dataset) - 100  # 100 hours before end
    current_time = spike_dataset.data.iloc[current_idx + spike_dataset.sequence_length]['timestamp']

    print(f"Forecast Origin: {current_time}")
    print(f"Forecasting: {min(hours_ahead)} to {max(hours_ahead)} hours ahead")
    print()

    # Get spike predictions
    high_probs, low_probs, extreme_probs = predict_spike_probabilities(spike_model, spike_dataset, current_idx)

    # Get price forecasts
    dart_idx = current_idx + spike_dataset.sequence_length + 168  # Align with DART lookback
    da_forecasts, rt_forecasts = predict_price_forecasts(dart_model, dart_df, dart_features, dart_idx)

    # Display results
    print("-"*80)
    print("SPIKE PROBABILITIES & PRICE FORECASTS")
    print("-"*80)
    print(f"{'Horizon':<10} {'Spike':<12} {'DA Price (P50)':<15} {'DA Range':<20} {'RT Price (P50)':<15} {'RT Range':<20}")
    print(f"{'(hours)':<10} {'High/Extreme':<12} {'($/MWh)':<15} {'(P10-P90)':<20} {'($/MWh)':<15} {'(P10-P90)':<20}")
    print("-"*80)

    for h in hours_ahead:
        idx = h - 1

        # Spike probabilities
        spike_str = f"{high_probs[idx]*100:.1f}% / {extreme_probs[idx]*100:.1f}%"

        # DA prices (quantiles: P10, P25, P50, P75, P90)
        da_p50 = da_forecasts[idx, 2]
        da_p10 = da_forecasts[idx, 0]
        da_p90 = da_forecasts[idx, 4]
        da_range = f"${da_p10:.1f} - ${da_p90:.1f}"

        # RT prices
        rt_p50 = rt_forecasts[idx, 2]
        rt_p10 = rt_forecasts[idx, 0]
        rt_p90 = rt_forecasts[idx, 4]
        rt_range = f"${rt_p10:.1f} - ${rt_p90:.1f}"

        print(f"{h:<10} {spike_str:<12} ${da_p50:>6.2f}       {da_range:<20} ${rt_p50:>6.2f}       {rt_range:<20}")

    print("-"*80)
    print()

    # Summary statistics
    print("="*80)
    print("SUMMARY STATISTICS (48-hour forecast)")
    print("="*80)
    print()

    print(f"Spike Probability:")
    print(f"  High Spike (>$400):    {high_probs.mean()*100:.1f}% avg, max {high_probs.max()*100:.1f}% @ {high_probs.argmax()+1}h")
    print(f"  Extreme Spike (>$1000): {extreme_probs.mean()*100:.1f}% avg, max {extreme_probs.max()*100:.1f}% @ {extreme_probs.argmax()+1}h")
    print()

    print(f"DA Prices (P50):")
    print(f"  Mean:   ${da_forecasts[:, 2].mean():.2f}/MWh")
    print(f"  Range:  ${da_forecasts[:, 2].min():.2f} - ${da_forecasts[:, 2].max():.2f}/MWh")
    print(f"  Spread: ${(da_forecasts[:, 4] - da_forecasts[:, 0]).mean():.2f}/MWh (P10-P90)")
    print()

    print(f"RT Prices (P50):")
    print(f"  Mean:   ${rt_forecasts[:, 2].mean():.2f}/MWh")
    print(f"  Range:  ${rt_forecasts[:, 2].min():.2f} - ${rt_forecasts[:, 2].max():.2f}/MWh")
    print(f"  Spread: ${(rt_forecasts[:, 4] - rt_forecasts[:, 0]).mean():.2f}/MWh (P10-P90)")
    print()

    print("="*80)
    print("✓ Demo Complete!")
    print("="*80)
    print()
    print("Models Ready For:")
    print("  • SCED 60 Integration (Battalion Energy)")
    print("  • Revenue Optimization")
    print("  • Risk Management")
    print("  • Live Forecasting")

if __name__ == "__main__":
    demo_forecast()

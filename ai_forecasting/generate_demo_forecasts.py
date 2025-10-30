#!/usr/bin/env python3
"""
Generate 15 Walk-Forward Demo Forecasts for GAMBIT_ESS1
========================================================

Generate 48-hour DA+RT price forecasts for 15 strategic dates.
Walk-forward validation: Only use data BEFORE forecast origin.

For Mercuria demo - Friday 1 PM.
"""

import polars as pl
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import pickle
import json

print("="*80)
print("GENERATING 15 DEMO FORECASTS FOR GAMBIT_ESS1")
print("="*80)
print(f"Started: {datetime.now()}")

# ============================================================================
# 1. LOAD TRAINED MODEL
# ============================================================================

print("\n" + "="*80)
print("1. LOADING TRAINED MODEL")
print("="*80)

MODEL_DIR = Path("models")

# Load scalers
with open(MODEL_DIR / 'scaler_hist_fast.pkl', 'rb') as f:
    scaler_hist = pickle.load(f)
with open(MODEL_DIR / 'scaler_future_fast.pkl', 'rb') as f:
    scaler_future = pickle.load(f)
with open(MODEL_DIR / 'scaler_y_fast.pkl', 'rb') as f:
    scaler_y = pickle.load(f)

print("✓ Loaded scalers")

# Define model architecture (same as training)
class FastTransformer(nn.Module):
    def __init__(self, hist_dim, future_dim, d_model=128, nhead=4, num_layers=2):
        super().__init__()

        self.hist_proj = nn.Linear(hist_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=256, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.future_proj = nn.Linear(future_dim, d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=256, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        self.da_head = nn.Linear(d_model, 1)
        self.rt_head = nn.Linear(d_model, 1)

    def forward(self, x_hist, x_future):
        memory = self.encoder(self.hist_proj(x_hist))
        decoded = self.decoder(self.future_proj(x_future), memory)
        return self.da_head(decoded).squeeze(-1), self.rt_head(decoded).squeeze(-1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize model with correct dimensions (from training)
hist_dim = 13  # From training
future_dim = 8  # From training

model = FastTransformer(
    hist_dim=hist_dim,
    future_dim=future_dim,
    d_model=128,
    nhead=4,
    num_layers=2
).to(device)

# Load trained weights
model.load_state_dict(torch.load(MODEL_DIR / 'fast_model_best.pth'))
model.eval()

print(f"✓ Loaded model on {device}")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================================
# 2. LOAD DATA
# ============================================================================

print("\n" + "="*80)
print("2. LOADING DATA")
print("="*80)

DATA_FILE = "/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/master_enhanced_with_net_load_reserves_2019_2025.parquet"

df_pl = pl.read_parquet(DATA_FILE)
print(f"Loaded: {len(df_pl):,} records")

# CRITICAL FEATURES (same as training)
critical_features = [
    'timestamp',
    'price_da',
    'price_mean',
    *[f'price_da_lag_{i}h' for i in range(1, 25)],
    *[f'price_mean_lag_{i}h' for i in range(1, 25)],
    'net_load_MW',
    'wind_generation_MW',
    'solar_generation_MW',
    'renewable_penetration_pct',
    'net_load_ramp_1h',
    'net_load_ramp_3h',
    'reserve_margin_pct',
    'tight_reserves_flag',
    'critical_reserves_flag',
    'ordc_online_reserves_min',
    'ordc_scarcity_indicator_max',
    'ordc_critical_indicator_max',
    'load_forecast_mean',
    'KHOU_temp',
    'hour_sin',
    'hour_cos',
    'day_of_week_sin',
    'day_of_week_cos',
    'month_sin',
    'month_cos',
    'is_weekend',
]

available_features = [f for f in critical_features if f in df_pl.columns]
df = df_pl.select(available_features).to_pandas()
df = df.dropna()
df = df.sort_values('timestamp').reset_index(drop=True)

print(f"After dropna: {len(df):,} records")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

# ============================================================================
# 3. SELECT 15 STRATEGIC DEMO DATES
# ============================================================================

print("\n" + "="*80)
print("3. SELECTING 15 STRATEGIC DEMO DATES")
print("="*80)

# Select dates across different seasons and market conditions
# Focus on 2024-2025 (most recent data)
# Ensure dates have actual price data for comparison

demo_dates = []

# Filter to 2024-2025 data
df_2024_2025 = df[df['timestamp'] >= '2024-01-01']

if len(df_2024_2025) > 0:
    # Strategy: Pick dates every ~month, with some interesting events
    # Start from Jan 2024, pick dates with good data

    # Get unique months in 2024-2025
    df_2024_2025['year_month'] = pd.to_datetime(df_2024_2025['timestamp']).dt.to_period('M')
    months = df_2024_2025['year_month'].unique()

    for month in sorted(months)[:15]:  # Take first 15 months
        # For each month, pick a date in the middle
        month_data = df_2024_2025[df_2024_2025['year_month'] == month]

        # Pick a date around the 15th of the month
        mid_month = month_data.iloc[len(month_data) // 2]
        demo_dates.append(mid_month['timestamp'])

else:
    # Fallback: Use 2023 data if 2024 not available
    df_2023 = df[df['timestamp'] >= '2023-01-01']

    # Pick every ~25 days
    for i in range(0, min(15 * 600, len(df_2023)), 600):  # ~25 days = 600 hours
        demo_dates.append(df_2023.iloc[i]['timestamp'])

# Ensure we have exactly 15 dates
demo_dates = demo_dates[:15]

print(f"\n✓ Selected {len(demo_dates)} demo dates:")
for i, date in enumerate(demo_dates, 1):
    print(f"  {i:2}. {date}")

# ============================================================================
# 4. GENERATE FORECASTS
# ============================================================================

print("\n" + "="*80)
print("4. GENERATING WALK-FORWARD FORECASTS")
print("="*80)

LOOKBACK = 168  # 1 week
HORIZON = 48    # 48 hours

hist_features = [
    *[f'price_da_lag_{i}h' for i in range(1, 25) if f'price_da_lag_{i}h' in df.columns],
    *[f'price_mean_lag_{i}h' for i in range(1, 25) if f'price_mean_lag_{i}h' in df.columns],
    'net_load_MW',
    'wind_generation_MW',
    'solar_generation_MW',
    'renewable_penetration_pct',
    'net_load_ramp_1h',
    'net_load_ramp_3h',
    'reserve_margin_pct',
    'tight_reserves_flag',
    'critical_reserves_flag',
    'ordc_online_reserves_min',
    'ordc_scarcity_indicator_max',
    'ordc_critical_indicator_max',
    'load_forecast_mean',
    'KHOU_temp',
]
hist_features = [f for f in hist_features if f in df.columns]

future_features_encoded = [
    'hour_sin',
    'hour_cos',
    'day_of_week_sin',
    'day_of_week_cos',
    'month_sin',
    'month_cos',
    'is_weekend',
]
if 'load_forecast_mean' in df.columns:
    future_features_encoded.append('load_forecast_mean')
future_features_encoded = [f for f in future_features_encoded if f in df.columns]

forecasts = []

for demo_date in demo_dates:
    # Find index of forecast origin
    origin_idx = df[df['timestamp'] == demo_date].index[0]

    # Ensure we have enough lookback and horizon
    if origin_idx < LOOKBACK or origin_idx + HORIZON >= len(df):
        print(f"  ⚠️  Skipping {demo_date} (insufficient data)")
        continue

    # Extract historical window (168 hours BEFORE origin)
    hist_window = df.iloc[origin_idx - LOOKBACK:origin_idx][hist_features].values

    # Extract future features (48 hours AFTER origin)
    future_window = df.iloc[origin_idx:origin_idx + HORIZON][future_features_encoded].values

    # Extract actual prices for comparison (ground truth)
    actual_da = df.iloc[origin_idx:origin_idx + HORIZON]['price_da'].values
    actual_rt = df.iloc[origin_idx:origin_idx + HORIZON]['price_mean'].values
    timestamps = df.iloc[origin_idx:origin_idx + HORIZON]['timestamp'].values

    # Normalize
    hist_scaled = scaler_hist.transform(hist_window)
    future_scaled = scaler_future.transform(future_window)

    # Convert to tensors
    X_hist = torch.FloatTensor(hist_scaled).unsqueeze(0).to(device)  # [1, 168, hist_dim]
    X_future = torch.FloatTensor(future_scaled).unsqueeze(0).to(device)  # [1, 48, future_dim]

    # Generate forecast
    with torch.no_grad():
        pred_da_scaled, pred_rt_scaled = model(X_hist, X_future)

    # Denormalize predictions
    pred_da_scaled = pred_da_scaled.cpu().numpy()[0]  # [48]
    pred_rt_scaled = pred_rt_scaled.cpu().numpy()[0]  # [48]

    # Stack and denormalize
    pred_combined = np.stack([pred_da_scaled, pred_rt_scaled], axis=-1)  # [48, 2]
    pred_denorm = scaler_y.inverse_transform(pred_combined)

    pred_da = pred_denorm[:, 0]
    pred_rt = pred_denorm[:, 1]

    # Calculate errors
    mae_da = np.mean(np.abs(pred_da - actual_da))
    mae_rt = np.mean(np.abs(pred_rt - actual_rt))

    # Build forecast object
    forecast = {
        'origin_timestamp': str(demo_date),
        'forecast_type': '48h_da_rt',
        'battery_id': 'GAMBIT_ESS1',
        'horizon_hours': 48,
        'mae_da': float(mae_da),
        'mae_rt': float(mae_rt),
        'hourly_forecast': []
    }

    for h in range(HORIZON):
        forecast['hourly_forecast'].append({
            'timestamp': str(timestamps[h]),
            'hour_ahead': h + 1,
            'price_da_forecast': float(pred_da[h]),
            'price_rt_forecast': float(pred_rt[h]),
            'price_da_actual': float(actual_da[h]),
            'price_rt_actual': float(actual_rt[h]),
            'error_da': float(pred_da[h] - actual_da[h]),
            'error_rt': float(pred_rt[h] - actual_rt[h]),
        })

    forecasts.append(forecast)

    print(f"  ✓ {demo_date}: MAE_DA = ${mae_da:.2f}/MWh, MAE_RT = ${mae_rt:.2f}/MWh")

print(f"\n✓ Generated {len(forecasts)} forecasts")

# ============================================================================
# 5. SAVE FORECASTS
# ============================================================================

print("\n" + "="*80)
print("5. SAVING FORECASTS")
print("="*80)

# Save to JSON
output_file = Path("demo_forecasts_gambit_ess1.json")
with open(output_file, 'w') as f:
    json.dump(forecasts, f, indent=2)

print(f"\n✓ Saved: {output_file}")
print(f"  Size: {output_file.stat().st_size / 1024:.1f} KB")
print(f"  Forecasts: {len(forecasts)}")

# Summary statistics
print("\n" + "="*80)
print("FORECAST PERFORMANCE SUMMARY")
print("="*80)

mae_das = [f['mae_da'] for f in forecasts]
mae_rts = [f['mae_rt'] for f in forecasts]

print(f"\nDay-Ahead Forecasts:")
print(f"  Mean MAE: ${np.mean(mae_das):.2f}/MWh")
print(f"  Median MAE: ${np.median(mae_das):.2f}/MWh")
print(f"  Best MAE: ${np.min(mae_das):.2f}/MWh")
print(f"  Worst MAE: ${np.max(mae_das):.2f}/MWh")

print(f"\nReal-Time Forecasts:")
print(f"  Mean MAE: ${np.mean(mae_rts):.2f}/MWh")
print(f"  Median MAE: ${np.median(mae_rts):.2f}/MWh")
print(f"  Best MAE: ${np.min(mae_rts):.2f}/MWh")
print(f"  Worst MAE: ${np.max(mae_rts):.2f}/MWh")

print("\n" + "="*80)
print("✓ DEMO FORECASTS READY!")
print("="*80)
print(f"Finished: {datetime.now()}")
print(f"\nNext: Load into forecast API and prepare visualizations for Friday demo")

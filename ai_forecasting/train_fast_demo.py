#!/usr/bin/env python3
"""
FAST Training for Demo - Feature Selection Version
==================================================

Use ONLY the most critical features to get fast training (~30-60 min per epoch):
- Price lags (last 24h)
- Net load features
- Reserve margin
- ORDC scarcity
- Temperature
- Temporal features

Goal: Verify model works in 2-3 hours, then iterate if needed.
"""

import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import pickle
from tqdm import tqdm

print("="*80)
print("FAST TRAINING FOR DEMO - FEATURE SELECTION")
print("="*80)

# ============================================================================
# 1. LOAD DATA AND SELECT CRITICAL FEATURES ONLY
# ============================================================================

DATA_FILE = "/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/master_enhanced_with_net_load_reserves_2019_2025.parquet"
OUTPUT_DIR = Path("models")
OUTPUT_DIR.mkdir(exist_ok=True)

print("\nLoading data...")
df_pl = pl.read_parquet(DATA_FILE)
print(f"Full dataset: {len(df_pl):,} records, {len(df_pl.columns)} columns")

# CRITICAL FEATURES ONLY - Keep it minimal!
critical_features = [
    'timestamp',

    # Target prices
    'price_da',
    'price_mean',  # RT price

    # Price lags (last 24 hours = 24 features)
    *[f'price_da_lag_{i}h' for i in range(1, 25)],
    *[f'price_mean_lag_{i}h' for i in range(1, 25)],

    # Net load features (NEW - CRITICAL!)
    'net_load_MW',
    'wind_generation_MW',
    'solar_generation_MW',
    'renewable_penetration_pct',
    'net_load_ramp_1h',
    'net_load_ramp_3h',

    # Reserve margin (NEW - CRITICAL!)
    'reserve_margin_pct',
    'tight_reserves_flag',
    'critical_reserves_flag',

    # ORDC scarcity (existing - CRITICAL!)
    'ordc_online_reserves_min',
    'ordc_scarcity_indicator_max',
    'ordc_critical_indicator_max',

    # Load forecast
    'load_forecast_mean',

    # Temperature (aggregate)
    'KHOU_temp',  # Houston

    # Temporal (already encoded)
    'hour_sin',
    'hour_cos',
    'day_of_week_sin',
    'day_of_week_cos',
    'month_sin',
    'month_cos',
    'is_weekend',
]

# Select only features that exist
available_features = [f for f in critical_features if f in df_pl.columns]
print(f"\nSelected {len(available_features)} critical features (from {len(critical_features)} requested)")

df = df_pl.select(available_features).to_pandas()
df = df.dropna()
print(f"After dropna: {len(df):,} records")

# ============================================================================
# 2. PREPARE FEATURES
# ============================================================================

print("\nPreparing features...")

# Historical features (past week context)
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

# Future features (known ahead for forecast horizon - already encoded in dataset)
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

print(f"\nHistorical features: {len(hist_features)}")
print(f"Future features: {len(future_features_encoded)}")

# ============================================================================
# 3. CREATE SEQUENCES
# ============================================================================

print("\nCreating sequences...")

LOOKBACK = 168  # 1 week
HORIZON = 48    # 48 hours ahead

X_hist, X_future, y_da, y_rt = [], [], [], []

for i in range(LOOKBACK, len(df) - HORIZON):
    # Historical window
    hist_window = df.iloc[i-LOOKBACK:i][hist_features].values

    # Future features for forecast horizon
    future_window = df.iloc[i:i+HORIZON][future_features_encoded].values

    # Targets
    da_target = df.iloc[i:i+HORIZON]['price_da'].values
    rt_target = df.iloc[i:i+HORIZON]['price_mean'].values

    X_hist.append(hist_window)
    X_future.append(future_window)
    y_da.append(da_target)
    y_rt.append(rt_target)

X_hist = np.array(X_hist)
X_future = np.array(X_future)
y_da = np.array(y_da)
y_rt = np.array(y_rt)

print(f"\nSequences created:")
print(f"  X_hist: {X_hist.shape}")
print(f"  X_future: {X_future.shape}")
print(f"  y_da: {y_da.shape}")
print(f"  y_rt: {y_rt.shape}")

# ============================================================================
# 4. NORMALIZE
# ============================================================================

print("\nNormalizing...")

scaler_hist = StandardScaler()
scaler_future = StandardScaler()
scaler_y = StandardScaler()

X_hist_flat = X_hist.reshape(-1, X_hist.shape[-1])
X_hist_scaled = scaler_hist.fit_transform(X_hist_flat).reshape(X_hist.shape)

X_future_flat = X_future.reshape(-1, X_future.shape[-1])
X_future_scaled = scaler_future.fit_transform(X_future_flat).reshape(X_future.shape)

y_combined = np.stack([y_da, y_rt], axis=-1)
y_flat = y_combined.reshape(-1, 2)
y_scaled = scaler_y.fit_transform(y_flat).reshape(y_combined.shape)
y_da_scaled = y_scaled[:, :, 0]
y_rt_scaled = y_scaled[:, :, 1]

# Save scalers
with open(OUTPUT_DIR / 'scaler_hist_fast.pkl', 'wb') as f:
    pickle.dump(scaler_hist, f)
with open(OUTPUT_DIR / 'scaler_future_fast.pkl', 'wb') as f:
    pickle.dump(scaler_future, f)
with open(OUTPUT_DIR / 'scaler_y_fast.pkl', 'wb') as f:
    pickle.dump(scaler_y, f)

# ============================================================================
# 5. TRAIN/VAL SPLIT
# ============================================================================

split_idx = int(0.8 * len(X_hist_scaled))

X_hist_train = torch.FloatTensor(X_hist_scaled[:split_idx])
X_future_train = torch.FloatTensor(X_future_scaled[:split_idx])
y_da_train = torch.FloatTensor(y_da_scaled[:split_idx])
y_rt_train = torch.FloatTensor(y_rt_scaled[:split_idx])

X_hist_val = torch.FloatTensor(X_hist_scaled[split_idx:])
X_future_val = torch.FloatTensor(X_future_scaled[split_idx:])
y_da_val = torch.FloatTensor(y_da_scaled[split_idx:])
y_rt_val = torch.FloatTensor(y_rt_scaled[split_idx:])

print(f"\nTrain samples: {len(X_hist_train):,}")
print(f"Val samples: {len(X_hist_val):,}")

# ============================================================================
# 6. DATASET AND DATALOADER
# ============================================================================

class PriceDataset(Dataset):
    def __init__(self, X_hist, X_future, y_da, y_rt):
        self.X_hist = X_hist
        self.X_future = X_future
        self.y_da = y_da
        self.y_rt = y_rt

    def __len__(self):
        return len(self.X_hist)

    def __getitem__(self, idx):
        return self.X_hist[idx], self.X_future[idx], self.y_da[idx], self.y_rt[idx]

train_dataset = PriceDataset(X_hist_train, X_future_train, y_da_train, y_rt_train)
val_dataset = PriceDataset(X_hist_val, X_future_val, y_da_val, y_rt_val)

# LARGER BATCH SIZE for faster training!
BATCH_SIZE = 64  # Increased from 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

print(f"Batch size: {BATCH_SIZE}")
print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")

# ============================================================================
# 7. MODEL (SMALLER for fast training)
# ============================================================================

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
model = FastTransformer(
    hist_dim=X_hist_train.shape[-1],
    future_dim=X_future_train.shape[-1],
    d_model=128,  # Smaller model
    nhead=4,
    num_layers=2
).to(device)

print(f"\nUsing device: {device}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================================
# 8. TRAINING LOOP
# ============================================================================

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

best_val_loss = float('inf')
patience = 5
patience_counter = 0

print("\n" + "="*80)
print("STARTING FAST TRAINING")
print("="*80)

for epoch in range(50):
    # Train
    model.train()
    train_loss = 0

    for X_h, X_f, y_d, y_r in tqdm(train_loader, desc=f"Epoch {epoch+1}/50"):
        X_h, X_f = X_h.to(device), X_f.to(device)
        y_d, y_r = y_d.to(device), y_r.to(device)

        optimizer.zero_grad()
        pred_da, pred_rt = model(X_h, X_f)

        loss = criterion(pred_da, y_d) + criterion(pred_rt, y_r)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Validate
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for X_h, X_f, y_d, y_r in val_loader:
            X_h, X_f = X_h.to(device), X_f.to(device)
            y_d, y_r = y_d.to(device), y_r.to(device)

            pred_da, pred_rt = model(X_h, X_f)
            loss = criterion(pred_da, y_d) + criterion(pred_rt, y_r)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), OUTPUT_DIR / 'fast_model_best.pth')
        print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

print("\n" + "="*80)
print("✓ TRAINING COMPLETE!")
print("="*80)
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Model saved: {OUTPUT_DIR / 'fast_model_best.pth'}")

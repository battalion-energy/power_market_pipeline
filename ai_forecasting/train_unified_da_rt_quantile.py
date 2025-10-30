#!/usr/bin/env python3
"""
Unified DA + RT Price Forecaster with Quantile Regression
Predicts BOTH Day-Ahead and Real-Time prices for 48 hours with confidence intervals

Perfect for Mercuria Demo:
- Shows DA price forecasts (P10, P25, P50, P75, P90)
- Shows RT price forecasts (P10, P25, P50, P75, P90)
- Side-by-side comparison
- DA-RT spread analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import json
import math
from tqdm import tqdm

# =====================
# Configuration
# =====================

DATA_FILE = "/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/master_features_enhanced_with_ordc_load_2019_2025.parquet"
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# Model hyperparameters
D_MODEL = 256
NHEAD = 8
NUM_ENCODER_LAYERS = 4
NUM_DECODER_LAYERS = 4
DIM_FEEDFORWARD = 1024
DROPOUT = 0.1

# Training
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-5

# Forecasting
LOOKBACK_HOURS = 168
FORECAST_HORIZON = 48
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]

# =====================
# Quantile Loss
# =====================

class QuantileLoss(nn.Module):
    """Quantile regression loss."""
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = torch.tensor(quantiles, dtype=torch.float32)

    def forward(self, y_pred, y_true):
        """
        y_pred: (batch, horizon, n_quantiles)
        y_true: (batch, horizon)
        """
        y_true = y_true.unsqueeze(-1).expand_as(y_pred)
        errors = y_true - y_pred
        quantiles = self.quantiles.to(y_pred.device).view(1, 1, -1)
        loss = torch.max((quantiles - 1) * errors, quantiles * errors)
        return loss.mean()

# =====================
# Positional Encoding
# =====================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# =====================
# Unified DA+RT Transformer
# =====================

class UnifiedDART_Forecaster(nn.Module):
    """
    Predicts BOTH DA and RT prices with quantiles.

    Outputs:
    - DA predictions: (batch, 48, 5 quantiles)
    - RT predictions: (batch, 48, 5 quantiles)
    """
    def __init__(
        self,
        input_dim,
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        n_quantiles=5,
        forecast_horizon=48,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_quantiles = n_quantiles
        self.forecast_horizon = forecast_horizon

        # Shared encoder
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # Shared decoder queries
        self.decoder_queries = nn.Parameter(torch.randn(forecast_horizon, d_model))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        # Separate output heads for DA and RT
        self.da_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, n_quantiles),
        )

        self.rt_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, n_quantiles),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x_hist):
        """
        Returns:
            da_quantiles: (batch, horizon, n_quantiles)
            rt_quantiles: (batch, horizon, n_quantiles)
        """
        batch_size = x_hist.size(0)

        # Encode
        x_enc = self.input_proj(x_hist)
        x_enc = self.pos_encoder(x_enc)
        memory = self.encoder(x_enc)

        # Decode
        tgt = self.decoder_queries.unsqueeze(0).expand(batch_size, -1, -1)
        decoder_out = self.decoder(tgt, memory)

        # Predict DA and RT separately
        da_quantiles = self.da_head(decoder_out)
        rt_quantiles = self.rt_head(decoder_out)

        return da_quantiles, rt_quantiles

# =====================
# Dataset
# =====================

class UnifiedDataset(Dataset):
    """Dataset for DA + RT forecasting."""
    def __init__(self, df, feature_cols, lookback=168, horizon=48):
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.lookback = lookback
        self.horizon = horizon
        self.valid_indices = list(range(lookback, len(df) - horizon))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        i = self.valid_indices[idx]

        # Historical features
        x_hist = self.df[self.feature_cols].iloc[i-self.lookback:i].values.astype(np.float32)

        # Future targets (DA and RT)
        y_da = self.df['price_da'].iloc[i+1:i+1+self.horizon].values.astype(np.float32)
        y_rt = self.df['price_rt'].iloc[i+1:i+1+self.horizon].values.astype(np.float32)

        return torch.from_numpy(x_hist), torch.from_numpy(y_da), torch.from_numpy(y_rt)

# =====================
# Feature Engineering
# =====================

def prepare_data(df_pl):
    """Prepare features with both DA and RT prices."""
    print("Preparing unified DA + RT features...")

    df = df_pl.to_pandas()

    # Handle datetime/timestamp index
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime').sort_index()
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
    elif df.index.name in ['datetime', 'timestamp']:
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
    else:
        print("ERROR: No datetime/timestamp column found")
        return None, None

    # RT prices - use price_mean as RT price (aggregated across hubs)
    if 'price_mean' in df.columns:
        df['price_rt'] = df['price_mean']
    else:
        # Fallback: try to find hub-level RT columns
        rt_cols = [col for col in df.columns if 'RT_' in col and 'HB_' in col]
        if not rt_cols:
            print("ERROR: No RT price columns found (neither 'price_mean' nor 'RT_HB_*')")
            return None, None
        df['price_rt'] = df[rt_cols].mean(axis=1)

    # DA prices - already exists in dataset
    if 'price_da' not in df.columns:
        # Fallback: try to find hub-level DA columns
        da_cols = [col for col in df.columns if 'DA_' in col and 'HB_' in col]
        if not da_cols:
            print("ERROR: No DA price columns found (neither 'price_da' nor 'DA_HB_*')")
            return None, None
        df['price_da'] = df[da_cols].mean(axis=1)

    # DA-RT spread
    df['da_rt_spread'] = df['price_da'] - df['price_rt']
    df['da_rt_spread_pct'] = 100 * df['da_rt_spread'] / (df['price_da'] + 1e-6)

    # Temporal features
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)

    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Lag features for both DA and RT
    for price_type in ['price_da', 'price_rt']:
        for lag in [1, 24, 48, 168]:
            df[f'{price_type}_lag_{lag}'] = df[price_type].shift(lag)

    # Rolling stats
    for price_type in ['price_da', 'price_rt']:
        for window in [24, 168]:
            df[f'{price_type}_roll_mean_{window}'] = df[price_type].rolling(window).mean()
            df[f'{price_type}_roll_std_{window}'] = df[price_type].rolling(window).std()

    # AS prices
    as_cols = ['REGUP', 'REGDN', 'RRS', 'NSPIN', 'ECRS']
    available_as = [col for col in as_cols if col in df.columns]
    if available_as:
        df['as_total'] = df[available_as].sum(axis=1)
        df['as_total_lag_1'] = df['as_total'].shift(1)

    # Drop NaN
    df = df.dropna()

    if len(df) == 0:
        print("ERROR: No data after feature engineering")
        return None, None

    print(f"Features created. Dataset: {len(df):,} samples")

    # Select feature columns (exclude targets and raw hub prices if they exist)
    exclude_cols = ['price_rt', 'price_da', 'hour', 'dayofweek', 'month', 'price_mean']
    # Add hub-level columns if they exist
    rt_cols = [col for col in df.columns if 'RT_' in col and 'HB_' in col]
    da_cols = [col for col in df.columns if 'DA_' in col and 'HB_' in col]
    exclude_cols.extend(rt_cols)
    exclude_cols.extend(da_cols)
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    return df, feature_cols

# =====================
# Training
# =====================

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss_da = 0
    total_loss_rt = 0
    n_batches = 0

    for x_hist, y_da, y_rt in tqdm(dataloader, desc="Training", leave=False):
        x_hist = x_hist.to(device)
        y_da = y_da.to(device)
        y_rt = y_rt.to(device)

        # Forward
        da_pred, rt_pred = model(x_hist)

        # Loss for both DA and RT
        loss_da = criterion(da_pred, y_da)
        loss_rt = criterion(rt_pred, y_rt)
        loss = loss_da + loss_rt

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss_da += loss_da.item()
        total_loss_rt += loss_rt.item()
        n_batches += 1

    return total_loss_da / n_batches, total_loss_rt / n_batches

@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss_da = 0
    total_loss_rt = 0
    n_batches = 0

    all_da_preds = []
    all_da_targets = []
    all_rt_preds = []
    all_rt_targets = []

    for x_hist, y_da, y_rt in tqdm(dataloader, desc="Validating", leave=False):
        x_hist = x_hist.to(device)
        y_da = y_da.to(device)
        y_rt = y_rt.to(device)

        da_pred, rt_pred = model(x_hist)

        loss_da = criterion(da_pred, y_da)
        loss_rt = criterion(rt_pred, y_rt)

        total_loss_da += loss_da.item()
        total_loss_rt += loss_rt.item()
        n_batches += 1

        # Collect median predictions (P50)
        all_da_preds.append(da_pred[:, :, 2].cpu().numpy())
        all_da_targets.append(y_da.cpu().numpy())
        all_rt_preds.append(rt_pred[:, :, 2].cpu().numpy())
        all_rt_targets.append(y_rt.cpu().numpy())

    avg_loss_da = total_loss_da / n_batches
    avg_loss_rt = total_loss_rt / n_batches

    # MAE on median predictions
    da_preds = np.concatenate(all_da_preds, axis=0)
    da_targets = np.concatenate(all_da_targets, axis=0)
    mae_da = np.mean(np.abs(da_preds - da_targets))

    rt_preds = np.concatenate(all_rt_preds, axis=0)
    rt_targets = np.concatenate(all_rt_targets, axis=0)
    mae_rt = np.mean(np.abs(rt_preds - rt_targets))

    return avg_loss_da, avg_loss_rt, mae_da, mae_rt

# =====================
# Main
# =====================

def main():
    print("="*60)
    print("Unified DA + RT Forecaster with Confidence Intervals")
    print("="*60)
    print(f"Started: {datetime.now()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    if not Path(DATA_FILE).exists():
        print("ERROR: Data file not found")
        return

    print(f"\nLoading: {DATA_FILE}")
    df_pl = pl.read_parquet(DATA_FILE)
    print(f"Loaded: {len(df_pl):,} rows")

    # Feature engineering
    df, feature_cols = prepare_data(df_pl)
    if df is None:
        return

    # Split
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    df_train = df.iloc[:train_end]
    df_val = df.iloc[train_end:val_end]
    df_test = df.iloc[val_end:]

    print(f"\nSplit:")
    print(f"  Train: {len(df_train):,}")
    print(f"  Val:   {len(df_val):,}")
    print(f"  Test:  {len(df_test):,}")

    # Scale
    scaler = StandardScaler()
    df_train[feature_cols] = scaler.fit_transform(df_train[feature_cols])
    df_val[feature_cols] = scaler.transform(df_val[feature_cols])
    df_test[feature_cols] = scaler.transform(df_test[feature_cols])

    # Datasets
    train_dataset = UnifiedDataset(df_train, feature_cols)
    val_dataset = UnifiedDataset(df_val, feature_cols)
    test_dataset = UnifiedDataset(df_test, feature_cols)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Model
    print(f"\n{'='*60}")
    print("Creating Unified DA+RT Model")
    print(f"{'='*60}")

    model = UnifiedDART_Forecaster(
        input_dim=len(feature_cols),
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        n_quantiles=len(QUANTILES),
        forecast_horizon=FORECAST_HORIZON,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")
    print(f"Quantiles: {QUANTILES}")

    # Training setup
    criterion = QuantileLoss(QUANTILES)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Train
    print(f"\n{'='*60}")
    print(f"Training for {EPOCHS} epochs")
    print(f"{'='*60}")

    best_val_loss = float('inf')
    history = []

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")

        train_loss_da, train_loss_rt = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss_da, val_loss_rt, val_mae_da, val_mae_rt = validate(model, val_loader, criterion, device)

        total_val_loss = val_loss_da + val_loss_rt
        scheduler.step(total_val_loss)

        print(f"  Train Loss DA: {train_loss_da:.4f}  RT: {train_loss_rt:.4f}")
        print(f"  Val Loss   DA: {val_loss_da:.4f}  RT: {val_loss_rt:.4f}")
        print(f"  Val MAE    DA: ${val_mae_da:.2f}/MWh  RT: ${val_mae_rt:.2f}/MWh")

        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss_da': val_loss_da,
                'val_loss_rt': val_loss_rt,
                'val_mae_da': val_mae_da,
                'val_mae_rt': val_mae_rt,
            }, MODEL_DIR / "unified_da_rt_best.pth")
            print(f"  ✓ Best model saved")

        history.append({
            'epoch': epoch,
            'train_loss_da': train_loss_da,
            'train_loss_rt': train_loss_rt,
            'val_loss_da': val_loss_da,
            'val_loss_rt': val_loss_rt,
            'val_mae_da': val_mae_da,
            'val_mae_rt': val_mae_rt,
        })

    # Test
    print(f"\n{'='*60}")
    print("Test Evaluation")
    print(f"{'='*60}")

    checkpoint = torch.load(MODEL_DIR / "unified_da_rt_best.pth", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss_da, test_loss_rt, test_mae_da, test_mae_rt = validate(model, test_loader, criterion, device)
    print(f"DA - Test MAE: ${test_mae_da:.2f}/MWh")
    print(f"RT - Test MAE: ${test_mae_rt:.2f}/MWh")

    # Save metadata
    metadata = {
        'created': datetime.now().isoformat(),
        'model_type': 'UnifiedDART_Forecaster',
        'quantiles': QUANTILES,
        'forecast_horizon': FORECAST_HORIZON,
        'metrics': {
            'da': {
                'best_val_mae': float(checkpoint['val_mae_da']),
                'test_mae': float(test_mae_da),
            },
            'rt': {
                'best_val_mae': float(checkpoint['val_mae_rt']),
                'test_mae': float(test_mae_rt),
            }
        },
        'training_history': history,
    }

    with open(MODEL_DIR / "unified_da_rt_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    import joblib
    joblib.dump(scaler, MODEL_DIR / "unified_scaler.joblib")

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"DA Best Val MAE: ${checkpoint['val_mae_da']:.2f}/MWh")
    print(f"DA Test MAE:     ${test_mae_da:.2f}/MWh")
    print(f"RT Best Val MAE: ${checkpoint['val_mae_rt']:.2f}/MWh")
    print(f"RT Test MAE:     ${test_mae_rt:.2f}/MWh")
    print(f"\nModel: {MODEL_DIR / 'unified_da_rt_best.pth'}")
    print(f"Scaler: {MODEL_DIR / 'unified_scaler.joblib'}")
    print(f"\nReady for demo with DA + RT confidence intervals!")

if __name__ == "__main__":
    main()

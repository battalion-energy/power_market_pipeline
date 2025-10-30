#!/usr/bin/env python3
"""
Transformer-based Multi-Horizon Forecaster with Quantile Regression
Produces probabilistic 48-hour forecasts with confidence intervals (P10, P25, P50, P75, P90)
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

DATA_FILE = "/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/master_ml_dataset_2019_2025.parquet"
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# Model hyperparameters
D_MODEL = 256          # Embedding dimension
NHEAD = 8              # Number of attention heads
NUM_ENCODER_LAYERS = 4  # Transformer encoder layers
NUM_DECODER_LAYERS = 4  # Transformer decoder layers
DIM_FEEDFORWARD = 1024  # FFN hidden dimension
DROPOUT = 0.1

# Training hyperparameters
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-5

# Forecasting
LOOKBACK_HOURS = 168   # 7 days of history
FORECAST_HORIZON = 48  # 48 hours ahead
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]  # Confidence intervals

# =====================
# Quantile Loss
# =====================

class QuantileLoss(nn.Module):
    """Quantile regression loss (pinball loss)."""
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = torch.tensor(quantiles, dtype=torch.float32)

    def forward(self, y_pred, y_true):
        """
        y_pred: (batch, horizon, n_quantiles)
        y_true: (batch, horizon)
        """
        # Expand y_true to match quantiles dimension
        y_true = y_true.unsqueeze(-1).expand_as(y_pred)

        # Compute errors
        errors = y_true - y_pred

        # Quantile loss per quantile
        quantiles = self.quantiles.to(y_pred.device).view(1, 1, -1)
        loss = torch.max((quantiles - 1) * errors, quantiles * errors)

        return loss.mean()

# =====================
# Positional Encoding
# =====================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """x: (batch, seq_len, d_model)"""
        return x + self.pe[:, :x.size(1)]

# =====================
# Transformer Model
# =====================

class TransformerQuantileForecaster(nn.Module):
    """
    Encoder-Decoder Transformer for multi-horizon quantile forecasting.

    Architecture:
    - Encoder: Processes historical sequence (168h)
    - Decoder: Generates future predictions (48h × 5 quantiles)
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

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # Learnable decoder queries (one per forecast hour)
        self.decoder_queries = nn.Parameter(torch.randn(forecast_horizon, d_model))

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        # Output head: predict n_quantiles per horizon
        self.output_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, n_quantiles),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x_hist):
        """
        x_hist: (batch, lookback, input_dim) - Historical features

        Returns:
            (batch, horizon, n_quantiles) - Quantile predictions
        """
        batch_size = x_hist.size(0)

        # Encode historical sequence
        x_enc = self.input_proj(x_hist)  # (batch, lookback, d_model)
        x_enc = self.pos_encoder(x_enc)
        memory = self.encoder(x_enc)  # (batch, lookback, d_model)

        # Expand decoder queries for batch
        tgt = self.decoder_queries.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, horizon, d_model)

        # Decode
        decoder_out = self.decoder(tgt, memory)  # (batch, horizon, d_model)

        # Predict quantiles
        quantiles = self.output_head(decoder_out)  # (batch, horizon, n_quantiles)

        return quantiles

# =====================
# Dataset
# =====================

class TimeSeriesDataset(Dataset):
    """Time series dataset with lookback."""
    def __init__(self, df, feature_cols, target_col='price_target', lookback=168, horizon=48):
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.lookback = lookback
        self.horizon = horizon

        # Valid indices where we have enough history and future
        self.valid_indices = list(range(lookback, len(df) - horizon))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        i = self.valid_indices[idx]

        # Historical features (t-lookback to t)
        x_hist = self.df[self.feature_cols].iloc[i-self.lookback:i].values.astype(np.float32)

        # Future targets (t+1 to t+horizon)
        y_fut = self.df[self.target_col].iloc[i+1:i+1+self.horizon].values.astype(np.float32)

        return torch.from_numpy(x_hist), torch.from_numpy(y_fut)

# =====================
# Feature Engineering
# =====================

def prepare_data(df_pl):
    """Prepare features from Polars DataFrame."""
    print("Preparing features...")

    df = df_pl.to_pandas()

    # Ensure datetime index
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime').sort_index()

    # Target: RT price
    price_cols = [col for col in df.columns if 'RT_' in col and 'HB_' in col]
    if price_cols:
        df['price_target'] = df[price_cols].mean(axis=1)
    else:
        print("ERROR: No RT price columns found")
        return None, None

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

    # Lag features
    for lag in [1, 24, 48, 168]:
        df[f'price_lag_{lag}'] = df['price_target'].shift(lag)

    # Rolling stats
    for window in [24, 168]:
        df[f'price_roll_mean_{window}'] = df['price_target'].rolling(window).mean()
        df[f'price_roll_std_{window}'] = df['price_target'].rolling(window).std()

    # DA prices (if available)
    da_cols = [col for col in df.columns if 'DA_' in col and 'HB_' in col]
    if da_cols:
        df['price_da'] = df[da_cols].mean(axis=1)
        df['da_rt_spread'] = df['price_da'] - df['price_target']

    # AS prices (if available)
    as_cols = ['REGUP', 'REGDN', 'RRS', 'NSPIN', 'ECRS']
    available_as = [col for col in as_cols if col in df.columns]
    if available_as:
        df['as_total'] = df[available_as].sum(axis=1)

    # Drop NaN
    df = df.dropna()

    if len(df) == 0:
        print("ERROR: No data after feature engineering")
        return None, None

    print(f"Features created. Dataset: {len(df):,} samples")

    # Select feature columns
    exclude_cols = ['price_target'] + price_cols + da_cols
    feature_cols = [col for col in df.columns if col not in exclude_cols and col != 'hour' and col != 'dayofweek' and col != 'month']

    return df, feature_cols

# =====================
# Training
# =====================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0

    for x_hist, y_fut in tqdm(dataloader, desc="Training", leave=False):
        x_hist = x_hist.to(device)
        y_fut = y_fut.to(device)

        # Forward
        y_pred = model(x_hist)  # (batch, horizon, n_quantiles)

        # Loss
        loss = criterion(y_pred, y_fut)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches

@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    n_batches = 0

    all_preds = []
    all_targets = []

    for x_hist, y_fut in tqdm(dataloader, desc="Validating", leave=False):
        x_hist = x_hist.to(device)
        y_fut = y_fut.to(device)

        # Forward
        y_pred = model(x_hist)

        # Loss
        loss = criterion(y_pred, y_fut)

        total_loss += loss.item()
        n_batches += 1

        # Collect predictions (median quantile)
        all_preds.append(y_pred[:, :, 2].cpu().numpy())  # P50
        all_targets.append(y_fut.cpu().numpy())

    avg_loss = total_loss / n_batches

    # Compute MAE on median predictions
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    mae = np.mean(np.abs(preds - targets))

    return avg_loss, mae

# =====================
# Main
# =====================

def main():
    print("="*60)
    print("Transformer Quantile Forecaster - 48h with Confidence Intervals")
    print("="*60)
    print(f"Started: {datetime.now()}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Check data file
    if not Path(DATA_FILE).exists():
        alt_file = "/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/master_features_multihorizon_2019_2025.parquet"
        if Path(alt_file).exists():
            global DATA_FILE
            DATA_FILE = alt_file
        else:
            print(f"ERROR: Data file not found")
            return

    # Load data
    print(f"\nLoading: {DATA_FILE}")
    df_pl = pl.read_parquet(DATA_FILE)
    print(f"Loaded: {len(df_pl):,} rows")

    # Feature engineering
    df, feature_cols = prepare_data(df_pl)
    if df is None:
        return

    # Split data (time-based)
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    df_train = df.iloc[:train_end]
    df_val = df.iloc[train_end:val_end]
    df_test = df.iloc[val_end:]

    print(f"\nSplit:")
    print(f"  Train: {len(df_train):,} samples")
    print(f"  Val:   {len(df_val):,} samples")
    print(f"  Test:  {len(df_test):,} samples")

    # Scale features
    scaler = StandardScaler()
    df_train[feature_cols] = scaler.fit_transform(df_train[feature_cols])
    df_val[feature_cols] = scaler.transform(df_val[feature_cols])
    df_test[feature_cols] = scaler.transform(df_test[feature_cols])

    # Create datasets
    train_dataset = TimeSeriesDataset(df_train, feature_cols, lookback=LOOKBACK_HOURS, horizon=FORECAST_HORIZON)
    val_dataset = TimeSeriesDataset(df_val, feature_cols, lookback=LOOKBACK_HOURS, horizon=FORECAST_HORIZON)
    test_dataset = TimeSeriesDataset(df_test, feature_cols, lookback=LOOKBACK_HOURS, horizon=FORECAST_HORIZON)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print(f"\nDataLoaders ready:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")

    # Create model
    print(f"\n{'='*60}")
    print("Creating Model")
    print(f"{'='*60}")

    model = TransformerQuantileForecaster(
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
    print(f"Model parameters: {n_params:,}")
    print(f"Quantiles: {QUANTILES}")

    # Loss and optimizer
    criterion = QuantileLoss(QUANTILES)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # Training loop
    print(f"\n{'='*60}")
    print(f"Training for {EPOCHS} epochs")
    print(f"{'='*60}")

    best_val_loss = float('inf')
    history = []

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_mae = validate(model, val_loader, criterion, device)

        # Scheduler step
        scheduler.step(val_loss)

        # Print
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val MAE:    ${val_mae:.2f}/MWh")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_mae': val_mae,
            }, MODEL_DIR / "transformer_quantile_best.pth")
            print(f"  ✓ Best model saved (val_loss={val_loss:.4f})")

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_mae': val_mae,
        })

    # Test evaluation
    print(f"\n{'='*60}")
    print("Test Evaluation")
    print(f"{'='*60}")

    checkpoint = torch.load(MODEL_DIR / "transformer_quantile_best.pth")
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_mae = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test MAE:  ${test_mae:.2f}/MWh")

    # Save metadata
    metadata = {
        'created': datetime.now().isoformat(),
        'model_type': 'TransformerQuantileForecaster',
        'quantiles': QUANTILES,
        'forecast_horizon': FORECAST_HORIZON,
        'lookback_hours': LOOKBACK_HOURS,
        'n_features': len(feature_cols),
        'feature_names': feature_cols,
        'hyperparameters': {
            'd_model': D_MODEL,
            'nhead': NHEAD,
            'num_encoder_layers': NUM_ENCODER_LAYERS,
            'num_decoder_layers': NUM_DECODER_LAYERS,
            'dim_feedforward': DIM_FEEDFORWARD,
            'dropout': DROPOUT,
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'learning_rate': LEARNING_RATE,
        },
        'metrics': {
            'best_val_loss': float(best_val_loss),
            'best_val_mae': float(checkpoint['val_mae']),
            'test_loss': float(test_loss),
            'test_mae': float(test_mae),
        },
        'training_history': history,
    }

    with open(MODEL_DIR / "transformer_quantile_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    # Save scaler
    import joblib
    joblib.dump(scaler, MODEL_DIR / "transformer_scaler.joblib")

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Best Val MAE:  ${checkpoint['val_mae']:.2f}/MWh")
    print(f"Test MAE:      ${test_mae:.2f}/MWh")
    print(f"\nModel saved: {MODEL_DIR / 'transformer_quantile_best.pth'}")
    print(f"Scaler saved: {MODEL_DIR / 'transformer_scaler.joblib'}")
    print(f"\nReady for probabilistic forecasting with {len(QUANTILES)} quantiles!")

if __name__ == "__main__":
    main()

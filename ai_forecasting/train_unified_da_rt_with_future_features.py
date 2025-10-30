#!/usr/bin/env python3
"""
FIXED: Unified DA + RT Price Forecaster with Future Temporal Features

KEY FIX: Model now receives future temporal features (hour-of-day, day-of-week, etc.)
for the 48-hour forecast horizon, enabling it to capture diurnal patterns.

Previous bug: Model only saw historical data, couldn't predict hourly patterns.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import json
import math
from tqdm import tqdm

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
# FIXED: Model with Future Features
# =====================

class UnifiedDART_WithFutureFeatures(nn.Module):
    """
    FIXED: Now accepts future temporal features for forecast horizon

    Inputs:
        x_hist: (batch, seq_len_hist, hist_features) - Historical data
        x_future: (batch, forecast_horizon, future_features) - Future temporal features
                  (hour_sin, hour_cos, dayofweek_sin, dayofweek_cos, month_sin, month_cos, etc.)

    Outputs:
        da_quantiles: (batch, forecast_horizon, n_quantiles)
        rt_quantiles: (batch, forecast_horizon, n_quantiles)
    """
    def __init__(
        self,
        hist_feature_dim,
        future_feature_dim,
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

        # Encoder for historical data
        self.hist_proj = nn.Linear(hist_feature_dim, d_model)
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

        # Decoder with future temporal features
        self.future_proj = nn.Linear(future_feature_dim, d_model)

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

    def forward(self, x_hist, x_future):
        """
        Args:
            x_hist: (batch, seq_len_hist, hist_features) - Past data
            x_future: (batch, forecast_horizon, future_features) - Future temporal features

        Returns:
            da_quantiles: (batch, forecast_horizon, n_quantiles)
            rt_quantiles: (batch, forecast_horizon, n_quantiles)
        """
        # Encode historical data
        x_enc = self.hist_proj(x_hist)
        x_enc = self.pos_encoder(x_enc)
        memory = self.encoder(x_enc)  # (batch, seq_len_hist, d_model)

        # Decode with future temporal features
        x_dec = self.future_proj(x_future)  # (batch, forecast_horizon, d_model)
        x_dec = self.pos_encoder(x_dec)

        # Decoder attends to encoder output + processes future features
        decoded = self.decoder(x_dec, memory)  # (batch, forecast_horizon, d_model)

        # Predict quantiles
        da_pred = self.da_head(decoded)  # (batch, forecast_horizon, n_quantiles)
        rt_pred = self.rt_head(decoded)

        return da_pred, rt_pred


# =====================
# Data Preparation with Future Features
# =====================

def prepare_data_with_future_features(df_pl):
    """
    FIXED: Prepares both historical and future temporal features

    Returns:
        df: DataFrame with all features
        hist_features: List of historical feature names
        future_features: List of future temporal feature names (hour, dow, month)
    """
    print("Preparing unified DA + RT features with future temporal features...")

    # Convert to pandas
    df = df_pl.to_pandas()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()

    # Create price_rt from price_mean (RT price averaged across hubs)
    if 'price_rt' not in df.columns:
        if 'price_mean' in df.columns:
            df['price_rt'] = df['price_mean']
            print("  ✓ Using price_mean as price_rt")
        else:
            raise ValueError("ERROR: No price_rt or price_mean column found!")

    # price_da should already exist
    if 'price_da' not in df.columns:
        raise ValueError("ERROR: No price_da column found!")

    # Rolling statistics
    for window in [24, 168]:
        df[f'price_rt_roll_{window}h_mean'] = df['price_rt'].rolling(window).mean()
        df[f'price_rt_roll_{window}h_std'] = df['price_rt'].rolling(window).std()
        df[f'price_da_roll_{window}h_mean'] = df['price_da'].rolling(window).mean()
        df[f'price_da_roll_{window}h_std'] = df['price_da'].rolling(window).std()

    # Price spreads
    df['da_rt_spread'] = df['price_da'] - df['price_rt']
    # Avoid division by very small numbers - use larger epsilon
    df['da_rt_spread_pct'] = 100 * df['da_rt_spread'] / (np.abs(df['price_da']) + 1.0)

    # Temporal features (THESE WILL BE FUTURE FEATURES)
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)

    # Cyclical encoding for temporal features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Lag features for prices
    for price_type in ['price_da', 'price_rt']:
        for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
            df[f'{price_type}_lag{lag}'] = df[price_type].shift(lag)

    # Drop NaN from rolling windows and lags
    df = df.dropna()

    print(f"Features created. Dataset: {len(df):,} samples")

    # Data validation - check for NaN and inf
    print("\nData validation:")
    nan_counts = df.isna().sum()
    if nan_counts.sum() > 0:
        print(f"  ⚠️  WARNING: Found NaN values:")
        for col in nan_counts[nan_counts > 0].index:
            print(f"    {col}: {nan_counts[col]} NaN values")

    inf_counts = np.isinf(df.select_dtypes(include=[np.number])).sum()
    if inf_counts.sum() > 0:
        print(f"  ⚠️  WARNING: Found inf values:")
        for col in inf_counts[inf_counts > 0].index:
            print(f"    {col}: {inf_counts[col]} inf values")

    # Replace inf with NaN, then drop
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    print(f"  ✓ After cleaning inf/nan: {len(df):,} samples")

    # Define feature sets
    # Historical features: Everything except targets and raw temporal (we'll use cyclical)
    exclude_from_hist = ['price_rt', 'price_da', 'hour', 'dayofweek', 'month']
    hist_features = [col for col in df.columns if col not in exclude_from_hist]

    # Future temporal features: Only things we know in advance
    # CRITICAL: These are published ahead of time by ERCOT
    future_features = [
        'hour_sin', 'hour_cos',           # Hour-of-day cyclical
        'dow_sin', 'dow_cos',              # Day-of-week cyclical
        'month_sin', 'month_cos',          # Month cyclical
        'is_weekend',                      # Weekend indicator
        'load_forecast_mean',              # ERCOT publishes this day-ahead
    ]

    # Verify load forecast column exists
    if 'load_forecast_mean' not in df.columns:
        print("⚠️  Warning: load_forecast_mean not in dataset, excluding from future features")
        future_features = [f for f in future_features if f != 'load_forecast_mean']

    # Filter hist_features to exclude hub-level columns if present
    rt_cols = [col for col in df.columns if 'RT_' in col and 'HB_' in col]
    da_cols = [col for col in df.columns if 'DA_' in col and 'HB_' in col]
    hist_features = [f for f in hist_features if f not in rt_cols + da_cols]

    print(f"Historical features: {len(hist_features)}")
    print(f"Future temporal features: {len(future_features)}")
    print(f"Future features: {future_features}")

    # IMPORTANT: Normalize features to prevent numerical instability
    # Scale all numerical features to have mean=0, std=1
    print("\nNormalizing features...")
    scaler = StandardScaler()

    # Scale historical features
    df[hist_features] = scaler.fit_transform(df[hist_features])
    print(f"  ✓ Scaled {len(hist_features)} historical features")

    # Scale future features separately
    future_scaler = StandardScaler()
    df[future_features] = future_scaler.fit_transform(df[future_features])
    print(f"  ✓ Scaled {len(future_features)} future features")

    # Scale targets too
    target_scaler = StandardScaler()
    df[['price_da', 'price_rt']] = target_scaler.fit_transform(df[['price_da', 'price_rt']])
    print(f"  ✓ Scaled price targets")

    return df, hist_features, future_features


# =====================
# Dataset Class with Future Features
# =====================

class ForecastDatasetWithFuture(torch.utils.data.Dataset):
    """
    Returns:
        x_hist: Historical features (seq_len_hist, hist_features)
        x_future: Future temporal features (forecast_horizon, future_features)
        y_da: DA prices (forecast_horizon,)
        y_rt: RT prices (forecast_horizon,)
    """
    def __init__(self, df, hist_features, future_features, seq_len_hist=168, forecast_horizon=48):
        self.df = df
        self.hist_features = hist_features
        self.future_features = future_features
        self.seq_len_hist = seq_len_hist
        self.forecast_horizon = forecast_horizon

        # Calculate valid indices
        self.indices = list(range(seq_len_hist, len(df) - forecast_horizon))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]

        # Historical data
        x_hist = self.df[self.hist_features].iloc[actual_idx - self.seq_len_hist:actual_idx].values

        # Future temporal features
        x_future = self.df[self.future_features].iloc[actual_idx:actual_idx + self.forecast_horizon].values

        # Targets
        y_da = self.df['price_da'].iloc[actual_idx:actual_idx + self.forecast_horizon].values
        y_rt = self.df['price_rt'].iloc[actual_idx:actual_idx + self.forecast_horizon].values

        return (
            torch.FloatTensor(x_hist),
            torch.FloatTensor(x_future),
            torch.FloatTensor(y_da),
            torch.FloatTensor(y_rt)
        )


# =====================
# Quantile Loss
# =====================

def quantile_loss(preds, targets, quantiles):
    """
    Args:
        preds: (batch, horizon, n_quantiles)
        targets: (batch, horizon)
        quantiles: List of quantile levels [0.1, 0.25, 0.5, 0.75, 0.9]
    """
    errors = targets.unsqueeze(-1) - preds  # (batch, horizon, n_quantiles)
    quantiles_tensor = torch.tensor(quantiles, device=preds.device).view(1, 1, -1)
    loss = torch.maximum((quantiles_tensor - 1) * errors, quantiles_tensor * errors)
    return loss.mean()


# =====================
# Training Function
# =====================

def train_model(model, train_loader, val_loader, device, epochs=50, lr=0.001):
    """Train model with future temporal features"""
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []

        for x_hist, x_future, y_da, y_rt in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            x_hist = x_hist.to(device)
            x_future = x_future.to(device)
            y_da = y_da.to(device)
            y_rt = y_rt.to(device)

            optimizer.zero_grad()

            da_pred, rt_pred = model(x_hist, x_future)

            loss_da = quantile_loss(da_pred, y_da, quantiles)
            loss_rt = quantile_loss(rt_pred, y_rt, quantiles)
            loss = loss_da + loss_rt

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []

        with torch.no_grad():
            for x_hist, x_future, y_da, y_rt in val_loader:
                x_hist = x_hist.to(device)
                x_future = x_future.to(device)
                y_da = y_da.to(device)
                y_rt = y_rt.to(device)

                da_pred, rt_pred = model(x_hist, x_future)

                loss_da = quantile_loss(da_pred, y_da, quantiles)
                loss_rt = quantile_loss(rt_pred, y_rt, quantiles)
                loss = loss_da + loss_rt

                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        scheduler.step(val_loss)

        print(f'Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}')

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
            }, '../models/unified_da_rt_with_future_best.pth')
            print(f'  ✓ Saved new best model')
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print('  Early stopping triggered')
                break

    return model


# =====================
# Main Training
# =====================

if __name__ == "__main__":
    DATA_FILE = "/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/master_enhanced_with_net_load_reserves_2019_2025.parquet"

    print("Loading data...")
    df_pl = pl.read_parquet(DATA_FILE)

    df, hist_features, future_features = prepare_data_with_future_features(df_pl)

    # Train/val split
    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx]
    df_val = df.iloc[split_idx:]

    # Create datasets
    train_dataset = ForecastDatasetWithFuture(df_train, hist_features, future_features)
    val_dataset = ForecastDatasetWithFuture(df_val, hist_features, future_features)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = UnifiedDART_WithFutureFeatures(
        hist_feature_dim=len(hist_features),
        future_feature_dim=len(future_features),
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        n_quantiles=5,
        forecast_horizon=48,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    model = train_model(model, train_loader, val_loader, device, epochs=50, lr=0.001)

    print("\n✓ Training complete!")
    print("Model saved to: ../models/unified_da_rt_with_future_best.pth")

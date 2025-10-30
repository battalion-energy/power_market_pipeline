#!/usr/bin/env python3
"""
Quick 48-Hour Price Forecasting Model for Mercuria Demo
Simplified LSTM that trains in 1-2 hours on RTX 4070

Predicts hourly prices for next 48 hours with confidence intervals
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pickle

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")


class PriceDataset(Dataset):
    """Dataset for 48-hour price forecasting."""

    def __init__(self, data_df, lookback_hours=168, forecast_hours=48, hub='HB_HOUSTON'):
        """
        Args:
            data_df: DataFrame with datetime index and price columns
            lookback_hours: Historical window (168 = 7 days)
            forecast_hours: Future prediction window (48 = 2 days)
            hub: Price hub to predict
        """
        self.data = data_df.copy()
        self.lookback = lookback_hours
        self.forecast = forecast_hours
        self.hub = hub

        # Features to use
        self.feature_cols = [
            # RT prices (current hub)
            f'{hub}_mean', f'{hub}_max', f'{hub}_std',
            # Temporal
            'hour_of_day', 'day_of_week', 'month',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
            'is_weekend',
        ]

        # Add other hubs as features
        for other_hub in ['HB_NORTH', 'HB_SOUTH', 'HB_WEST']:
            if f'{other_hub}_mean' in self.data.columns:
                self.feature_cols.append(f'{other_hub}_mean')

        # Add DA prices if available
        if f'{hub}_da' in self.data.columns:
            self.feature_cols.append(f'{hub}_da')

        # Add AS prices if available
        as_cols = ['REGUP', 'REGDN', 'RRS', 'NSPIN', 'ECRS']
        for as_col in as_cols:
            if as_col in self.data.columns:
                self.feature_cols.append(as_col)

        # Filter to available columns
        self.feature_cols = [c for c in self.feature_cols if c in self.data.columns]

        # Target: RT price to forecast
        self.target_col = f'{hub}_mean'

        # Normalize features
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

        self.data[self.feature_cols] = self.scaler_X.fit_transform(self.data[self.feature_cols])
        self.data[[self.target_col]] = self.scaler_y.fit_transform(self.data[[self.target_col]])

        logger.info(f"Dataset: {len(self.data)} hours")
        logger.info(f"Features: {len(self.feature_cols)}")
        logger.info(f"Lookback: {lookback_hours}h, Forecast: {forecast_hours}h")

    def __len__(self):
        return len(self.data) - self.lookback - self.forecast

    def __getitem__(self, idx):
        # Historical features (lookback window)
        hist_start = idx
        hist_end = idx + self.lookback
        X_hist = self.data.iloc[hist_start:hist_end][self.feature_cols].values

        # Future features (known at forecast time - just temporal for now)
        fut_start = hist_end
        fut_end = hist_end + self.forecast
        temporal_cols = ['hour_of_day', 'day_of_week', 'month',
                        'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
                        'is_weekend']
        temporal_cols = [c for c in temporal_cols if c in self.feature_cols]
        X_fut = self.data.iloc[fut_start:fut_end][temporal_cols].values

        # Target prices (next 48 hours)
        y = self.data.iloc[fut_start:fut_end][self.target_col].values

        return (
            torch.FloatTensor(X_hist),
            torch.FloatTensor(X_fut),
            torch.FloatTensor(y)
        )


class QuickLSTM48h(nn.Module):
    """
    Simplified LSTM for 48-hour price forecasting.

    Architecture:
    - Encoder LSTM: Process historical data (168h)
    - Decoder LSTM: Generate future predictions (48h)
    - Linear output: One price per hour
    """

    def __init__(self, hist_features, fut_features, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()

        # Encoder: Process historical data
        self.encoder = nn.LSTM(
            input_size=hist_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Decoder: Generate future predictions
        self.decoder = nn.LSTM(
            input_size=fut_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Output layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x_hist, x_fut):
        # Encode historical context
        _, (h_n, c_n) = self.encoder(x_hist)

        # Decode future predictions using historical context
        decoder_out, _ = self.decoder(x_fut, (h_n, c_n))

        # Generate predictions for each hour
        predictions = self.fc(decoder_out).squeeze(-1)

        return predictions


def train_model(train_loader, val_loader, model, epochs=30, lr=1e-3):
    """Train the model."""
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_loader)
    )

    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for x_hist, x_fut, y in train_loader:
            x_hist = x_hist.to(device)
            x_fut = x_fut.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_pred = model(x_hist, x_fut)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_hist, x_fut, y in val_loader:
                x_hist = x_hist.to(device)
                x_fut = x_fut.to(device)
                y = y.to(device)

                y_pred = model(x_hist, x_fut)
                loss = criterion(y_pred, y)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, 'models/da_price_48h_best.pth')
            logger.info(f"  ✓ Saved best model (val_loss: {val_loss:.6f})")

    return model, history


def plot_training_history(history):
    """Plot training curves."""
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.title('48-Hour Price Forecast Training')
    plt.savefig('training_48h_forecast.png', dpi=150, bbox_inches='tight')
    logger.info("Saved training plot: training_48h_forecast.png")


def evaluate_model(model, test_loader, dataset):
    """Evaluate model and calculate MAE by forecast horizon."""
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x_hist, x_fut, y in test_loader:
            x_hist = x_hist.to(device)
            x_fut = x_fut.to(device)

            y_pred = model(x_hist, x_fut)

            all_preds.append(y_pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Denormalize predictions
    all_preds_orig = dataset.scaler_y.inverse_transform(all_preds.reshape(-1, 1)).reshape(all_preds.shape)
    all_targets_orig = dataset.scaler_y.inverse_transform(all_targets.reshape(-1, 1)).reshape(all_targets.shape)

    # Calculate MAE by forecast horizon
    logger.info("\n" + "="*60)
    logger.info("EVALUATION RESULTS")
    logger.info("="*60)

    overall_mae = np.abs(all_preds_orig - all_targets_orig).mean()
    logger.info(f"Overall MAE: ${overall_mae:.2f}/MWh")

    # MAE by forecast horizon
    for h in [1, 6, 12, 24, 48]:
        if h <= all_preds_orig.shape[1]:
            mae_h = np.abs(all_preds_orig[:, h-1] - all_targets_orig[:, h-1]).mean()
            logger.info(f"MAE at {h}h ahead: ${mae_h:.2f}/MWh")

    logger.info("="*60 + "\n")

    return all_preds_orig, all_targets_orig


def main():
    """Main training pipeline."""
    logger.info("="*80)
    logger.info("48-HOUR PRICE FORECASTING MODEL - TRAINING")
    logger.info("="*80)

    # Load master dataset
    data_file = Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/master_ml_dataset_2019_2025.parquet")

    if not data_file.exists():
        logger.error(f"Master dataset not found: {data_file}")
        logger.info("Run prepare_ml_data.py first to create master dataset")
        return

    logger.info(f"Loading data from {data_file}")
    df = pd.read_parquet(data_file)
    logger.info(f"Loaded {len(df):,} hourly records")
    logger.info(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")

    # Sort by datetime
    df = df.sort_values('datetime').reset_index(drop=True)

    # Train/Val/Test split (chronological)
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    logger.info(f"\nData split:")
    logger.info(f"  Train: {len(train_df):,} hours ({train_df['datetime'].min()} to {train_df['datetime'].max()})")
    logger.info(f"  Val:   {len(val_df):,} hours ({val_df['datetime'].min()} to {val_df['datetime'].max()})")
    logger.info(f"  Test:  {len(test_df):,} hours ({test_df['datetime'].min()} to {test_df['datetime'].max()})")

    # Create datasets
    train_dataset = PriceDataset(train_df, lookback_hours=168, forecast_hours=48)
    val_dataset = PriceDataset(val_df, lookback_hours=168, forecast_hours=48)
    test_dataset = PriceDataset(test_df, lookback_hours=168, forecast_hours=48)

    # Save scalers for inference
    Path('models').mkdir(exist_ok=True)
    with open('models/price_scalers.pkl', 'wb') as f:
        pickle.dump({
            'scaler_X': train_dataset.scaler_X,
            'scaler_y': train_dataset.scaler_y,
            'feature_cols': train_dataset.feature_cols,
            'hub': train_dataset.hub
        }, f)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Create model
    hist_features = len(train_dataset.feature_cols)
    fut_features = len(['hour_of_day', 'day_of_week', 'month',
                       'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
                       'is_weekend'])

    model = QuickLSTM48h(hist_features=hist_features, fut_features=fut_features).to(device)
    logger.info(f"\nModel created: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Train model
    logger.info("\nStarting training...")
    model, history = train_model(train_loader, val_loader, model, epochs=30, lr=1e-3)

    # Plot training history
    plot_training_history(history)

    # Evaluate on test set
    logger.info("\nEvaluating on test set...")
    test_preds, test_targets = evaluate_model(model, test_loader, test_dataset)

    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*80)
    logger.info(f"Best model saved: models/da_price_48h_best.pth")
    logger.info(f"Scalers saved: models/price_scalers.pkl")
    logger.info(f"Ready for demo inference!")
    logger.info("="*80 + "\n")


if __name__ == "__main__":
    main()

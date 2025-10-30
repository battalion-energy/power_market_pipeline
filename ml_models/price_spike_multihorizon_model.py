"""
Multi-Horizon Price Spike Prediction Model

Predicts spike probabilities for next 1-48 hours to support DAM bidding decisions.

Architecture:
- Input: 59 market/weather features
- Output: 144 binary probabilities (48 hours × 3 targets: high/low/extreme)
- Transformer-based with shared encoder and multiple prediction heads

Use Case:
At 10am (DAM close), battery needs to allocate capacity:
- High spike probability → reserve for RT market
- Low spike probability → bid into DAM/AS markets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score
import matplotlib.pyplot as plt


class MultiHorizonDataset(Dataset):
    """Dataset for multi-horizon (1-48h) price spike prediction."""

    def __init__(self, features_df: pd.DataFrame, sequence_length: int = 12):
        """
        Args:
            features_df: Master feature DataFrame with all features and labels
            sequence_length: Number of hourly intervals to use as sequence (12 = 12 hours)
        """
        self.sequence_length = sequence_length
        self.data = features_df.copy()

        # Feature columns (59 base features)
        self.feature_cols = [
            # Price features
            'price_mean', 'price_min', 'price_max', 'price_std',
            'price_volatility', 'price_range', 'price_change_intra',

            # DA-RT spread
            'price_da', 'da_rt_spread', 'da_rt_spread_pct',

            # AS prices (ECRS excluded - only available since June 2023)
            'REGUP', 'REGDN', 'RRS', 'NSPIN', 'as_total', 'as_vs_rt_spread',

            # Weather features (24 total)
            'temp_avg', 'temp_max_hourly', 'temp_min_hourly', 'temp_std_cities',
            'temp_max_daily', 'temp_min_daily', 'temp_range_daily',
            'humidity_avg', 'precip_total',
            'wind_speed_avg', 'wind_speed_max', 'wind_speed_std',
            'solar_irrad_avg', 'solar_irrad_max', 'solar_irrad_std', 'solar_irrad_clear_sky',
            'cloud_cover', 'cloud_cover_pct',
            'heat_wave', 'cold_snap',
            'cooling_degree_days', 'heating_degree_days',

            # Time features
            'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
            'day_of_week_sin', 'day_of_week_cos',
            'is_weekend', 'is_peak_hour', 'season',
            'year', 'quarter', 'month',
            'post_winter_storm', 'high_renewable_era', 'years_since_2019',
        ]

        # Filter to available columns
        self.feature_cols = [c for c in self.feature_cols if c in self.data.columns]

        # Label columns (144 total: 48 horizons × 3 targets)
        self.label_cols_high = [f'spike_high_{h}h' for h in range(1, 49)]
        self.label_cols_low = [f'spike_low_{h}h' for h in range(1, 49)]
        self.label_cols_extreme = [f'spike_extreme_{h}h' for h in range(1, 49)]
        self.all_label_cols = self.label_cols_high + self.label_cols_low + self.label_cols_extreme

        # Verify all labels exist
        missing_labels = [c for c in self.all_label_cols if c not in self.data.columns]
        if missing_labels:
            print(f"⚠️  Missing labels: {len(missing_labels)}")
            print(f"   First few: {missing_labels[:5]}")

        # Handle NaN and outliers in features
        print(f"\nCleaning features:")
        for col in self.feature_cols:
            # Fill NaN with median ONLY for truly missing data (holidays/gaps)
            # DO NOT fill ECRS - it's excluded because it didn't exist pre-2023
            if self.data[col].isna().any():
                nan_count = self.data[col].isna().sum()
                nan_pct = nan_count / len(self.data) * 100

                # Only fill if <5% NaN (holidays/gaps), otherwise skip
                if nan_pct < 5:
                    median_val = self.data[col].median()
                    self.data[col] = self.data[col].fillna(median_val)
                    print(f"  {col}: filled {nan_count} NaN ({nan_pct:.1f}%) with median {median_val:.2f}")
                else:
                    print(f"  {col}: SKIPPED - {nan_pct:.1f}% NaN (market didn't exist)")

            # Clip extreme outliers (beyond 5 std devs)
            mean = self.data[col].mean()
            std = self.data[col].std()
            if std > 0:
                lower = mean - 5 * std
                upper = mean + 5 * std
                clipped = ((self.data[col] < lower) | (self.data[col] > upper)).sum()
                if clipped > 0:
                    self.data[col] = self.data[col].clip(lower, upper)
                    print(f"  {col}: clipped {clipped} outliers")

        # Normalize features
        self.feature_mean = self.data[self.feature_cols].mean()
        self.feature_std = self.data[self.feature_cols].std()
        self.data[self.feature_cols] = (
            (self.data[self.feature_cols] - self.feature_mean) /
            (self.feature_std + 1e-8)
        )

        # Verify no NaN after normalization
        if self.data[self.feature_cols].isna().any().any():
            print("⚠️  WARNING: NaN values still present after cleaning!")
            nan_cols = self.data[self.feature_cols].columns[self.data[self.feature_cols].isna().any()].tolist()
            print(f"   Columns with NaN: {nan_cols}")

        # Calculate class imbalance for each target type
        high_rate = self.data[self.label_cols_high[0]].mean()
        low_rate = self.data[self.label_cols_low[0]].mean()
        extreme_rate = self.data[self.label_cols_extreme[0]].mean()

        print(f"Dataset: {len(self.data):,} samples")
        print(f"Features: {len(self.feature_cols)}")
        print(f"Labels: {len(self.all_label_cols)} (48 horizons × 3 targets)")
        print(f"\nClass distribution (1h ahead):")
        print(f"  High (>$400): {high_rate*100:.2f}%")
        print(f"  Low (<$20): {low_rate*100:.2f}%")
        print(f"  Extreme (>$1000): {extreme_rate*100:.2f}%")

    def __len__(self):
        return len(self.data) - self.sequence_length - 48  # Need 48 hours ahead

    def __getitem__(self, idx):
        # Get sequence of features (current time t and t-11 to t)
        feature_seq = self.data.iloc[idx:idx+self.sequence_length][self.feature_cols].values

        # Get all targets (144 binary labels)
        target_row = self.data.iloc[idx+self.sequence_length]
        targets = target_row[self.all_label_cols].values

        # Ensure targets are numeric (pandas sometimes returns object dtype)
        targets = targets.astype(np.float32)

        return (
            torch.FloatTensor(feature_seq),
            torch.FloatTensor(targets)
        )


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class MultiHorizonTransformer(nn.Module):
    """
    Multi-horizon transformer for battery trading decisions.

    Predicts spike probabilities for next 1-48 hours across 3 targets:
    - High prices (>$400): Discharge opportunities
    - Low prices (<$20): Charge opportunities
    - Extreme spikes (>$1000): Risk management

    Architecture:
    1. Shared transformer encoder (6 layers, 8 heads)
    2. Three prediction heads (one per target type)
    3. Each head outputs 48 probabilities (one per hour)
    """

    def __init__(self, input_dim: int, d_model: int = 512, nhead: int = 8,
                 num_layers: int = 6, dropout: float = 0.1):
        super().__init__()

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Shared transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Attention pooling
        self.attention_pool = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # Three prediction heads (high, low, extreme)
        # Each outputs 48 probabilities (one per horizon)
        self.head_high = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 48)  # 48 horizons
        )

        self.head_low = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 48)  # 48 horizons
        )

        self.head_extreme = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 48)  # 48 horizons
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)

        # Project input
        x = self.input_proj(x)  # (batch, seq_len, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoding
        x = self.transformer(x)  # (batch, seq_len, d_model)

        # Attention pooling
        query = x.mean(dim=1, keepdim=True)  # (batch, 1, d_model)
        pooled, _ = self.attention_pool(query, x, x)  # (batch, 1, d_model)
        pooled = pooled.squeeze(1)  # (batch, d_model)

        # Three prediction heads
        out_high = self.head_high(pooled)      # (batch, 48)
        out_low = self.head_low(pooled)        # (batch, 48)
        out_extreme = self.head_extreme(pooled)  # (batch, 48)

        # Concatenate outputs: [high_1h, ..., high_48h, low_1h, ..., low_48h, extreme_1h, ..., extreme_48h]
        output = torch.cat([out_high, out_low, out_extreme], dim=1)  # (batch, 144)

        return output


class MultiHorizonFocalLoss(nn.Module):
    """
    Focal Loss for multi-horizon multi-target prediction.

    Applies focal loss to all 144 outputs, with different alpha values
    for each target type based on class imbalance.
    """

    def __init__(self, alpha_high: float = 0.75, alpha_low: float = 0.6,
                 alpha_extreme: float = 0.8, gamma: float = 2.0):
        super().__init__()
        self.alpha_high = alpha_high
        self.alpha_low = alpha_low
        self.alpha_extreme = alpha_extreme
        self.gamma = gamma

    def forward(self, inputs, targets):
        # inputs: (batch, 144) logits
        # targets: (batch, 144) binary labels

        # Split into three groups
        inputs_high = inputs[:, :48]
        inputs_low = inputs[:, 48:96]
        inputs_extreme = inputs[:, 96:]

        targets_high = targets[:, :48]
        targets_low = targets[:, 48:96]
        targets_extreme = targets[:, 96:]

        # Calculate focal loss for each group
        loss_high = self._focal_loss(inputs_high, targets_high, self.alpha_high)
        loss_low = self._focal_loss(inputs_low, targets_low, self.alpha_low)
        loss_extreme = self._focal_loss(inputs_extreme, targets_extreme, self.alpha_extreme)

        # Weighted average (high and extreme more important than low)
        total_loss = 0.4 * loss_high + 0.2 * loss_low + 0.4 * loss_extreme

        return total_loss

    def _focal_loss(self, inputs, targets, alpha):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        probs = torch.sigmoid(inputs)
        p_t = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = alpha * focal_weight * BCE_loss
        return focal_loss.mean()


class MultiHorizonModelTrainer:
    """Train multi-horizon price spike model."""

    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
              epochs: int = 100, batch_size: int = 256, lr: float = 1e-4,
              sequence_length: int = 12) -> MultiHorizonTransformer:
        """
        Train multi-horizon model with FP16 mixed precision.

        Args:
            train_df: Training data with features and 144 labels
            val_df: Validation data
            epochs: Number of epochs
            batch_size: Batch size
            lr: Learning rate
            sequence_length: Hours of history to use
        """

        print(f"\n{'='*80}")
        print("TRAINING MULTI-HORIZON PRICE SPIKE MODEL")
        print(f"{'='*80}\n")

        # Create datasets
        train_dataset = MultiHorizonDataset(train_df, sequence_length=sequence_length)
        val_dataset = MultiHorizonDataset(val_df, sequence_length=sequence_length)

        # Data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                               shuffle=False, num_workers=4, pin_memory=True)

        # Create model
        input_dim = len(train_dataset.feature_cols)
        model = MultiHorizonTransformer(input_dim=input_dim).to(self.device)

        # Optimizer & scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_loader)
        )

        # Loss function
        criterion = MultiHorizonFocalLoss(
            alpha_high=0.75,    # High prices: 0.97% positive
            alpha_low=0.6,      # Low prices: 39.55% positive
            alpha_extreme=0.8   # Extreme: 0.55% positive
        )

        # Mixed precision training
        scaler = GradScaler()

        # Training loop
        best_auc = 0.0
        history = {'train_loss': [], 'val_loss': [], 'val_auc_high_1h': [],
                   'val_auc_high_24h': [], 'val_auc_high_48h': []}

        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            for batch_features, batch_targets in train_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)

                optimizer.zero_grad()

                with autocast():
                    output = model(batch_features)
                    loss = criterion(output, batch_targets)

                scaler.scale(loss).backward()

                # Unscale and clip gradients for stability
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            model.eval()
            val_loss = 0
            all_preds = []
            all_targets = []

            with torch.no_grad():
                for batch_features, batch_targets in val_loader:
                    batch_features = batch_features.to(self.device)
                    batch_targets = batch_targets.to(self.device)

                    with autocast():
                        output = model(batch_features)
                        loss = criterion(output, batch_targets)

                    val_loss += loss.item()
                    all_preds.append(output.cpu().numpy())
                    all_targets.append(batch_targets.cpu().numpy())

            val_loss /= len(val_loader)

            # Calculate metrics
            all_preds = np.vstack(all_preds)
            all_targets = np.vstack(all_targets)

            # Apply sigmoid to convert logits to probabilities
            all_probs = 1 / (1 + np.exp(-all_preds))

            # Calculate AUC for key horizons (high prices)
            # Only calculate if we have positive examples
            def safe_auc(targets, probs):
                if targets.sum() < 5:  # Need at least 5 positive examples
                    return 0.5  # Return random baseline
                try:
                    return roc_auc_score(targets, probs)
                except:
                    return 0.5

            auc_high_1h = safe_auc(all_targets[:, 0], all_probs[:, 0])
            auc_high_24h = safe_auc(all_targets[:, 23], all_probs[:, 23])
            auc_high_48h = safe_auc(all_targets[:, 47], all_probs[:, 47])

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_auc_high_1h'].append(auc_high_1h)
            history['val_auc_high_24h'].append(auc_high_24h)
            history['val_auc_high_48h'].append(auc_high_48h)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss: {val_loss:.4f}")
                print(f"  Val AUC (high prices):")
                print(f"    1h ahead: {auc_high_1h:.4f}")
                print(f"    24h ahead: {auc_high_24h:.4f}")
                print(f"    48h ahead: {auc_high_48h:.4f}")

            # Save best model (based on 24h AUC - key for DAM decisions)
            if auc_high_24h > best_auc:
                best_auc = auc_high_24h
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_auc_24h': auc_high_24h,
                }, 'ml_models/multihorizon_model_best.pth')

        print(f"\n{'='*80}")
        print(f"Training Complete!")
        print(f"Best Validation AUC (24h ahead): {best_auc:.4f}")
        print(f"{'='*80}\n")

        return model, history


if __name__ == "__main__":
    print("Multi-Horizon Price Spike Prediction Model")
    print("Predicts 48 hours × 3 targets = 144 binary probabilities")
    print("\nFor battery capacity allocation decisions at DAM close (10am)")

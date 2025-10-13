"""
Model 3: RT Price Spike Probability Prediction (MOST CRITICAL)

Transformer-based model to predict probability of RT price spike in next 1-6 hours.

Target Performance: AUC > 0.88 (Industry benchmark from Fluence AI)

Spike Definition (ANY triggers spike label):
1. Statistical: Price > μ + 3σ (rolling 30-day)
2. Economic: Price > $1000/MWh
3. Scarcity: ORDC adder > $500/MWh

Key Features:
- Forecast errors (load, wind, solar) - CRITICAL
- ORDC reserve metrics
- Weather extremes
- System stress indicators
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


class PriceSpikeDataset(Dataset):
    """Dataset for price spike prediction."""

    def __init__(self, features_df: pd.DataFrame, sequence_length: int = 12):
        """
        Args:
            features_df: Master feature DataFrame with all engineered features
            sequence_length: Number of 5-min intervals to use (12 = 1 hour)
        """
        self.sequence_length = sequence_length
        self.data = features_df.copy()

        # Feature columns (100-150 features)
        self.feature_cols = [
            # Forecast errors (CRITICAL)
            'load_error_mw', 'load_error_pct', 'load_error_1h', 'load_error_3h',
            'wind_error_mw', 'wind_error_pct', 'wind_error_3h', 'wind_error_6h',
            'solar_error_mw', 'solar_error_pct', 'solar_error_3h', 'solar_error_6h',

            # ORDC & Reserves (when available)
            # 'reserve_margin', 'reserve_error', 'distance_to_3000mw',
            # 'distance_to_2000mw', 'distance_to_1000mw', 'ordc_adder',

            # Weather features (27 total from NASA POWER satellite data)
            # Temperature
            'temp_avg', 'temp_max_daily', 'temp_min_daily', 'temp_std_cities', 'temp_range_daily',
            # Humidity & Precipitation
            'humidity_avg', 'precip_total',
            # Wind (from wind farm locations)
            'wind_speed_avg', 'wind_speed_min', 'wind_speed_max', 'wind_speed_std', 'wind_direction_avg',
            'wind_calm', 'wind_strong',
            # Solar (from solar farm locations)
            'solar_irrad_avg', 'solar_irrad_min', 'solar_irrad_max', 'solar_irrad_std',
            'solar_irrad_clear_sky', 'cloud_cover', 'cloud_cover_pct',
            # Demand indicators
            'cooling_degree_days', 'heating_degree_days',
            # Extreme weather indicators
            'heat_wave', 'cold_snap',

            # Net load
            'net_load', 'net_load_pct_capacity', 'net_load_ramp_1h', 'net_load_ramp_3h',
            'extreme_net_load',

            # Temporal (cyclical encoding)
            'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
            'month_sin', 'month_cos', 'is_weekend', 'season',
        ]

        # Filter to available columns
        self.feature_cols = [c for c in self.feature_cols if c in self.data.columns]

        # Normalize features
        self.feature_mean = self.data[self.feature_cols].mean()
        self.feature_std = self.data[self.feature_cols].std()
        self.data[self.feature_cols] = (
            (self.data[self.feature_cols] - self.feature_mean) /
            (self.feature_std + 1e-8)
        )

        # Target: price_spike binary label
        self.target_col = 'price_spike'

        print(f"Dataset: {len(self.data):,} samples")
        print(f"Features: {len(self.feature_cols)}")
        print(f"Spike rate: {self.data[self.target_col].mean()*100:.2f}%")

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        # Get sequence of features
        feature_seq = self.data.iloc[idx:idx+self.sequence_length][self.feature_cols].values

        # Get target (spike in next interval)
        target = self.data.iloc[idx+self.sequence_length][self.target_col]

        return (
            torch.FloatTensor(feature_seq),
            torch.FloatTensor([target])
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
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]


class PriceSpikeTransformer(nn.Module):
    """
    Transformer-based price spike prediction model.

    Architecture:
    1. Input embedding
    2. Positional encoding
    3. Transformer encoder (6 layers, 8 heads)
    4. Multi-head attention pooling
    5. Binary classifier
    """

    def __init__(self, input_dim: int, d_model: int = 512, nhead: int = 8,
                 num_layers: int = 6, dropout: float = 0.1):
        super().__init__()

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Multi-head attention pooling
        self.attention_pool = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
            # No sigmoid - will use BCEWithLogitsLoss instead
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)

        # Project input
        x = self.input_proj(x)  # (batch, seq_len, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoding
        x = self.transformer(x)  # (batch, seq_len, d_model)

        # Attention pooling (aggregate sequence)
        # Use mean of sequence as query
        query = x.mean(dim=1, keepdim=True)  # (batch, 1, d_model)
        pooled, _ = self.attention_pool(query, x, x)  # (batch, 1, d_model)
        pooled = pooled.squeeze(1)  # (batch, d_model)

        # Classifier
        output = self.classifier(pooled)  # (batch, 1)

        return output


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    focal_loss = -α(1-p_t)^γ * log(p_t)

    where:
    - α: class weight (0.75 for minority class)
    - γ: focusing parameter (2.0 to focus on hard examples)
    - p_t: predicted probability of correct class

    Uses BCEWithLogitsLoss for FP16 safety.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # inputs: (batch, 1) logits (NOT probabilities)
        # targets: (batch, 1) binary labels

        # Use BCEWithLogitsLoss for numerical stability and FP16 safety
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Convert logits to probabilities for focal weight calculation
        probs = torch.sigmoid(inputs)
        p_t = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = self.alpha * focal_weight * BCE_loss

        return focal_loss.mean()


class PriceSpikeModelTrainer:
    """Train price spike prediction model with RTX 4070 optimization."""

    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
              epochs: int = 100, batch_size: int = 256, lr: float = 1e-4,
              sequence_length: int = 12) -> PriceSpikeTransformer:
        """
        Train price spike model with mixed precision (FP16) for RTX 4070.

        Args:
            train_df: Training data with features and labels
            val_df: Validation data
            epochs: Number of epochs
            batch_size: Batch size (256-512 optimal for RTX 4070 with FP16)
            lr: Learning rate
        """

        print(f"\n{'='*80}")
        print("TRAINING RT PRICE SPIKE MODEL (Model 3)")
        print(f"{'='*80}\n")

        # Create datasets
        train_dataset = PriceSpikeDataset(train_df, sequence_length=sequence_length)
        val_dataset = PriceSpikeDataset(val_df, sequence_length=sequence_length)

        # Handle class imbalance with weighted sampling
        spike_rate = train_df['price_spike'].mean()
        class_weights = torch.FloatTensor([
            1.0 / (1 - spike_rate),  # Non-spike weight
            1.0 / spike_rate          # Spike weight
        ])

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                               shuffle=False, num_workers=4, pin_memory=True)

        # Create model
        input_dim = len(train_dataset.feature_cols)
        model = PriceSpikeTransformer(input_dim=input_dim).to(self.device)

        # Optimizer & scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_loader)
        )

        # Loss function (Focal Loss for imbalanced data)
        criterion = FocalLoss(alpha=0.75, gamma=2.0)

        # Mixed precision training (FP16 for RTX 4070)
        scaler = GradScaler()

        # Training loop
        best_auc = 0.0
        history = {'train_loss': [], 'val_loss': [], 'val_auc': [], 'val_f1': []}

        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            for batch_features, batch_target in train_loader:
                batch_features = batch_features.to(self.device)
                batch_target = batch_target.to(self.device)

                optimizer.zero_grad()

                # Forward pass with autocast (FP16)
                with autocast():
                    output = model(batch_features)
                    loss = criterion(output, batch_target)

                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
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
                for batch_features, batch_target in val_loader:
                    batch_features = batch_features.to(self.device)
                    batch_target = batch_target.to(self.device)

                    with autocast():
                        output = model(batch_features)
                        loss = criterion(output, batch_target)

                    val_loss += loss.item()
                    all_preds.extend(output.cpu().numpy())
                    all_targets.extend(batch_target.cpu().numpy())

            val_loss /= len(val_loader)

            # Calculate metrics
            all_preds = np.array(all_preds).flatten()
            all_targets = np.array(all_targets).flatten()

            # Apply sigmoid to convert logits to probabilities
            all_probs = 1 / (1 + np.exp(-all_preds))

            val_auc = roc_auc_score(all_targets, all_probs)
            val_f1 = f1_score(all_targets, (all_probs > 0.5).astype(int))

            # Precision@5% (top 5% predicted spikes)
            threshold_95 = np.percentile(all_probs, 95)
            precision_at_5 = (all_targets[all_probs > threshold_95]).mean()

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_auc'].append(val_auc)
            history['val_f1'].append(val_f1)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss: {val_loss:.4f}")
                print(f"  Val AUC: {val_auc:.4f} (Target: > 0.88)")
                print(f"  Val F1: {val_f1:.4f}")
                print(f"  Precision@5%: {precision_at_5:.4f}")

            # Save best model
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_auc': val_auc,
                    'val_f1': val_f1,
                }, 'models/price_spike_model_best.pth')

        print(f"\n{'='*80}")
        print(f"Training Complete!")
        print(f"Best Validation AUC: {best_auc:.4f}")
        print(f"Target AUC: > 0.88 (Industry Benchmark)")
        print(f"{'='*80}\n")

        return model, history

    def plot_training_history(self, history: Dict[str, List]):
        """Plot training metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        axes[0, 0].plot(history['train_loss'], label='Train Loss')
        axes[0, 0].plot(history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].legend()

        axes[0, 1].plot(history['val_auc'], label='Val AUC')
        axes[0, 1].axhline(y=0.88, color='r', linestyle='--', label='Target (0.88)')
        axes[0, 1].set_title('Validation AUC')
        axes[0, 1].legend()

        axes[1, 0].plot(history['val_f1'], label='Val F1')
        axes[1, 0].set_title('Validation F1 Score')
        axes[1, 0].legend()

        plt.tight_layout()
        plt.savefig('training_history_price_spike.png', dpi=150)
        print("📊 Saved training history plot")


if __name__ == "__main__":
    print("RT Price Spike Prediction Model (Model 3)")
    print("Transformer-based architecture optimized for RTX 4070")
    print("\nTarget Performance: AUC > 0.88 (Fluence AI Benchmark)")
    print("\nReady to train once feature engineering is complete!")

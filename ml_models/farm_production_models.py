"""
Wind/Solar Farm Production Prediction Models

These small specialized models predict production for individual wind farms and solar farms
given weather conditions. Their predictions feed into the larger price forecasting models.

Key Benefits:
1. Weather → Production mapping for each farm
2. More accurate than system-wide forecasts
3. Captures local weather effects
4. Feeds into price spike prediction
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


class WeatherToProductionDataset(Dataset):
    """Dataset for weather → production mapping."""

    def __init__(self, weather_data: pd.DataFrame, production_data: pd.DataFrame,
                 sequence_length: int = 24):
        """
        Args:
            weather_data: Weather features (temp, wind_speed, solar_irradiance, etc.)
            production_data: Actual production data
            sequence_length: Hours of history to use
        """
        self.sequence_length = sequence_length

        # Merge weather and production on timestamp
        self.data = pd.merge(weather_data, production_data,
                            on='timestamp', how='inner')

        # Features: temp, humidity, wind_speed, pressure, cloud_cover, etc.
        self.weather_features = [
            'temp', 'dwpt', 'rhum', 'wspd', 'wdir', 'pres'
        ]

        # Normalize features
        self.feature_mean = self.data[self.weather_features].mean()
        self.feature_std = self.data[self.weather_features].std()
        self.data[self.weather_features] = (
            (self.data[self.weather_features] - self.feature_mean) / self.feature_std
        )

        # Normalize production
        self.prod_mean = self.data['production'].mean()
        self.prod_std = self.data['production'].std()
        self.data['production_normalized'] = (
            (self.data['production'] - self.prod_mean) / self.prod_std
        )

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        # Get sequence of weather features
        weather_seq = self.data.iloc[idx:idx+self.sequence_length][self.weather_features].values

        # Get target production (next hour)
        target_production = self.data.iloc[idx+self.sequence_length]['production_normalized']

        return (
            torch.FloatTensor(weather_seq),
            torch.FloatTensor([target_production])
        )


class WindFarmProductionModel(nn.Module):
    """
    LSTM-based model to predict wind farm production from weather.

    Input: Weather sequence (temp, wind speed, direction, pressure, etc.)
    Output: Wind farm production (MW)
    """

    def __init__(self, input_dim: int = 6, hidden_dim: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )

        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout)

        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        # x: (batch, sequence_length, input_dim)

        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Self-attention on LSTM outputs
        # Need to permute for attention: (seq_len, batch, hidden_dim)
        lstm_out = lstm_out.permute(1, 0, 2)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = attn_out.permute(1, 0, 2)  # Back to (batch, seq, hidden)

        # Take last timestep
        final_hidden = attn_out[:, -1, :]

        # FC layers
        output = self.fc_layers(final_hidden)

        return output


class SolarFarmProductionModel(nn.Module):
    """
    CNN-LSTM model to predict solar farm production from weather.

    Solar production has different patterns than wind:
    - More dependent on cloud cover, solar angle
    - Diurnal patterns (zero at night)
    - Faster ramps with cloud movement
    """

    def __init__(self, input_dim: int = 6, cnn_channels: int = 32,
                 lstm_hidden: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()

        # 1D CNN to capture short-term patterns
        self.conv1 = nn.Conv1d(input_dim, cnn_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1)

        # LSTM for temporal dependencies
        self.lstm = nn.LSTM(
            cnn_channels, lstm_hidden, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )

        # Hour embedding (solar has strong diurnal pattern)
        self.hour_embed = nn.Embedding(24, 8)

        # FC layers
        self.fc_layers = nn.Sequential(
            nn.Linear(lstm_hidden + 8, 32),  # LSTM + hour embedding
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
            nn.ReLU()  # Solar production >= 0
        )

    def forward(self, x, hour):
        # x: (batch, sequence_length, input_dim)
        # hour: (batch,) hour of day

        # CNN expects (batch, channels, sequence)
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.permute(0, 2, 1)  # Back to (batch, seq, channels)

        # LSTM
        lstm_out, _ = self.lstm(x)
        final_hidden = lstm_out[:, -1, :]

        # Hour embedding
        hour_emb = self.hour_embed(hour)

        # Concatenate LSTM output and hour embedding
        combined = torch.cat([final_hidden, hour_emb], dim=1)

        # FC layers
        output = self.fc_layers(combined)

        return output


class FarmProductionModelTrainer:
    """Train wind/solar farm production models."""

    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

    def train_wind_farm_model(self, farm_name: str, weather_data: pd.DataFrame,
                              production_data: pd.DataFrame,
                              epochs: int = 50, batch_size: int = 128) -> WindFarmProductionModel:
        """Train a wind farm production model."""

        print(f"\n{'='*80}")
        print(f"Training Wind Farm Model: {farm_name}")
        print(f"{'='*80}")

        # Create dataset
        dataset = WeatherToProductionDataset(weather_data, production_data)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Create model
        model = WindFarmProductionModel().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # Training loop
        best_val_loss = float('inf')
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for batch_weather, batch_target in train_loader:
                batch_weather = batch_weather.to(self.device)
                batch_target = batch_target.to(self.device)

                optimizer.zero_grad()
                output = model(batch_weather)
                loss = criterion(output, batch_target)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_weather, batch_target in val_loader:
                    batch_weather = batch_weather.to(self.device)
                    batch_target = batch_target.to(self.device)

                    output = model(batch_weather)
                    loss = criterion(output, batch_target)
                    val_loss += loss.item()

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f'models/{farm_name}_wind_model.pth')

        print(f"✅ Training complete. Best val loss: {best_val_loss:.4f}")
        return model

    def train_solar_farm_model(self, farm_name: str, weather_data: pd.DataFrame,
                               production_data: pd.DataFrame,
                               epochs: int = 50, batch_size: int = 128) -> SolarFarmProductionModel:
        """Train a solar farm production model."""

        print(f"\n{'='*80}")
        print(f"Training Solar Farm Model: {farm_name}")
        print(f"{'='*80}")

        # Similar to wind farm training
        # This is a placeholder - will need hour-of-day feature
        model = SolarFarmProductionModel().to(self.device)

        print(f"⚠️  Solar farm training - needs hour-of-day feature in dataset")
        return model

    def predict_all_farms(self, weather_forecast: pd.DataFrame,
                         wind_farms: Dict[str, WindFarmProductionModel],
                         solar_farms: Dict[str, SolarFarmProductionModel]) -> pd.DataFrame:
        """
        Predict production for all wind and solar farms.

        This feeds into the price forecasting models as enhanced features.
        """
        predictions = []

        # Wind farms
        for farm_name, model in wind_farms.items():
            model.eval()
            with torch.no_grad():
                # Prepare weather data
                # ... feature engineering ...
                # farm_pred = model(weather_features)
                pass

        # Solar farms
        for farm_name, model in solar_farms.items():
            model.eval()
            with torch.no_grad():
                # Prepare weather data + hour
                # farm_pred = model(weather_features, hour)
                pass

        return pd.DataFrame(predictions)


if __name__ == "__main__":
    # Example usage
    print("Wind/Solar Farm Production Models")
    print("These models will be trained once farm-level data is available")
    print("They enhance the price forecasting models with detailed renewable predictions")

"""
Feature Engineering Pipeline for ERCOT Price Forecasting

This module creates all features for the three ML models:
- Model 1: DA Price Forecasting
- Model 2: RT Price Forecasting
- Model 3: RT Price Spike Probability (CRITICAL)

Key Features:
1. Forecast Error Features (load, wind, solar)
2. ORDC & Reserve Metrics
3. Weather Extremes
4. System Stress Indicators
5. Temporal Features
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class ERCOTFeatureEngineer:
    """Feature engineering for ERCOT price forecasting."""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)

    def load_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load all available datasets with placeholders for incomplete downloads."""
        datasets = {}

        # Wind power production (actual + STWPF forecasts)
        try:
            wind_files = sorted((self.data_dir / "Wind_Power_Production").glob("*.csv"))
            if wind_files:
                datasets['wind'] = pd.concat([pd.read_csv(f) for f in wind_files])
                print(f"✅ Loaded wind data: {len(datasets['wind']):,} records")
        except Exception as e:
            print(f"⚠️  Wind data not ready: {e}")
            datasets['wind'] = pd.DataFrame()  # Placeholder

        # Solar power production (actual + STPPF forecasts)
        try:
            solar_files = sorted((self.data_dir / "Solar_Power_Production").glob("*.csv"))
            if solar_files:
                datasets['solar'] = pd.concat([pd.read_csv(f) for f in solar_files])
                print(f"✅ Loaded solar data: {len(datasets['solar']):,} records")
        except Exception as e:
            print(f"⚠️  Solar data not ready: {e}")
            datasets['solar'] = pd.DataFrame()

        # Load forecasts
        try:
            load_files = sorted((self.data_dir / "Load_Forecast_By_Weather_Zone").glob("*.csv"))
            if load_files:
                datasets['load_forecast'] = pd.concat([pd.read_csv(f) for f in load_files])
                print(f"✅ Loaded load forecast: {len(datasets['load_forecast']):,} records")
        except Exception as e:
            print(f"⚠️  Load forecast not ready: {e}")
            datasets['load_forecast'] = pd.DataFrame()

        # Actual load
        try:
            actual_load_files = sorted((self.data_dir / "Actual_System_Load_By_Weather_Zone").glob("*.csv"))
            if actual_load_files:
                datasets['actual_load'] = pd.concat([pd.read_csv(f) for f in actual_load_files])
                print(f"✅ Loaded actual load: {len(datasets['actual_load']):,} records")
        except Exception as e:
            print(f"⚠️  Actual load not ready: {e}")
            datasets['actual_load'] = pd.DataFrame()

        # Fuel mix
        try:
            fuel_files = sorted((self.data_dir / "Fuel_Mix").glob("*.csv"))
            if fuel_files:
                datasets['fuel_mix'] = pd.concat([pd.read_csv(f) for f in fuel_files])
                print(f"✅ Loaded fuel mix: {len(datasets['fuel_mix']):,} records")
        except Exception as e:
            print(f"⚠️  Fuel mix not ready: {e}")
            datasets['fuel_mix'] = pd.DataFrame()

        # System demand
        try:
            demand_files = sorted((self.data_dir / "System_Wide_Demand").glob("*.csv"))
            if demand_files:
                datasets['system_demand'] = pd.concat([pd.read_csv(f) for f in demand_files])
                print(f"✅ Loaded system demand: {len(datasets['system_demand']):,} records")
        except Exception as e:
            print(f"⚠️  System demand not ready: {e}")
            datasets['system_demand'] = pd.DataFrame()

        # DA/RT Prices (targets)
        try:
            dam_files = sorted((self.data_dir / "DAM_Settlement_Point_Prices").glob("*.csv"))
            if dam_files:
                datasets['dam_prices'] = pd.concat([pd.read_csv(f) for f in dam_files])
                print(f"✅ Loaded DAM prices: {len(datasets['dam_prices']):,} records")
        except Exception as e:
            print(f"⚠️  DAM prices not ready: {e}")
            datasets['dam_prices'] = pd.DataFrame()

        return datasets

    def calculate_forecast_errors(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate forecast error features - CRITICAL for price spike prediction.

        Forecast errors drive RT price spikes:
        1. Load forecast error (unexpected demand)
        2. Wind forecast error (generation shortfall)
        3. Solar forecast error (cloud cover surprise)
        """
        features = []

        if not datasets['wind'].empty:
            # Wind forecast error (multiple columns in data)
            # Column format from API: actual GEN, STWPF forecast, etc.
            wind_df = datasets['wind'].copy()

            # Extract system-wide values (need to parse column structure)
            # This is a placeholder - will need actual column mapping
            if 'genSystemWide' in wind_df.columns and 'STWPFSystemWide' in wind_df.columns:
                wind_df['wind_error_mw'] = wind_df['genSystemWide'] - wind_df['STWPFSystemWide']
                wind_df['wind_error_pct'] = (wind_df['wind_error_mw'] / wind_df['STWPFSystemWide'].replace(0, np.nan)) * 100

                # Cumulative errors (compounding effect)
                wind_df['wind_error_3h'] = wind_df['wind_error_mw'].rolling(3).sum()
                wind_df['wind_error_6h'] = wind_df['wind_error_mw'].rolling(6).sum()

                features.append(wind_df[['timestamp', 'wind_error_mw', 'wind_error_pct', 'wind_error_3h', 'wind_error_6h']])

        if not datasets['solar'].empty:
            # Solar forecast error
            solar_df = datasets['solar'].copy()

            if 'genSystemWide' in solar_df.columns and 'STPPFSystemWide' in solar_df.columns:
                solar_df['solar_error_mw'] = solar_df['genSystemWide'] - solar_df['STPPFSystemWide']
                solar_df['solar_error_pct'] = (solar_df['solar_error_mw'] / solar_df['STPPFSystemWide'].replace(0, np.nan)) * 100

                # Cumulative errors
                solar_df['solar_error_3h'] = solar_df['solar_error_mw'].rolling(3).sum()
                solar_df['solar_error_6h'] = solar_df['solar_error_mw'].rolling(6).sum()

                features.append(solar_df[['timestamp', 'solar_error_mw', 'solar_error_pct', 'solar_error_3h', 'solar_error_6h']])

        if not datasets['load_forecast'].empty and not datasets['actual_load'].empty:
            # Load forecast error
            load_forecast = datasets['load_forecast'].copy()
            actual_load = datasets['actual_load'].copy()

            # Merge on timestamp
            load_merged = pd.merge(actual_load, load_forecast, on='timestamp', suffixes=('_actual', '_forecast'))

            if 'load_actual' in load_merged.columns and 'load_forecast' in load_merged.columns:
                load_merged['load_error_mw'] = load_merged['load_actual'] - load_merged['load_forecast']
                load_merged['load_error_pct'] = (load_merged['load_error_mw'] / load_merged['load_forecast'].replace(0, np.nan)) * 100

                # Cumulative errors (5-min intervals, so 12 per hour)
                load_merged['load_error_1h'] = load_merged['load_error_mw'].rolling(12).sum()
                load_merged['load_error_3h'] = load_merged['load_error_mw'].rolling(36).sum()
                load_merged['load_error_6h'] = load_merged['load_error_mw'].rolling(72).sum()

                features.append(load_merged[['timestamp', 'load_error_mw', 'load_error_pct', 'load_error_1h', 'load_error_3h', 'load_error_6h']])

        # Combine all forecast error features
        if features:
            forecast_errors = features[0]
            for feat in features[1:]:
                forecast_errors = pd.merge(forecast_errors, feat, on='timestamp', how='outer')
            return forecast_errors

        return pd.DataFrame()

    def calculate_ordc_features(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate ORDC (Operating Reserve Demand Curve) features.

        ORDC drives scarcity pricing in ERCOT:
        - Reserve margin % (critical indicator)
        - Distance to thresholds (3000, 2000, 1000 MW)
        - Reserve error (HA forecast vs RT actual)
        - ORDC price adder calculation
        """
        ordc_features = pd.DataFrame()

        # This requires reserve data (not in current download list)
        # Placeholder for when reserve data is available
        # Will calculate:
        # - reserve_margin = (online_reserves / system_load) * 100
        # - distance_to_3000mw, distance_to_2000mw, distance_to_1000mw
        # - reserve_error = ha_forecast - rt_actual
        # - ordc_adder = calculate_ordc_price(reserves, voll=9000)

        print("⚠️  ORDC features require reserve data (add to download list)")
        return ordc_features

    def calculate_weather_features(self, weather_dir: Path) -> pd.DataFrame:
        """
        Calculate weather extreme features from Texas city data.

        Weather extremes drive price spikes:
        - Heat waves (>100°F for 3+ hours)
        - Cold snaps (<20°F)
        - Temperature deviation from normal
        - Weather forecast errors
        """
        weather_features = []

        # Texas major cities
        texas_cities = ['houston', 'dallas', 'austin', 'san_antonio', 'el_paso',
                       'fort_worth', 'corpus_christi', 'laredo', 'amarillo', 'lubbock']

        for city in texas_cities:
            city_files = list(weather_dir.glob(f"{city}_*.csv"))
            if not city_files:
                continue

            for file in city_files:
                try:
                    df = pd.read_csv(file)
                    df['timestamp'] = pd.to_datetime(df['time'])
                    df['city'] = city

                    # Heat wave indicator
                    df['temp_f'] = df['temp'] * 9/5 + 32  # Convert C to F
                    df['heat_wave'] = (df['temp_f'] > 100).rolling(3).sum() >= 3

                    # Cold snap indicator
                    df['cold_snap'] = (df['temp_f'] < 20).rolling(6).sum() >= 6

                    # Temperature changes (rapid weather fronts)
                    df['temp_change_1h'] = df['temp'].diff(1)
                    df['temp_change_3h'] = df['temp'].diff(3)
                    df['temp_change_6h'] = df['temp'].diff(6)

                    # Wind speed (affects wind generation)
                    if 'wspd' in df.columns:
                        df['wind_speed_change_3h'] = df['wspd'].diff(3)

                    weather_features.append(df[['timestamp', 'city', 'temp', 'temp_f', 'heat_wave',
                                                 'cold_snap', 'temp_change_1h', 'temp_change_3h',
                                                 'temp_change_6h', 'wspd']])
                except Exception as e:
                    print(f"Error loading {file}: {e}")

        if weather_features:
            return pd.concat(weather_features, ignore_index=True)

        return pd.DataFrame()

    def calculate_net_load_features(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate net load features (load - renewables).

        Net load as % of capacity is a key scarcity indicator.
        """
        if datasets['actual_load'].empty:
            return pd.DataFrame()

        net_load_features = datasets['actual_load'].copy()

        # Assume thermal capacity of ~80,000 MW (ERCOT typical)
        THERMAL_CAPACITY = 80000

        if not datasets['wind'].empty and not datasets['solar'].empty:
            # Merge wind and solar generation
            wind = datasets['wind'][['timestamp', 'genSystemWide']].rename(columns={'genSystemWide': 'wind_gen'})
            solar = datasets['solar'][['timestamp', 'genSystemWide']].rename(columns={'genSystemWide': 'solar_gen'})

            net_load_features = net_load_features.merge(wind, on='timestamp', how='left')
            net_load_features = net_load_features.merge(solar, on='timestamp', how='left')

            # Calculate net load
            net_load_features['net_load'] = net_load_features['load'] - net_load_features['wind_gen'].fillna(0) - net_load_features['solar_gen'].fillna(0)
            net_load_features['net_load_pct_capacity'] = (net_load_features['net_load'] / THERMAL_CAPACITY) * 100

            # Net load ramps (rapid changes)
            net_load_features['net_load_ramp_1h'] = net_load_features['net_load'].diff(12)  # 5-min intervals
            net_load_features['net_load_ramp_3h'] = net_load_features['net_load'].diff(36)

            # Extreme net load indicator (>90% of thermal capacity)
            net_load_features['extreme_net_load'] = (net_load_features['net_load_pct_capacity'] > 90).astype(int)

        return net_load_features[['timestamp', 'net_load', 'net_load_pct_capacity', 'net_load_ramp_1h', 'net_load_ramp_3h', 'extreme_net_load']]

    def calculate_temporal_features(self, df: pd.DataFrame, timestamp_col: str = 'timestamp') -> pd.DataFrame:
        """
        Add temporal features with cyclical encoding.

        Cyclical encoding preserves periodicity (hour 23 is close to hour 0).
        """
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])

        # Extract time components
        df['hour'] = df[timestamp_col].dt.hour
        df['day_of_week'] = df[timestamp_col].dt.dayofweek
        df['month'] = df[timestamp_col].dt.month
        df['day_of_year'] = df[timestamp_col].dt.dayofyear

        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Day type
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        # Season
        df['season'] = df['month'].apply(lambda x:
            0 if x in [12, 1, 2] else  # Winter
            1 if x in [3, 4, 5] else    # Spring
            2 if x in [6, 7, 8] else    # Summer
            3)                           # Fall

        return df

    def create_price_spike_labels(self, prices_df: pd.DataFrame,
                                  price_col: str = 'price',
                                  statistical_threshold_std: float = 3.0,
                                  economic_threshold: float = 1000.0,
                                  ordc_threshold: float = 500.0) -> pd.DataFrame:
        """
        Create binary labels for price spikes.

        Spike definition (ANY of):
        1. Statistical: price > μ + 3σ (rolling 30-day)
        2. Economic: price > $1000/MWh
        3. Scarcity: ORDC adder > $500/MWh
        """
        df = prices_df.copy()

        # Statistical threshold (rolling 30-day mean + 3*std)
        rolling_mean = df[price_col].rolling(30*24, min_periods=24).mean()  # 30 days hourly
        rolling_std = df[price_col].rolling(30*24, min_periods=24).std()
        statistical_threshold = rolling_mean + (statistical_threshold_std * rolling_std)

        df['spike_statistical'] = (df[price_col] > statistical_threshold).astype(int)

        # Economic threshold
        df['spike_economic'] = (df[price_col] > economic_threshold).astype(int)

        # ORDC threshold (if available)
        if 'ordc_adder' in df.columns:
            df['spike_ordc'] = (df['ordc_adder'] > ordc_threshold).astype(int)
        else:
            df['spike_ordc'] = 0

        # Combined spike indicator (ANY condition triggers spike)
        df['price_spike'] = ((df['spike_statistical'] == 1) |
                            (df['spike_economic'] == 1) |
                            (df['spike_ordc'] == 1)).astype(int)

        spike_pct = df['price_spike'].mean() * 100
        print(f"Price spike frequency: {spike_pct:.2f}% of observations")

        return df

    def build_master_feature_set(self,
                                 weather_dir: Optional[Path] = None) -> pd.DataFrame:
        """
        Build complete feature set for all models.

        Returns master DataFrame with all features ready for ML models.
        """
        print("="*80)
        print("BUILDING MASTER FEATURE SET")
        print("="*80)

        # Load datasets
        datasets = self.load_datasets()

        # Calculate all feature groups
        print("\n1. Calculating forecast error features...")
        forecast_errors = self.calculate_forecast_errors(datasets)

        print("\n2. Calculating ORDC features...")
        ordc_features = self.calculate_ordc_features(datasets)

        print("\n3. Calculating weather features...")
        if weather_dir:
            weather_features = self.calculate_weather_features(weather_dir)
        else:
            weather_features = pd.DataFrame()

        print("\n4. Calculating net load features...")
        net_load_features = self.calculate_net_load_features(datasets)

        print("\n5. Merging all features...")
        # Start with forecast errors as base
        master_df = forecast_errors if not forecast_errors.empty else pd.DataFrame()

        # Merge other feature groups
        if not ordc_features.empty:
            master_df = pd.merge(master_df, ordc_features, on='timestamp', how='outer')

        if not weather_features.empty:
            master_df = pd.merge(master_df, weather_features, on='timestamp', how='outer')

        if not net_load_features.empty:
            master_df = pd.merge(master_df, net_load_features, on='timestamp', how='outer')

        print("\n6. Adding temporal features...")
        if not master_df.empty:
            master_df = self.calculate_temporal_features(master_df, 'timestamp')

        print("\n7. Adding price targets and spike labels...")
        if not datasets['dam_prices'].empty:
            master_df = pd.merge(master_df, datasets['dam_prices'], on='timestamp', how='left')
            master_df = self.create_price_spike_labels(master_df)

        print(f"\n✅ Master feature set created: {len(master_df):,} rows, {len(master_df.columns)} features")

        return master_df


if __name__ == "__main__":
    # Test the feature engineering pipeline
    import os
    from dotenv import load_dotenv
    load_dotenv()
    data_dir = Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data")
    weather_dir = Path(os.getenv('WEATHER_DATA_DIR', '/pool/ssd8tb/data/weather_data'))

    fe = ERCOTFeatureEngineer(data_dir)
    master_features = fe.build_master_feature_set(weather_dir=weather_dir)

    # Save to parquet for fast loading
    if not master_features.empty:
        output_file = data_dir / "master_features.parquet"
        master_features.to_parquet(output_file, index=False)
        print(f"\n💾 Saved master features to: {output_file}")

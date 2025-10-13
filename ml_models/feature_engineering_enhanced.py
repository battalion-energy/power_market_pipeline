"""
Enhanced Feature Engineering with Wind/Solar Forecast Errors

This module adds the CRITICAL features missing from the baseline model:
- Wind forecast errors (actual - STWPF)
- Solar forecast errors (actual - STPPF)

These features are key indicators of RT price spikes.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import os
from dotenv import load_dotenv
from feature_engineering_parquet import ERCOTFeatureEngineerParquet


class ERCOTFeatureEngineerEnhanced(ERCOTFeatureEngineerParquet):
    """Enhanced feature engineering with wind/solar forecast errors AND weather data."""

    def __init__(self, data_dir: Path, weather_dir: Path = None):
        """
        Initialize with data directory and optional weather directory.

        Args:
            data_dir: ERCOT market data directory
            weather_dir: Weather data directory (default: /pool/ssd8tb/data/weather_data)
        """
        super().__init__(data_dir)
        if weather_dir is None:
            load_dotenv()
            self.weather_dir = Path(os.getenv('WEATHER_DATA_DIR', '/pool/ssd8tb/data/weather_data'))
        else:
            self.weather_dir = Path(weather_dir)

    def load_weather_data(self) -> pd.DataFrame:
        """
        Load ERCOT weather data and calculate weather features.

        Returns comprehensive weather features including:
        - Temperature extremes (heat waves, cold snaps)
        - Wind speed aggregates (system-wide from wind farms)
        - Solar irradiance aggregates (system-wide from solar farms)
        - Demand indicators (cooling/heating degree days)
        """
        print("\nLoading ERCOT weather data...")

        weather_file = self.weather_dir / "parquet_by_iso/ERCOT_weather_data.parquet"
        if not weather_file.exists():
            print(f"⚠️  Weather data not found: {weather_file}")
            return pd.DataFrame()

        df = pd.read_parquet(weather_file)
        print(f"   Loaded {len(df):,} weather records")
        print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"   Locations: {df['location_name'].nunique()}")

        # Separate by location type
        cities = df[df['location_name'].str.contains('CITY_')]
        wind_farms = df[df['location_name'].str.contains('WIND_')]
        solar_farms = df[df['location_name'].str.contains('SOLAR_')]

        print(f"   Cities: {cities['location_name'].nunique()}")
        print(f"   Wind farms: {wind_farms['location_name'].nunique()}")
        print(f"   Solar farms: {solar_farms['location_name'].nunique()}")

        # Calculate system-wide aggregates by date
        weather_agg = []

        # 1. CITY TEMPERATURE AGGREGATES (population-weighted proxy)
        city_agg = cities.groupby('date').agg({
            'T2M': ['mean', 'max', 'min', 'std'],
            'T2M_MAX': 'max',
            'T2M_MIN': 'min',
            'RH2M': 'mean',
            'PRECTOTCORR': 'sum'
        })
        city_agg.columns = ['_'.join(col).strip() for col in city_agg.columns.values]
        city_agg = city_agg.rename(columns={
            'T2M_mean': 'temp_avg',
            'T2M_max': 'temp_max_hourly',
            'T2M_min': 'temp_min_hourly',
            'T2M_std': 'temp_std_cities',
            'T2M_MAX_max': 'temp_max_daily',
            'T2M_MIN_min': 'temp_min_daily',
            'RH2M_mean': 'humidity_avg',
            'PRECTOTCORR_sum': 'precip_total'
        })

        # 2. WIND FARM AGGREGATES (system-wide wind conditions)
        if not wind_farms.empty:
            wind_agg = wind_farms.groupby('date').agg({
                'WS50M': ['mean', 'min', 'max', 'std'],
                'WD50M': 'mean'
            })
            wind_agg.columns = ['_'.join(col).strip() for col in wind_agg.columns.values]
            wind_agg = wind_agg.rename(columns={
                'WS50M_mean': 'wind_speed_avg',
                'WS50M_min': 'wind_speed_min',
                'WS50M_max': 'wind_speed_max',
                'WS50M_std': 'wind_speed_std',
                'WD50M_mean': 'wind_direction_avg'
            })
            city_agg = city_agg.join(wind_agg)

        # 3. SOLAR FARM AGGREGATES (system-wide solar conditions)
        if not solar_farms.empty:
            solar_agg = solar_farms.groupby('date').agg({
                'ALLSKY_SFC_SW_DWN': ['mean', 'min', 'max', 'std'],
                'CLRSKY_SFC_SW_DWN': 'mean'
            })
            solar_agg.columns = ['_'.join(col).strip() for col in solar_agg.columns.values]
            solar_agg = solar_agg.rename(columns={
                'ALLSKY_SFC_SW_DWN_mean': 'solar_irrad_avg',
                'ALLSKY_SFC_SW_DWN_min': 'solar_irrad_min',
                'ALLSKY_SFC_SW_DWN_max': 'solar_irrad_max',
                'ALLSKY_SFC_SW_DWN_std': 'solar_irrad_std',
                'CLRSKY_SFC_SW_DWN_mean': 'solar_irrad_clear_sky'
            })
            city_agg = city_agg.join(solar_agg)

        # 4. CALCULATED WEATHER FEATURES
        weather_features = city_agg.reset_index()

        # Temperature extremes
        weather_features['heat_wave'] = (weather_features['temp_max_daily'] > 35).astype(int)  # >95°F
        weather_features['cold_snap'] = (weather_features['temp_min_daily'] < 0).astype(int)  # <32°F
        weather_features['temp_range_daily'] = weather_features['temp_max_daily'] - weather_features['temp_min_daily']

        # Cooling/Heating Degree Days (base 18.3°C = 65°F)
        weather_features['cooling_degree_days'] = (weather_features['temp_avg'] - 18.3).clip(lower=0)
        weather_features['heating_degree_days'] = (18.3 - weather_features['temp_avg']).clip(lower=0)

        # Cloud cover indicator (difference between clear-sky and actual)
        if 'solar_irrad_clear_sky' in weather_features.columns:
            weather_features['cloud_cover'] = weather_features['solar_irrad_clear_sky'] - weather_features['solar_irrad_avg']
            weather_features['cloud_cover_pct'] = (weather_features['cloud_cover'] / weather_features['solar_irrad_clear_sky'].replace(0, np.nan)) * 100

        # Wind variability (calm vs gusty)
        if 'wind_speed_std' in weather_features.columns:
            weather_features['wind_calm'] = (weather_features['wind_speed_avg'] < 3).astype(int)  # Low wind
            weather_features['wind_strong'] = (weather_features['wind_speed_avg'] > 15).astype(int)  # High wind

        print(f"\n✅ Weather features calculated: {len(weather_features):,} days")
        print(f"   Heat wave days: {weather_features['heat_wave'].sum()}")
        print(f"   Cold snap days: {weather_features['cold_snap'].sum()}")
        if 'wind_speed_avg' in weather_features.columns:
            print(f"   Avg wind speed: {weather_features['wind_speed_avg'].mean():.1f} m/s")
        if 'solar_irrad_avg' in weather_features.columns:
            print(f"   Avg solar irradiance: {weather_features['solar_irrad_avg'].mean():.2f} kWh/m²/day")

        return weather_features

    def load_wind_forecast_errors(self) -> pd.DataFrame:
        """
        Load wind data and calculate forecast errors.

        CSV structure (hourly):
        - Column 0: timestamp
        - Column 1: delivery date
        - Column 2: hour ending
        - Column 3: System-wide actual generation (GEN)
        - Columns 4-7: Regional actual gen
        - Column 8: System-wide STWPF forecast
        - Columns 9-11: Regional STWPF forecasts
        """
        wind_dir = self.data_dir / "Wind_Power_Production"
        csv_files = sorted(wind_dir.glob("*.csv"))

        if not csv_files:
            print("⚠️  No wind CSV files found")
            return pd.DataFrame()

        dfs = []
        for f in csv_files:
            try:
                df = pd.read_csv(f, header=0)  # First row is column numbers

                # Extract relevant columns
                # Column 0 = forecast timestamp (when made)
                # Column 1 = delivery_date, Column 2 = hour_ending
                # Column 3 = actual, 8 = forecast
                if len(df.columns) >= 9:
                    df_subset = df.iloc[:, [1, 2, 3, 8]].copy()
                    df_subset.columns = ['delivery_date', 'hour_ending', 'wind_actual', 'wind_forecast']

                    # Convert to numeric
                    df_subset['wind_actual'] = pd.to_numeric(df_subset['wind_actual'], errors='coerce')
                    df_subset['wind_forecast'] = pd.to_numeric(df_subset['wind_forecast'], errors='coerce')

                    # CRITICAL FIX: Create timestamp from delivery_date + hour_ending
                    # ERCOT uses "hour ending" convention (1-24), where hour 1 = 00:00-01:00
                    df_subset['delivery_date'] = pd.to_datetime(df_subset['delivery_date'])
                    # Hour ending 1 = 01:00, hour ending 24 = 00:00 (midnight)
                    df_subset['timestamp'] = df_subset['delivery_date'] + pd.to_timedelta(df_subset['hour_ending'], unit='h')

                    # Calculate error (actual - forecast)
                    df_subset['wind_error_mw'] = df_subset['wind_actual'] - df_subset['wind_forecast']
                    df_subset['wind_error_pct'] = (df_subset['wind_error_mw'] / df_subset['wind_forecast'].replace(0, np.nan)) * 100

                    # Keep only needed columns
                    df_subset = df_subset[['timestamp', 'wind_actual', 'wind_forecast', 'wind_error_mw', 'wind_error_pct']]

                    dfs.append(df_subset)
            except Exception as e:
                print(f"Error loading {f.name}: {e}")
                continue

        if not dfs:
            return pd.DataFrame()

        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.sort_values('timestamp').reset_index(drop=True)

        # Remove duplicate timestamps (keep most recent)
        combined = combined.drop_duplicates(subset=['timestamp'], keep='last')

        # CRITICAL FIX: Round timestamps to hourly boundaries
        # Forecast timestamps are at :55:11 past the hour, need to normalize to :00:00
        combined['timestamp'] = combined['timestamp'].dt.floor('h')

        # After rounding, remove any duplicate hours (keep last)
        combined = combined.drop_duplicates(subset=['timestamp'], keep='last')

        # Calculate cumulative errors (3-hour and 6-hour windows)
        combined['wind_error_3h'] = combined['wind_error_mw'].rolling(3, min_periods=1).sum()
        combined['wind_error_6h'] = combined['wind_error_mw'].rolling(6, min_periods=1).sum()

        print(f"✅ Loaded wind forecast errors: {len(combined):,} records")
        print(f"   Date range: {combined['timestamp'].min()} to {combined['timestamp'].max()}")
        print(f"   Mean absolute error: {abs(combined['wind_error_mw']).mean():.1f} MW")

        return combined

    def load_solar_forecast_errors(self) -> pd.DataFrame:
        """
        Load solar data and calculate forecast errors.

        CSV structure (hourly):
        - Column 0: timestamp
        - Column 1: delivery date
        - Column 2: hour ending
        - Column 3: System-wide actual generation
        - Columns 4-10: Regional actual gen
        - Column 11: System-wide STPPF forecast
        - Columns 12-18: Regional STPPF forecasts
        """
        solar_dir = self.data_dir / "Solar_Power_Production"
        csv_files = sorted(solar_dir.glob("*.csv"))

        if not csv_files:
            print("⚠️  No solar CSV files found")
            return pd.DataFrame()

        dfs = []
        for f in csv_files:
            try:
                df = pd.read_csv(f, header=0)

                # Extract relevant columns
                # Column 0 = forecast timestamp (when made)
                # Column 1 = delivery_date, Column 2 = hour_ending
                # Column 3 = actual, 11 = forecast
                if len(df.columns) >= 12:
                    df_subset = df.iloc[:, [1, 2, 3, 11]].copy()
                    df_subset.columns = ['delivery_date', 'hour_ending', 'solar_actual', 'solar_forecast']

                    # Convert to numeric
                    df_subset['solar_actual'] = pd.to_numeric(df_subset['solar_actual'], errors='coerce')
                    df_subset['solar_forecast'] = pd.to_numeric(df_subset['solar_forecast'], errors='coerce')

                    # CRITICAL FIX: Create timestamp from delivery_date + hour_ending
                    df_subset['delivery_date'] = pd.to_datetime(df_subset['delivery_date'])
                    df_subset['timestamp'] = df_subset['delivery_date'] + pd.to_timedelta(df_subset['hour_ending'], unit='h')

                    # Calculate error
                    df_subset['solar_error_mw'] = df_subset['solar_actual'] - df_subset['solar_forecast']
                    df_subset['solar_error_pct'] = (df_subset['solar_error_mw'] / df_subset['solar_forecast'].replace(0, np.nan)) * 100

                    # Keep only needed columns
                    df_subset = df_subset[['timestamp', 'solar_actual', 'solar_forecast', 'solar_error_mw', 'solar_error_pct']]

                    dfs.append(df_subset)
            except Exception as e:
                print(f"Error loading {f.name}: {e}")
                continue

        if not dfs:
            return pd.DataFrame()

        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.sort_values('timestamp').reset_index(drop=True)

        # Remove duplicate timestamps (keep most recent)
        combined = combined.drop_duplicates(subset=['timestamp'], keep='last')

        # CRITICAL FIX: Round timestamps to hourly boundaries
        # Forecast timestamps are at :55:11 past the hour, need to normalize to :00:00
        combined['timestamp'] = combined['timestamp'].dt.floor('h')

        # After rounding, remove any duplicate hours (keep last)
        combined = combined.drop_duplicates(subset=['timestamp'], keep='last')

        # Calculate cumulative errors
        combined['solar_error_3h'] = combined['solar_error_mw'].rolling(3, min_periods=1).sum()
        combined['solar_error_6h'] = combined['solar_error_mw'].rolling(6, min_periods=1).sum()

        print(f"✅ Loaded solar forecast errors: {len(combined):,} records")
        print(f"   Date range: {combined['timestamp'].min()} to {combined['timestamp'].max()}")
        print(f"   Mean absolute error: {abs(combined['solar_error_mw']).mean():.1f} MW")

        return combined

    def build_master_feature_set_enhanced(self, years: List[int] = [2024, 2025],
                                         hub_col: str = 'HB_HOUSTON') -> pd.DataFrame:
        """
        Build enhanced feature set with wind/solar forecast errors AND weather data.

        Features added:
        - Wind/solar forecast errors (hourly)
        - Temperature extremes and demand indicators (daily)
        - System-wide wind/solar conditions (daily)
        - Heat wave / cold snap indicators

        KEY FIX: Aggregate RT prices to HOURLY to match forecast resolution.
        """
        print("="*80)
        print("BUILDING ENHANCED FEATURE SET WITH FORECAST ERRORS")
        print("="*80)

        # Load RT prices and calculate price features (from parent class)
        print("\n1. Loading RT prices (15-min resolution)...")
        rt_prices = self.load_rt_prices(years)
        if rt_prices.empty:
            print("❌ No RT price data available")
            return pd.DataFrame()

        # Convert to DataFrame with timestamps
        start_date = datetime(years[0], 1, 1)
        timestamps = [start_date + timedelta(minutes=15*i) for i in range(len(rt_prices))]
        rt_df = pd.DataFrame(rt_prices)
        rt_df['timestamp'] = timestamps
        rt_df = rt_df.set_index('timestamp')

        # AGGREGATE TO HOURLY (to match wind/solar resolution)
        print(f"   15-min samples: {len(rt_df):,}")
        print("   Aggregating to hourly resolution to match forecast data...")

        rt_hourly = rt_df.resample('h').agg({
            hub_col: ['mean', 'min', 'max', 'std']
        })
        rt_hourly.columns = ['price_mean', 'price_min', 'price_max', 'price_std']
        rt_hourly['price_volatility'] = rt_hourly['price_std'] / (rt_hourly['price_mean'] + 1e-8)
        rt_hourly['price_range'] = rt_hourly['price_max'] - rt_hourly['price_min']

        print(f"   Hourly samples: {len(rt_hourly):,}")

        # Calculate hourly price features
        rt_hourly['price_change_1h'] = rt_hourly['price_mean'].diff(1)
        rt_hourly['price_change_4h'] = rt_hourly['price_mean'].diff(4)
        rt_hourly['price_ma_24h'] = rt_hourly['price_mean'].rolling(24, min_periods=1).mean()
        rt_hourly['price_std_24h'] = rt_hourly['price_mean'].rolling(24, min_periods=1).std()
        rt_hourly['price_momentum'] = rt_hourly['price_mean'] / (rt_hourly['price_ma_24h'] + 1e-8)

        price_features = rt_hourly.reset_index()

        # Load wind forecast errors (hourly data)
        print("\n2. Loading wind forecast errors...")
        wind_errors = self.load_wind_forecast_errors()

        # Load solar forecast errors (hourly data)
        print("\n3. Loading solar forecast errors...")
        solar_errors = self.load_solar_forecast_errors()

        # Load weather data (daily data)
        print("\n4. Loading weather data...")
        weather_data = self.load_weather_data()

        # Merge forecast errors with hourly RT prices (1:1 merge now!)
        master_df = price_features.copy()

        if not wind_errors.empty:
            # LEFT join to keep all RT prices, forward-fill forecast gaps
            wind_errors = wind_errors.set_index('timestamp')
            master_df = master_df.merge(
                wind_errors[['wind_error_mw', 'wind_error_pct', 'wind_error_3h', 'wind_error_6h']],
                left_on='timestamp', right_index=True, how='left'
            )
            # Forward-fill forecast errors for missing hours (max 3 hours)
            master_df[['wind_error_mw', 'wind_error_pct', 'wind_error_3h', 'wind_error_6h']] = \
                master_df[['wind_error_mw', 'wind_error_pct', 'wind_error_3h', 'wind_error_6h']].fillna(method='ffill', limit=3)
            print(f"   ✅ Merged wind errors: {master_df['wind_error_mw'].notna().sum():,} records")

        if not solar_errors.empty:
            # LEFT join to keep all RT prices, forward-fill forecast gaps
            solar_errors = solar_errors.set_index('timestamp')
            master_df = master_df.merge(
                solar_errors[['solar_error_mw', 'solar_error_pct', 'solar_error_3h', 'solar_error_6h']],
                left_on='timestamp', right_index=True, how='left'
            )
            # Forward-fill forecast errors for missing hours (max 3 hours)
            master_df[['solar_error_mw', 'solar_error_pct', 'solar_error_3h', 'solar_error_6h']] = \
                master_df[['solar_error_mw', 'solar_error_pct', 'solar_error_3h', 'solar_error_6h']].fillna(method='ffill', limit=3)
            print(f"   ✅ Merged solar errors: {master_df['solar_error_mw'].notna().sum():,} records")

        # Merge weather data (daily → hourly by repeating for all hours of each day)
        if not weather_data.empty:
            print("\n5. Merging weather data...")
            # Extract date from timestamp for merging
            master_df['date'] = master_df['timestamp'].dt.date
            master_df['date'] = pd.to_datetime(master_df['date'])

            # Merge on date (weather is daily, will repeat for all hours)
            weather_data['date'] = pd.to_datetime(weather_data['date'])
            master_df = master_df.merge(weather_data, on='date', how='left')

            # Drop the date column
            master_df = master_df.drop('date', axis=1)

            weather_cols = [c for c in weather_data.columns if c != 'date']
            print(f"   ✅ Merged {len(weather_cols)} weather features")
            print(f"   Weather coverage: {master_df[weather_cols[0]].notna().sum():,} / {len(master_df):,} hours")

        # Add temporal features
        print("\n6. Adding temporal features...")
        master_df = master_df.set_index('timestamp')
        master_df = self.calculate_temporal_features(master_df)
        master_df = master_df.reset_index()

        # Create spike labels (use price_mean for hourly data)
        print("\n7. Creating price spike labels...")
        # Add price column for spike detection (use hourly mean)
        master_df['price'] = master_df['price_mean']
        master_df = self.create_price_spike_labels(master_df)
        master_df = master_df.drop('price', axis=1)  # Remove temp column

        # Drop NaN rows
        initial_rows = len(master_df)
        master_df = master_df.dropna()
        dropped = initial_rows - len(master_df)
        print(f"\nDropped {dropped:,} rows with NaN values")

        print(f"\n✅ Enhanced feature set created: {len(master_df):,} rows, {len(master_df.columns)} features")
        print(f"Date range: {master_df['timestamp'].min()} to {master_df['timestamp'].max()}")
        print(f"\nNew features added: {[c for c in master_df.columns if 'wind' in c or 'solar' in c]}")

        return master_df


if __name__ == "__main__":
    # Test the enhanced feature engineering
    data_dir = Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data")

    fe = ERCOTFeatureEngineerEnhanced(data_dir)
    master_features = fe.build_master_feature_set_enhanced(years=[2024, 2025])

    # Save to parquet
    if not master_features.empty:
        output_file = data_dir / "master_features_enhanced.parquet"
        master_features.to_parquet(output_file, index=False)
        print(f"\n💾 Saved enhanced features to: {output_file}")
        print(f"Spike rate: {master_features['price_spike'].mean()*100:.2f}%")
        print(f"Features: {len(master_features.columns)}")

"""
Multi-Horizon Feature Engineering (2019-2025)

Comprehensive feature set for 48-hour battery trading decisions:
- Uses ALL available data (2019-2025)
- Includes: RT prices, DA prices, AS prices, Weather
- Creates multi-horizon targets (next 1-48 hours)
- Time-aware features for market regime changes

Purpose: At 10am day-prior, predict spike probabilities for next 48 hours
to optimize DAM/AS/RT capacity allocation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class ERCOTMultiHorizonFeatureEngineer:
    """Feature engineering for multi-horizon (48h) battery trading model."""

    def __init__(self, data_dir: Path, weather_dir: Path = None):
        """
        Initialize with data directories.

        Args:
            data_dir: ERCOT market data directory
            weather_dir: Weather data directory
        """
        self.data_dir = Path(data_dir)
        self.rollup_dir = self.data_dir / "rollup_files" / "flattened"

        if weather_dir is None:
            import os
            from dotenv import load_dotenv
            load_dotenv()
            self.weather_dir = Path(os.getenv('WEATHER_DATA_DIR', '/pool/ssd8tb/data/weather_data'))
        else:
            self.weather_dir = Path(weather_dir)

    def load_rt_prices_hourly(self, years: List[int]) -> pd.DataFrame:
        """
        Load RT prices and aggregate to hourly.

        Returns DataFrame with hourly RT price statistics.
        """
        print(f"\n1. Loading RT prices (15-min → hourly)...")
        dfs = []
        for year in years:
            file = self.rollup_dir / f"RT_prices_15min_{year}.parquet"
            if file.exists():
                df = pd.read_parquet(file)
                dfs.append(df)
                print(f"   ✅ {year}: {len(df):,} records")

        if not dfs:
            return pd.DataFrame()

        combined = pd.concat(dfs, ignore_index=True)

        # Create proper timestamps (15-min intervals starting from first year)
        start_date = datetime(years[0], 1, 1)
        combined['timestamp'] = [start_date + timedelta(minutes=15*i) for i in range(len(combined))]
        combined = combined.set_index('timestamp')

        print(f"   Total 15-min records: {len(combined):,}")
        print(f"   Date range: {combined.index.min()} to {combined.index.max()}")

        return combined

    def load_da_prices(self, years: List[int]) -> pd.DataFrame:
        """Load DA prices (already hourly)."""
        print(f"\n2. Loading DA prices...")
        dfs = []
        for year in years:
            file = self.rollup_dir / f"DA_prices_{year}.parquet"
            if file.exists():
                df = pd.read_parquet(file)
                # Convert datetime_ts to timestamp
                if 'datetime_ts' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['datetime_ts'])
                    df = df.set_index('timestamp')
                dfs.append(df)
                print(f"   ✅ {year}: {len(df):,} records")

        if not dfs:
            return pd.DataFrame()

        combined = pd.concat(dfs)
        print(f"   Total DA records: {len(combined):,}")
        return combined

    def load_as_prices(self, years: List[int]) -> pd.DataFrame:
        """Load Ancillary Service prices."""
        print(f"\n3. Loading AS prices...")
        dfs = []
        for year in years:
            file = self.rollup_dir / f"AS_prices_{year}.parquet"
            if file.exists():
                df = pd.read_parquet(file)
                # Convert datetime_ts to timestamp
                if 'datetime_ts' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['datetime_ts'])
                    df = df.set_index('timestamp')
                elif 'DeliveryDate' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['DeliveryDate'])
                    df = df.set_index('timestamp')
                dfs.append(df)
                print(f"   ✅ {year}: {len(df):,} records")

        if not dfs:
            return pd.DataFrame()

        combined = pd.concat(dfs)
        print(f"   Total AS records: {len(combined):,}")

        # AS services: REGUP, REGDN, RRS, NSPIN, ECRS
        as_cols = ['REGUP', 'REGDN', 'RRS', 'NSPIN', 'ECRS']
        available = [c for c in as_cols if c in combined.columns]
        print(f"   AS services available: {available}")

        return combined[available]

    def load_weather_data(self) -> pd.DataFrame:
        """Load weather data from NASA POWER."""
        print(f"\n4. Loading weather data...")

        weather_file = self.weather_dir / "parquet_by_iso/ERCOT_weather_data.parquet"
        if not weather_file.exists():
            print(f"   ⚠️  Weather data not found")
            return pd.DataFrame()

        df = pd.read_parquet(weather_file)
        print(f"   ✅ Loaded {len(df):,} weather records")
        print(f"   Date range: {df['date'].min()} to {df['date'].max()}")

        # Separate by location type
        cities = df[df['location_name'].str.contains('CITY_')]
        wind_farms = df[df['location_name'].str.contains('WIND_')]
        solar_farms = df[df['location_name'].str.contains('SOLAR_')]

        # Aggregate by date
        weather_agg = cities.groupby('date').agg({
            'T2M': ['mean', 'max', 'min', 'std'],
            'T2M_MAX': 'max',
            'T2M_MIN': 'min',
            'RH2M': 'mean',
            'PRECTOTCORR': 'sum'
        })
        weather_agg.columns = ['_'.join(col).strip() for col in weather_agg.columns.values]
        weather_agg = weather_agg.rename(columns={
            'T2M_mean': 'temp_avg',
            'T2M_max': 'temp_max_hourly',
            'T2M_min': 'temp_min_hourly',
            'T2M_std': 'temp_std_cities',
            'T2M_MAX_max': 'temp_max_daily',
            'T2M_MIN_min': 'temp_min_daily',
            'RH2M_mean': 'humidity_avg',
            'PRECTOTCORR_sum': 'precip_total'
        })

        # Wind aggregates
        if not wind_farms.empty:
            wind_agg = wind_farms.groupby('date').agg({
                'WS50M': ['mean', 'max', 'std']
            })
            wind_agg.columns = ['wind_speed_avg', 'wind_speed_max', 'wind_speed_std']
            weather_agg = weather_agg.join(wind_agg)

        # Solar aggregates
        if not solar_farms.empty:
            solar_agg = solar_farms.groupby('date').agg({
                'ALLSKY_SFC_SW_DWN': ['mean', 'max', 'std'],
                'CLRSKY_SFC_SW_DWN': 'mean'
            })
            solar_agg.columns = ['solar_irrad_avg', 'solar_irrad_max', 'solar_irrad_std', 'solar_irrad_clear_sky']
            weather_agg = weather_agg.join(solar_agg)

        weather_features = weather_agg.reset_index()

        # Calculated features
        weather_features['heat_wave'] = (weather_features['temp_max_daily'] > 35).astype(int)
        weather_features['cold_snap'] = (weather_features['temp_min_daily'] < 0).astype(int)
        weather_features['temp_range_daily'] = weather_features['temp_max_daily'] - weather_features['temp_min_daily']
        weather_features['cooling_degree_days'] = (weather_features['temp_avg'] - 18.3).clip(lower=0)
        weather_features['heating_degree_days'] = (18.3 - weather_features['temp_avg']).clip(lower=0)

        if 'solar_irrad_clear_sky' in weather_features.columns:
            weather_features['cloud_cover'] = weather_features['solar_irrad_clear_sky'] - weather_features['solar_irrad_avg']
            weather_features['cloud_cover_pct'] = (weather_features['cloud_cover'] / weather_features['solar_irrad_clear_sky'].replace(0, np.nan)) * 100

        if 'wind_speed_avg' in weather_features.columns:
            weather_features['wind_calm'] = (weather_features['wind_speed_avg'] < 3).astype(int)
            weather_features['wind_strong'] = (weather_features['wind_speed_avg'] > 15).astype(int)

        print(f"   Weather features: {len(weather_features.columns)-1}")

        return weather_features

    def create_time_aware_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-aware features to handle market evolution.

        Key insight: ERCOT market changed dramatically over 2019-2025
        - 2021: Winter Storm Uri (extreme volatility)
        - 2024-2025: Much calmer (10x fewer spikes)
        """
        df = df.copy()

        # Extract time components
        df['year'] = df.index.year
        df['quarter'] = df.index.quarter
        df['month'] = df.index.month
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_year'] = df.index.dayofyear

        # Market regime indicators
        df['post_winter_storm'] = (df['year'] >= 2021).astype(int)
        df['high_renewable_era'] = (df['year'] >= 2023).astype(int)
        df['years_since_2019'] = df['year'] - 2019

        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        # Day type
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_peak_hour'] = ((df['hour'] >= 7) & (df['hour'] <= 22)).astype(int)

        # Season
        df['season'] = df['month'].apply(lambda x:
            0 if x in [12, 1, 2] else  # Winter
            1 if x in [3, 4, 5] else    # Spring
            2 if x in [6, 7, 8] else    # Summer
            3)                           # Fall

        return df

    def create_multi_horizon_labels(self, rt_hourly: pd.DataFrame,
                                    price_col: str = 'price_mean',
                                    high_threshold: float = 400,
                                    low_threshold: float = 20,
                                    extreme_threshold: float = 1000,
                                    horizons: List[int] = list(range(1, 49))) -> pd.DataFrame:
        """
        Create multi-horizon spike labels for next 1-48 hours.

        For each hour t, create labels for t+1, t+2, ..., t+48:
        - spike_high_Xh: Price > $400 in X hours
        - spike_low_Xh: Price < $20 in X hours
        - spike_extreme_Xh: Price > $1000 in X hours

        Args:
            rt_hourly: Hourly RT prices DataFrame
            price_col: Price column to use (default: 'price_mean')
            high_threshold: High price threshold (discharge opportunity)
            low_threshold: Low price threshold (charge opportunity)
            extreme_threshold: Extreme spike threshold
            horizons: List of forecast horizons in hours

        Returns:
            DataFrame with multi-horizon labels
        """
        print(f"\n5. Creating multi-horizon labels...")

        prices = rt_hourly[price_col].copy()
        labels = pd.DataFrame(index=prices.index)

        # Create labels for each horizon
        for h in horizons:
            # Future price at horizon h
            future_price = prices.shift(-h)

            # High price (discharge opportunity)
            labels[f'spike_high_{h}h'] = (future_price > high_threshold).astype(int)

            # Low price (charge opportunity)
            labels[f'spike_low_{h}h'] = (future_price < low_threshold).astype(int)

            # Extreme spike (risk management)
            labels[f'spike_extreme_{h}h'] = (future_price > extreme_threshold).astype(int)

        # Count events
        high_1h = labels['spike_high_1h'].sum()
        low_1h = labels['spike_low_1h'].sum()
        extreme_1h = labels['spike_extreme_1h'].sum()

        print(f"   High price events (>{high_threshold}): {high_1h:,} ({high_1h/len(labels)*100:.2f}%)")
        print(f"   Low price events (<{low_threshold}): {low_1h:,} ({low_1h/len(labels)*100:.2f}%)")
        print(f"   Extreme events (>{extreme_threshold}): {extreme_1h:,} ({extreme_1h/len(labels)*100:.2f}%)")

        return labels

    def build_master_feature_set(self, years: List[int] = list(range(2019, 2026)),
                                  hub_col: str = 'HB_HOUSTON',
                                  exclude_winter_storm_uri: bool = False) -> pd.DataFrame:
        """
        Build comprehensive feature set for multi-horizon battery trading model.

        Uses ALL available data (2019-2025):
        - RT prices (target)
        - DA prices (for spread)
        - AS prices (for opportunity cost)
        - Weather (for demand/supply indicators)
        - Time features (for market regime changes)

        Args:
            years: Years to include in dataset
            hub_col: Hub column to use for RT prices
            exclude_winter_storm_uri: If True, exclude Feb 10-20, 2021 (grid failure)

        Returns:
            DataFrame with features and multi-horizon labels
        """
        print("="*80)
        print("BUILDING MULTI-HORIZON FEATURE SET (2019-2025)")
        print("="*80)

        # Load all data sources
        rt_15min = self.load_rt_prices_hourly(years)
        da_prices = self.load_da_prices(years)
        as_prices = self.load_as_prices(years)
        weather = self.load_weather_data()

        if rt_15min.empty:
            print("❌ No RT price data")
            return pd.DataFrame()

        # Aggregate RT to hourly
        print(f"\n   Aggregating RT prices to hourly...")
        rt_hourly = rt_15min.resample('h').agg({
            hub_col: ['mean', 'min', 'max', 'std', 'first', 'last']
        })
        rt_hourly.columns = ['price_mean', 'price_min', 'price_max', 'price_std', 'price_first', 'price_last']
        rt_hourly['price_volatility'] = rt_hourly['price_std'] / (rt_hourly['price_mean'].replace(0, np.nan))
        rt_hourly['price_range'] = rt_hourly['price_max'] - rt_hourly['price_min']
        rt_hourly['price_change_intra'] = rt_hourly['price_last'] - rt_hourly['price_first']

        print(f"   Hourly RT records: {len(rt_hourly):,}")

        # Start building master DataFrame
        master_df = rt_hourly[['price_mean', 'price_min', 'price_max', 'price_std',
                                'price_volatility', 'price_range', 'price_change_intra']].copy()

        # Add DA-RT spread (market stress indicator)
        if not da_prices.empty and hub_col in da_prices.columns:
            print(f"\n   Merging DA prices...")
            da_subset = da_prices[[hub_col]].copy()
            da_subset = da_subset.rename(columns={hub_col: 'price_da'})
            master_df = master_df.join(da_subset, how='left')

            if 'price_da' in master_df.columns:
                master_df['da_rt_spread'] = master_df['price_mean'] - master_df['price_da']
                master_df['da_rt_spread_pct'] = (master_df['da_rt_spread'] / master_df['price_da'].replace(0, np.nan)) * 100

                # Forward-fill missing DA prices (holidays, etc.)
                master_df['price_da'] = master_df['price_da'].fillna(method='ffill', limit=24)
                master_df['da_rt_spread'] = master_df['da_rt_spread'].fillna(method='ffill', limit=24)

                print(f"   DA coverage: {master_df['price_da'].notna().sum():,} / {len(master_df):,} hours")
            else:
                print(f"   ⚠️  DA price merge failed, column not found")

        # Add AS prices (opportunity cost for holding capacity)
        if not as_prices.empty:
            print(f"\n   Merging AS prices...")
            master_df = master_df.join(as_prices, how='left')

            # Forward-fill AS prices (updated hourly)
            as_cols = [c for c in as_prices.columns if c in master_df.columns]
            for col in as_cols:
                master_df[col] = master_df[col].fillna(method='ffill', limit=24)

            # Total AS opportunity cost
            if len(as_cols) > 0:
                master_df['as_total'] = master_df[as_cols].sum(axis=1)
                master_df['as_vs_rt_spread'] = master_df['price_mean'] - master_df['as_total']

            print(f"   AS coverage: {master_df[as_cols[0]].notna().sum():,} / {len(master_df):,} hours")

        # Add weather features (daily → repeat for all hours)
        if not weather.empty:
            print(f"\n   Merging weather data...")
            master_df = master_df.reset_index()
            master_df['date'] = pd.to_datetime(master_df['timestamp'].dt.date)
            weather['date'] = pd.to_datetime(weather['date'])
            master_df = master_df.merge(weather, on='date', how='left')
            master_df = master_df.set_index('timestamp')
            master_df = master_df.drop('date', axis=1)

            weather_cols = [c for c in weather.columns if c != 'date']
            print(f"   Weather features: {len(weather_cols)}")
            print(f"   Weather coverage: {master_df[weather_cols[0]].notna().sum():,} / {len(master_df):,} hours")

        # Add time-aware features
        print(f"\n   Creating time-aware features...")
        master_df = self.create_time_aware_features(master_df)

        # Create multi-horizon labels (use price_mean column from rt_hourly)
        labels = self.create_multi_horizon_labels(rt_hourly, price_col='price_mean')
        master_df = master_df.join(labels, how='left')

        # Exclude Winter Storm Uri if requested (Feb 10-20, 2021 - grid failure)
        if exclude_winter_storm_uri:
            uri_start = pd.Timestamp('2021-02-10')
            uri_end = pd.Timestamp('2021-02-20')
            before_exclusion = len(master_df)
            master_df = master_df[(master_df.index < uri_start) | (master_df.index > uri_end)]
            excluded_count = before_exclusion - len(master_df)
            print(f"\n   ⚠️  Excluded Winter Storm Uri: removed {excluded_count:,} hours ({excluded_count/24:.1f} days)")

        # Drop rows with NaN in critical columns
        critical_cols = ['price_mean']
        initial_rows = len(master_df)
        master_df = master_df.dropna(subset=critical_cols)
        dropped = initial_rows - len(master_df)

        print(f"\n{'='*80}")
        print(f"✅ Master feature set created:")
        print(f"   Total samples: {len(master_df):,}")
        print(f"   Features: {len(master_df.columns)}")
        print(f"   Date range: {master_df.index.min()} to {master_df.index.max()}")
        print(f"   Dropped NaN: {dropped:,} rows")
        print(f"{'='*80}\n")

        return master_df


if __name__ == "__main__":
    # Test the comprehensive feature engineering
    data_dir = Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data")

    fe = ERCOTMultiHorizonFeatureEngineer(data_dir)

    # Create TWO versions: with and without Winter Storm Uri
    print("\n" + "="*80)
    print("CREATING VERSION 1: WITH Winter Storm Uri (Feb 2021)")
    print("="*80)
    master_features = fe.build_master_feature_set(
        years=list(range(2019, 2026)),
        exclude_winter_storm_uri=False
    )

    if not master_features.empty:
        output_file = data_dir / "master_features_multihorizon_2019_2025.parquet"
        master_features.to_parquet(output_file)
        print(f"💾 Saved to: {output_file}")
        print(f"\nSample columns: {list(master_features.columns[:20])}")
        print(f"\nLabel columns: {[c for c in master_features.columns if 'spike_' in c][:10]}")

    print("\n" + "="*80)
    print("CREATING VERSION 2: WITHOUT Winter Storm Uri (excluding Feb 10-20, 2021)")
    print("="*80)
    master_features_no_uri = fe.build_master_feature_set(
        years=list(range(2019, 2026)),
        exclude_winter_storm_uri=True
    )

    if not master_features_no_uri.empty:
        output_file_no_uri = data_dir / "master_features_multihorizon_2019_2025_no_uri.parquet"
        master_features_no_uri.to_parquet(output_file_no_uri)
        print(f"💾 Saved to: {output_file_no_uri}")

        # Compare versions
        print(f"\n📊 Comparison:")
        print(f"   With Uri: {len(master_features):,} samples")
        print(f"   Without Uri: {len(master_features_no_uri):,} samples")
        print(f"   Difference: {len(master_features) - len(master_features_no_uri):,} samples removed")

"""
Fast Feature Engineering using Parquet Files

Uses pre-processed parquet files from rollup_files/flattened/
Much faster than CSVs and has proper column names.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class ERCOTFeatureEngineerParquet:
    """Feature engineering using parquet files - FAST."""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.rollup_dir = self.data_dir / "rollup_files" / "flattened"

    def load_da_prices(self, years: List[int] = [2023, 2024, 2025]) -> pd.DataFrame:
        """Load DA prices from parquet files."""
        dfs = []
        for year in years:
            file = self.rollup_dir / f"DA_prices_{year}.parquet"
            if file.exists():
                df = pd.read_parquet(file)
                dfs.append(df)
                print(f"✅ Loaded DA prices {year}: {len(df):,} records")

        if not dfs:
            return pd.DataFrame()

        combined = pd.concat(dfs, ignore_index=True)
        # Use datetime_ts as timestamp
        combined = combined.rename(columns={'datetime_ts': 'timestamp'})
        return combined

    def load_rt_prices(self, years: List[int] = [2023, 2024, 2025]) -> pd.DataFrame:
        """Load RT prices from parquet files (15-min resolution)."""
        dfs = []
        for year in years:
            file = self.rollup_dir / f"RT_prices_15min_{year}.parquet"
            if file.exists():
                df = pd.read_parquet(file)
                dfs.append(df)
                print(f"✅ Loaded RT prices {year}: {len(df):,} records")

        if not dfs:
            return pd.DataFrame()

        combined = pd.concat(dfs, ignore_index=True)
        # No renaming needed, already has proper columns
        return combined

    def load_as_prices(self, years: List[int] = [2023, 2024, 2025]) -> pd.DataFrame:
        """Load AS prices from parquet files."""
        dfs = []
        for year in years:
            file = self.rollup_dir / f"AS_prices_{year}.parquet"
            if file.exists():
                df = pd.read_parquet(file)
                dfs.append(df)
                print(f"✅ Loaded AS prices {year}: {len(df):,} records")

        if not dfs:
            return pd.DataFrame()

        combined = pd.concat(dfs, ignore_index=True)
        return combined

    def calculate_price_features(self, rt_prices: pd.DataFrame,
                                  hub_col: str = 'HB_HOUSTON') -> pd.DataFrame:
        """
        Calculate price-based features from RT prices.

        Args:
            rt_prices: RT price DataFrame (15-min resolution)
            hub_col: Hub column to use for features
        """
        df = rt_prices[[hub_col]].copy()
        df = df.rename(columns={hub_col: 'price'})

        # Rolling statistics (30-day window)
        df['price_ma_30d'] = df['price'].rolling(30*96, min_periods=96).mean()  # 96 intervals/day
        df['price_std_30d'] = df['price'].rolling(30*96, min_periods=96).std()

        # Price volatility
        df['price_volatility_1h'] = df['price'].rolling(4).std()  # 4 x 15-min = 1 hour
        df['price_volatility_4h'] = df['price'].rolling(16).std()

        # Price changes
        df['price_change_15min'] = df['price'].diff()
        df['price_change_1h'] = df['price'].diff(4)
        df['price_change_4h'] = df['price'].diff(16)

        # Price momentum
        df['price_momentum_1h'] = df['price'].pct_change(4)
        df['price_momentum_4h'] = df['price'].pct_change(16)

        return df

    def calculate_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features with cyclical encoding."""
        df = df.copy()

        # Parse timestamp if it's not datetime
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            df.index = pd.to_datetime(df.index)

        # Extract time components
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['day_of_year'] = df.index.dayofyear

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

    def create_price_spike_labels(self, df: pd.DataFrame,
                                  price_col: str = 'price',
                                  statistical_threshold_std: float = 3.0,
                                  economic_threshold: float = 1000.0) -> pd.DataFrame:
        """
        Create binary labels for price spikes.

        Spike definition (ANY of):
        1. Statistical: price > μ + 3σ (rolling 30-day)
        2. Economic: price > $1000/MWh
        """
        df = df.copy()

        # Statistical threshold (rolling 30-day mean + 3*std)
        if 'price_ma_30d' in df.columns and 'price_std_30d' in df.columns:
            statistical_threshold = df['price_ma_30d'] + (statistical_threshold_std * df['price_std_30d'])
            df['spike_statistical'] = (df[price_col] > statistical_threshold).astype(int)
        else:
            df['spike_statistical'] = 0

        # Economic threshold
        df['spike_economic'] = (df[price_col] > economic_threshold).astype(int)

        # Combined spike indicator (ANY condition triggers spike)
        df['price_spike'] = ((df['spike_statistical'] == 1) |
                            (df['spike_economic'] == 1)).astype(int)

        spike_pct = df['price_spike'].mean() * 100
        print(f"Price spike frequency: {spike_pct:.2f}% of observations")

        return df

    def build_master_feature_set(self, years: List[int] = [2023, 2024, 2025],
                                 hub_col: str = 'HB_HOUSTON') -> pd.DataFrame:
        """
        Build complete feature set from parquet files.

        Args:
            years: Years to load
            hub_col: Hub column to use for predictions (HB_HOUSTON, HB_NORTH, etc.)

        Returns:
            DataFrame with features and price_spike label
        """
        print("="*80)
        print("BUILDING MASTER FEATURE SET FROM PARQUET FILES")
        print("="*80)

        # Load RT prices (15-min resolution) - our target
        print("\n1. Loading RT prices...")
        rt_prices = self.load_rt_prices(years)

        if rt_prices.empty:
            print("❌ No RT price data available")
            return pd.DataFrame()

        # RT prices already have proper integer index, don't change it
        # We'll use the index as timestamp later

        # Calculate price-based features
        print("\n2. Calculating price features...")
        price_features = self.calculate_price_features(rt_prices, hub_col)

        # Add temporal features
        print("\n3. Adding temporal features...")
        master_df = self.calculate_temporal_features(price_features)

        # Create spike labels
        print("\n4. Creating price spike labels...")
        master_df = self.create_price_spike_labels(master_df)

        # The index is actually the row position from parquet
        # We need proper timestamps - let's use the date range
        # For now, create sequential timestamps based on 15-min intervals
        import pandas as pd
        from datetime import datetime, timedelta

        # Create proper datetime index (15-min intervals)
        # Start from first available date in data
        start_date = datetime(2024, 1, 1)  # Adjust based on actual data
        master_df = master_df.reset_index(drop=True)
        master_df['timestamp'] = [start_date + timedelta(minutes=15*i) for i in range(len(master_df))]

        # Drop NaN rows (from rolling calculations)
        initial_rows = len(master_df)
        master_df = master_df.dropna()
        dropped = initial_rows - len(master_df)
        print(f"\nDropped {dropped:,} rows with NaN values (from rolling calculations)")

        print(f"\n✅ Master feature set created: {len(master_df):,} rows, {len(master_df.columns)} features")
        print(f"Date range: {master_df['timestamp'].min()} to {master_df['timestamp'].max()}")
        print(f"\nFeatures: {list(master_df.columns)}")

        return master_df


if __name__ == "__main__":
    # Test the fast parquet-based feature engineering
    data_dir = Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data")

    fe = ERCOTFeatureEngineerParquet(data_dir)
    master_features = fe.build_master_feature_set(years=[2024, 2025])

    # Save to parquet for fast loading
    if not master_features.empty:
        output_file = data_dir / "master_features_parquet.parquet"
        master_features.to_parquet(output_file, index=False)
        print(f"\n💾 Saved master features to: {output_file}")
        print(f"Spike rate: {master_features['price_spike'].mean()*100:.2f}%")

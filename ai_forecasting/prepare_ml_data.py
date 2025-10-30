#!/usr/bin/env python3
"""
Prepare ML Training Data - Processes New ERCOT Datasets
Runs overnight while data transfers from other computer

This script:
1. Monitors for new CSV files in transfer directories
2. Converts to Parquet (fast, columnar format)
3. Merges with existing data
4. Creates master feature dataset for ML training
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging
from glob import glob
import pyarrow.parquet as pq
import pyarrow as pa
from concurrent.futures import ProcessPoolExecutor, as_completed
import re

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'ml_data_prep_{datetime.now().strftime("%Y%m%d_%H%M")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data")
ERCOT_DATA = BASE_DIR / "ERCOT_data"
ROLLUP_DIR = ERCOT_DATA / "rollup_files" / "flattened"
ML_OUTPUT = ERCOT_DATA

# Critical datasets for ML (prioritize these)
PRIORITY_DATASETS = {
    # Name in directory -> Output name
    "Real-Time_ORDC_and_Reliability_Deployment_Price_Adders_and_Reserves_by_SCED_Interval": "ordc_reserves",
    "Seven-Day_Load_Forecast_by_Model_and_Weather_Zone": "load_forecast_7d",
    "Seven-Day_Load_Forecast_by_Forecast_Zone": "load_forecast_7d_fz",
    "Solar_Power_Production_-_Actual_5-Minute_Averaged_Values": "solar_5min",
    "Solar_Power_Production_-_Hourly_Averaged_Actual_and_Forecasted_Values": "solar_hourly",
    "System-Wide_Demand": "demand",
}


def extract_year_from_filename(filename):
    """Extract year from ERCOT filename patterns."""
    patterns = [
        r'\.(\d{4})\d{4}\.',  # .YYYYMMDD.
        r'_(\d{4})\d{4}_',    # _YYYYMMDD_
        r'_(\d{4})\d{4}\.',   # _YYYYMMDD.
        r'(\d{4})-\d{2}-\d{2}',  # YYYY-MM-DD
    ]

    for pattern in patterns:
        matches = re.findall(pattern, str(filename))
        for match in matches:
            year = int(match)
            if 2010 <= year <= 2030:
                return year

    return None


def find_csv_files(dataset_dir):
    """Find all CSV files in dataset directory."""
    csv_files = []

    # Check main directory
    for csv in Path(dataset_dir).glob("*.csv"):
        csv_files.append(csv)

    # Check csv subdirectory (if exists)
    csv_subdir = Path(dataset_dir) / "csv"
    if csv_subdir.exists():
        for csv in csv_subdir.glob("*.csv"):
            csv_files.append(csv)

    logger.info(f"Found {len(csv_files)} CSV files in {dataset_dir}")
    return sorted(csv_files)


def process_ordc_reserves(csv_files, output_dir):
    """
    Process ORDC and Reserves data - CRITICAL for spike prediction

    Extracts:
    - Online reserves (MW)
    - Reserve margin
    - ORDC price adder
    - Scarcity pricing triggers
    """
    logger.info("Processing ORDC and Reserves data...")

    all_data = []

    for csv_file in csv_files:
        try:
            logger.info(f"Reading {csv_file.name}")
            df = pd.read_csv(csv_file)

            # Parse timestamp
            if 'SCEDTimestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['SCEDTimestamp'])
            elif 'Interval' in df.columns:
                df['timestamp'] = pd.to_datetime(df['Interval'])

            # Extract key columns
            columns_to_keep = [
                'timestamp',
                'OnlineReserves',  # MW
                'OfflineReserves',  # MW
                'TotalReserves',  # MW
                'ORDCPriceAdder',  # $/MWh
                'ReliabilityDeploymentPriceAdder',  # $/MWh
            ]

            # Keep only columns that exist
            keep_cols = [c for c in columns_to_keep if c in df.columns]
            df = df[keep_cols]

            all_data.append(df)

        except Exception as e:
            logger.error(f"Error processing {csv_file}: {e}")

    if not all_data:
        logger.warning("No ORDC data processed")
        return None

    # Combine all years
    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.sort_values('timestamp').drop_duplicates()

    # Save by year
    for year in combined['timestamp'].dt.year.unique():
        year_data = combined[combined['timestamp'].dt.year == year]
        output_file = output_dir / f"ordc_reserves_{year}.parquet"
        year_data.to_parquet(output_file, index=False, engine='pyarrow')
        logger.info(f"Saved {len(year_data)} records to {output_file}")

    return combined


def process_load_forecasts(csv_files, output_dir):
    """
    Process Load Forecasts - CRITICAL for all models

    Extracts 7-day ahead load forecasts by zone/area
    """
    logger.info("Processing Load Forecasts...")

    all_data = []

    for csv_file in csv_files:
        try:
            logger.info(f"Reading {csv_file.name}")
            df = pd.read_csv(csv_file)

            # Parse timestamps
            if 'DeliveryDate' in df.columns and 'HourEnding' in df.columns:
                df['timestamp'] = pd.to_datetime(df['DeliveryDate']) + pd.to_timedelta(df['HourEnding'].astype(int) - 1, unit='h')
            elif 'DeliveryDateTime' in df.columns:
                df['timestamp'] = pd.to_datetime(df['DeliveryDateTime'])

            # Extract forecast columns
            forecast_cols = [c for c in df.columns if 'Forecast' in c or 'forecast' in c]

            # Keep timestamp + forecast columns
            keep_cols = ['timestamp'] + forecast_cols
            if 'WeatherZone' in df.columns:
                keep_cols.append('WeatherZone')
            if 'ForecastZone' in df.columns:
                keep_cols.append('ForecastZone')

            keep_cols = [c for c in keep_cols if c in df.columns]
            df = df[keep_cols]

            all_data.append(df)

        except Exception as e:
            logger.error(f"Error processing {csv_file}: {e}")

    if not all_data:
        logger.warning("No load forecast data processed")
        return None

    # Combine all years
    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.sort_values('timestamp').drop_duplicates()

    # Save by year
    for year in combined['timestamp'].dt.year.unique():
        year_data = combined[combined['timestamp'].dt.year == year]
        output_file = output_dir / f"load_forecast_{year}.parquet"
        year_data.to_parquet(output_file, index=False, engine='pyarrow')
        logger.info(f"Saved {len(year_data)} records to {output_file}")

    return combined


def process_solar_production(csv_files, output_dir):
    """Process Solar Power Production - Actual and Forecasted."""
    logger.info("Processing Solar Production...")

    all_data = []

    for csv_file in csv_files:
        try:
            logger.info(f"Reading {csv_file.name}")
            df = pd.read_csv(csv_file)

            # Parse timestamp
            if 'Interval' in df.columns:
                df['timestamp'] = pd.to_datetime(df['Interval'])
            elif 'DeliveryDateTime' in df.columns:
                df['timestamp'] = pd.to_datetime(df['DeliveryDateTime'])

            all_data.append(df)

        except Exception as e:
            logger.error(f"Error processing {csv_file}: {e}")

    if not all_data:
        return None

    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.sort_values('timestamp').drop_duplicates()

    # Save by year
    for year in combined['timestamp'].dt.year.unique():
        year_data = combined[combined['timestamp'].dt.year == year]
        output_file = output_dir / f"solar_production_{year}.parquet"
        year_data.to_parquet(output_file, index=False, engine='pyarrow')
        logger.info(f"Saved {len(year_data)} records to {output_file}")

    return combined


def create_master_ml_dataset(years=range(2019, 2026)):
    """
    Create master ML training dataset by merging all features.

    Combines:
    - RT prices (15-min)
    - DA prices (hourly)
    - AS prices (hourly)
    - ORDC reserves (5-min)
    - Load forecasts (hourly)
    - Solar/Wind production (5-min/hourly)
    - Weather data
    """
    logger.info("Creating master ML dataset...")

    master_data = []

    for year in years:
        logger.info(f"Processing year {year}...")

        # Load RT prices (15-min resolution)
        rt_file = ROLLUP_DIR / f"RT_prices_15min_{year}.parquet"
        if not rt_file.exists():
            logger.warning(f"Missing RT prices for {year}")
            continue

        df_rt = pd.read_parquet(rt_file)
        logger.info(f"Loaded {len(df_rt)} RT price records for {year}")

        # Aggregate to hourly (for easier merging)
        df_rt['hour'] = df_rt['datetime'].dt.floor('H')
        df_hourly = df_rt.groupby('hour').agg({
            'HB_HOUSTON': ['mean', 'min', 'max', 'std'],
            'HB_NORTH': ['mean', 'min', 'max', 'std'],
            'HB_SOUTH': ['mean', 'min', 'max', 'std'],
            'HB_WEST': ['mean', 'min', 'max', 'std'],
        }).reset_index()

        # Flatten column names
        df_hourly.columns = ['_'.join(col).strip('_') for col in df_hourly.columns.values]
        df_hourly.rename(columns={'hour_': 'datetime'}, inplace=True)

        # Load DA prices
        da_file = ROLLUP_DIR / f"DA_prices_{year}.parquet"
        if da_file.exists():
            df_da = pd.read_parquet(da_file)
            df_hourly = df_hourly.merge(df_da, on='datetime', how='left', suffixes=('_rt', '_da'))
            logger.info(f"Merged DA prices for {year}")

        # Load AS prices
        as_file = ROLLUP_DIR / f"AS_prices_{year}.parquet"
        if as_file.exists():
            df_as = pd.read_parquet(as_file)
            df_hourly = df_hourly.merge(df_as, on='datetime', how='left')
            logger.info(f"Merged AS prices for {year}")

        # Load ORDC reserves (if processed)
        ordc_file = ERCOT_DATA / f"ordc_reserves_{year}.parquet"
        if ordc_file.exists():
            df_ordc = pd.read_parquet(ordc_file)
            df_ordc['hour'] = df_ordc['timestamp'].dt.floor('H')
            df_ordc_hourly = df_ordc.groupby('hour').agg({
                'OnlineReserves': 'mean',
                'ORDCPriceAdder': 'mean',
            }).reset_index()
            df_ordc_hourly.rename(columns={'hour': 'datetime'}, inplace=True)
            df_hourly = df_hourly.merge(df_ordc_hourly, on='datetime', how='left')
            logger.info(f"Merged ORDC reserves for {year}")

        # Load Load forecasts (if processed)
        lf_file = ERCOT_DATA / f"load_forecast_{year}.parquet"
        if lf_file.exists():
            df_lf = pd.read_parquet(lf_file)
            df_lf['hour'] = df_lf['timestamp'].dt.floor('H')
            # Aggregate forecasts
            lf_cols = [c for c in df_lf.columns if 'Forecast' in c or 'forecast' in c]
            if lf_cols:
                df_lf_hourly = df_lf.groupby('hour')[lf_cols].mean().reset_index()
                df_lf_hourly.rename(columns={'hour': 'datetime'}, inplace=True)
                df_hourly = df_hourly.merge(df_lf_hourly, on='datetime', how='left')
                logger.info(f"Merged load forecasts for {year}")

        # Add temporal features
        df_hourly['hour_of_day'] = df_hourly['datetime'].dt.hour
        df_hourly['day_of_week'] = df_hourly['datetime'].dt.dayofweek
        df_hourly['month'] = df_hourly['datetime'].dt.month
        df_hourly['day_of_year'] = df_hourly['datetime'].dt.dayofyear
        df_hourly['is_weekend'] = (df_hourly['day_of_week'] >= 5).astype(int)

        # Cyclical encoding
        df_hourly['hour_sin'] = np.sin(2 * np.pi * df_hourly['hour_of_day'] / 24)
        df_hourly['hour_cos'] = np.cos(2 * np.pi * df_hourly['hour_of_day'] / 24)
        df_hourly['day_sin'] = np.sin(2 * np.pi * df_hourly['day_of_week'] / 7)
        df_hourly['day_cos'] = np.cos(2 * np.pi * df_hourly['day_of_week'] / 7)
        df_hourly['month_sin'] = np.sin(2 * np.pi * df_hourly['month'] / 12)
        df_hourly['month_cos'] = np.cos(2 * np.pi * df_hourly['month'] / 12)

        # Create spike labels (for Model 3)
        # Spike = RT price > $400/MWh
        df_hourly['spike_400'] = (df_hourly['HB_HOUSTON_mean'] > 400).astype(int)
        df_hourly['spike_1000'] = (df_hourly['HB_HOUSTON_mean'] > 1000).astype(int)

        master_data.append(df_hourly)
        logger.info(f"Year {year}: {len(df_hourly)} hours, {df_hourly['spike_400'].sum()} spikes")

    if not master_data:
        logger.error("No data to create master dataset")
        return None

    # Combine all years
    master_df = pd.concat(master_data, ignore_index=True)
    master_df = master_df.sort_values('datetime').reset_index(drop=True)

    # Save
    output_file = ERCOT_DATA / f"master_ml_dataset_{years[0]}_{years[-1]}.parquet"
    master_df.to_parquet(output_file, index=False, engine='pyarrow')
    logger.info(f"\n{'='*80}")
    logger.info(f"MASTER DATASET CREATED: {output_file}")
    logger.info(f"Total records: {len(master_df):,}")
    logger.info(f"Date range: {master_df['datetime'].min()} to {master_df['datetime'].max()}")
    logger.info(f"Total spikes >$400: {master_df['spike_400'].sum():,} ({master_df['spike_400'].mean()*100:.2f}%)")
    logger.info(f"Total spikes >$1000: {master_df['spike_1000'].sum():,} ({master_df['spike_1000'].mean()*100:.2f}%)")
    logger.info(f"Features: {len(master_df.columns)}")
    logger.info(f"{'='*80}\n")

    return master_df


def process_dataset(dataset_name, dataset_dir):
    """Process a single dataset."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing: {dataset_name}")
    logger.info(f"{'='*60}")

    csv_files = find_csv_files(dataset_dir)

    if not csv_files:
        logger.warning(f"No CSV files found in {dataset_dir}")
        return None

    output_dir = ERCOT_DATA

    if "ORDC" in dataset_name:
        return process_ordc_reserves(csv_files, output_dir)
    elif "Load_Forecast" in dataset_name:
        return process_load_forecasts(csv_files, output_dir)
    elif "Solar" in dataset_name:
        return process_solar_production(csv_files, output_dir)
    else:
        logger.info(f"No specific processor for {dataset_name}, skipping")
        return None


def main():
    """Main processing pipeline."""
    logger.info("="*80)
    logger.info("ML DATA PREPARATION - Starting")
    logger.info("="*80)
    logger.info(f"Base directory: {BASE_DIR}")
    logger.info(f"Output directory: {ERCOT_DATA}")
    logger.info(f"Time: {datetime.now()}")

    # Process priority datasets
    for dataset_name, short_name in PRIORITY_DATASETS.items():
        dataset_dir = BASE_DIR / dataset_name

        if not dataset_dir.exists():
            logger.warning(f"Dataset directory not found: {dataset_dir}")
            logger.info(f"Waiting for data transfer...")
            continue

        try:
            process_dataset(dataset_name, dataset_dir)
        except Exception as e:
            logger.error(f"Error processing {dataset_name}: {e}", exc_info=True)

    # Create master ML dataset
    try:
        logger.info("\n" + "="*80)
        logger.info("CREATING MASTER ML DATASET")
        logger.info("="*80)
        master_df = create_master_ml_dataset(years=range(2019, 2026))

        if master_df is not None:
            logger.info("\n✅ SUCCESS! Master ML dataset ready for training")
        else:
            logger.error("\n❌ Failed to create master dataset")

    except Exception as e:
        logger.error(f"Error creating master dataset: {e}", exc_info=True)

    logger.info("\n" + "="*80)
    logger.info("ML DATA PREPARATION - Complete")
    logger.info("="*80)


if __name__ == "__main__":
    main()

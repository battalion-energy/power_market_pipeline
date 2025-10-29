#!/usr/bin/env python3
"""
ERCOT Critical Data Processing Script
This script processes critical ERCOT data sources for ML price forecasting:
1. Creates CSV subdirectories in each critical data folder
2. Recursively extracts all zip files
3. Moves extracted CSV files to CSV subdirectories
4. Converts CSV files to yearly parquet files (handling schema changes)
5. Moves all parquet files to forecast_parquet_full_files directory
"""

import os
import zipfile
import shutil
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from collections import defaultdict
import re
from datetime import datetime
import logging
from multiprocessing import Pool, cpu_count
from functools import partial

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_processing.log'),
        logging.StreamHandler()
    ]
)

# Critical data sources for ML forecasting
CRITICAL_DATA_SOURCES = [
    # Tier 1 - Essential Price & Market Data
    "Settlement_Point_Prices_at_Resource_Nodes,_Hubs_and_Load_Zones",
    "DAM_Settlement_Point_Prices",
    "LMPs_by_Resource_Nodes,_Load_Zones_and_Trading_Hubs",
    "Real-Time_ORDC_and_Reliability_Deployment_Price_Adders_and_Reserves_by_SCED_Interval",
    "SCED_System_Lambda",
    "SCED_Shadow_Prices_and_Binding_Transmission_Constraints",

    # Tier 2 - Supply Data
    "Wind_Power_Production_-_Actual_5-Minute_Averaged_Values",
    "Solar_Power_Production_-_Actual_5-Minute_Averaged_Values",
    "Hourly_Resource_Outage_Capacity",
    "State_Estimator_Load_Report_-_Total_ERCOT_Generation",

    # Tier 3 - Demand Data
    "Actual_System_Load_by_Forecast_Zone",
    "Actual_System_Load_by_Weather_Zone",
    "System-Wide_Demand",
    "Intra-Hour_Load_Forecast_by_Weather_Zone",

    # Tier 4 - Market Offer/Bid Data
    "2-Day_DAM_and_SCED_Energy_Curves_Reports",
    "2-Day_Real_Time_Gen_and_Load_Data_Reports",
]

BASE_DIR = Path("/Users/enrico/data/ERCOT_data")
PARQUET_OUTPUT_DIR = BASE_DIR / "forecast_parquet_full_files"
NUM_CORES = 12  # Use all 12 CPU cores


def extract_year_from_filename(filename):
    """Extract year from various ERCOT filename formats."""
    # Try different date patterns
    patterns = [
        r'_(\d{4})\d{4}_',  # _YYYYMMDD_
        r'\.(\d{4})\d{4}\.',  # .YYYYMMDD.
        r'_(\d{4})_',  # _YYYY_
        r'(\d{4})',  # Any 4-digit year
    ]

    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            year = int(match.group(1))
            if 2010 <= year <= 2030:  # Reasonable year range
                return year

    logging.warning(f"Could not extract year from filename: {filename}")
    return None


def create_csv_directories(data_sources):
    """Create CSV subdirectories in each critical data folder."""
    logging.info("Creating CSV subdirectories...")

    for source in data_sources:
        source_path = BASE_DIR / source
        if not source_path.exists():
            logging.warning(f"Data source not found: {source}")
            continue

        csv_dir = source_path / "CSV"
        csv_dir.mkdir(exist_ok=True)
        logging.info(f"Created/verified CSV directory: {csv_dir}")


def extract_single_zip(zip_file_path):
    """Extract a single zip file (for parallel processing)."""
    try:
        zip_file = Path(zip_file_path)
        source_path = zip_file.parent
        extract_dir = source_path / zip_file.stem

        # Skip if already extracted
        if extract_dir.exists() and any(extract_dir.iterdir()):
            return f"Already extracted: {zip_file.name}"

        extract_dir.mkdir(exist_ok=True)

        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        return f"Extracted: {zip_file.name}"

    except Exception as e:
        return f"Error extracting {zip_file_path}: {e}"


def extract_zip_files(data_sources):
    """Recursively extract all zip files in critical folders using parallel processing."""
    logging.info("Extracting zip files in parallel...")

    all_zip_files = []
    for source in data_sources:
        source_path = BASE_DIR / source
        if not source_path.exists():
            continue

        zip_files = list(source_path.glob("*.zip"))
        all_zip_files.extend(zip_files)
        logging.info(f"Found {len(zip_files)} zip files in {source}")

    if all_zip_files:
        logging.info(f"Extracting {len(all_zip_files)} zip files using {NUM_CORES} cores...")
        with Pool(NUM_CORES) as pool:
            results = pool.map(extract_single_zip, all_zip_files)

        # Log results
        for result in results:
            if "Error" in result:
                logging.error(result)
            elif "Extracted" in result:
                logging.info(result)


def move_csv_files(data_sources):
    """Move all extracted CSV files to CSV subdirectories."""
    logging.info("Moving CSV files to CSV subdirectories...")

    for source in data_sources:
        source_path = BASE_DIR / source
        if not source_path.exists():
            continue

        csv_dir = source_path / "CSV"
        csv_count = 0

        # Find all CSV files in subdirectories (not in CSV folder)
        for csv_file in source_path.rglob("*.csv"):
            # Skip if already in CSV directory
            if "CSV" in csv_file.parts:
                continue

            try:
                dest_file = csv_dir / csv_file.name

                # Handle filename conflicts
                counter = 1
                while dest_file.exists():
                    stem = csv_file.stem
                    dest_file = csv_dir / f"{stem}_{counter}.csv"
                    counter += 1

                shutil.move(str(csv_file), str(dest_file))
                csv_count += 1

            except Exception as e:
                logging.error(f"Error moving {csv_file}: {e}")

        logging.info(f"Moved {csv_count} CSV files for {source}")


def infer_unified_schema(csv_files):
    """
    Infer a unified schema from multiple CSV files that accounts for all columns.
    Returns a dictionary mapping column names to their data types.
    """
    all_columns = {}

    # Sample up to 10 files to understand the schema variations
    sample_files = csv_files[:10] if len(csv_files) > 10 else csv_files

    for csv_file in sample_files:
        try:
            # Read just the first row to get column names and types
            df_sample = pd.read_csv(csv_file, nrows=100)

            for col in df_sample.columns:
                if col not in all_columns:
                    all_columns[col] = df_sample[col].dtype
                else:
                    # If we've seen this column before, use the more general type
                    existing_type = all_columns[col]
                    new_type = df_sample[col].dtype

                    # Promote to object (string) if types conflict
                    if existing_type != new_type:
                        all_columns[col] = 'object'

        except Exception as e:
            logging.warning(f"Error reading {csv_file} for schema inference: {e}")

    return all_columns


def process_year_to_parquet(args):
    """Process a single year of data to parquet (for parallel processing)."""
    source, year, year_files, unified_schema, csv_dir = args

    try:
        logging.info(f"Processing year {year} with {len(year_files)} files for {source}")

        dfs = []
        for csv_file in year_files:
            try:
                # Read CSV with all columns as string first to avoid type issues
                df = pd.read_csv(csv_file, dtype=str, low_memory=False)

                # Add any missing columns with None values
                for col in unified_schema:
                    if col not in df.columns:
                        df[col] = None

                # Reorder columns to match unified schema
                df = df[list(unified_schema.keys())]

                dfs.append(df)

            except Exception as e:
                return f"Error reading CSV in {source} year {year}: {e}"

        if not dfs:
            return f"No dataframes loaded for {source} year {year}"

        # Combine all dataframes for the year
        combined_df = pd.concat(dfs, ignore_index=True)

        # Convert types where possible (but keep as string if conversion fails)
        for col, dtype in unified_schema.items():
            if dtype != 'object':
                try:
                    combined_df[col] = pd.to_numeric(combined_df[col], errors='ignore')
                except:
                    pass

        # Create parquet file
        parquet_filename = f"{source}_{year}.parquet"
        parquet_path = csv_dir / parquet_filename

        combined_df.to_parquet(parquet_path, engine='pyarrow', compression='snappy')
        return f"Created: {parquet_filename} ({len(combined_df):,} rows)"

    except Exception as e:
        return f"Error processing year {year} for {source}: {e}"


def csv_to_yearly_parquet(data_sources):
    """Convert CSV files to yearly parquet files, handling schema changes."""
    logging.info("Converting CSV files to yearly parquet files in parallel...")

    all_tasks = []

    for source in data_sources:
        source_path = BASE_DIR / source
        csv_dir = source_path / "CSV"

        if not csv_dir.exists():
            logging.warning(f"CSV directory not found: {csv_dir}")
            continue

        # Find all CSV files
        csv_files = list(csv_dir.glob("*.csv"))
        if not csv_files:
            logging.warning(f"No CSV files found in {csv_dir}")
            continue

        logging.info(f"Processing {len(csv_files)} CSV files for {source}")

        # Group files by year
        files_by_year = defaultdict(list)
        for csv_file in csv_files:
            year = extract_year_from_filename(csv_file.name)
            if year:
                files_by_year[year].append(csv_file)

        # Infer unified schema for this data source
        unified_schema = infer_unified_schema(csv_files)
        logging.info(f"Unified schema has {len(unified_schema)} columns for {source}")

        # Create tasks for parallel processing
        for year, year_files in sorted(files_by_year.items()):
            all_tasks.append((source, year, year_files, unified_schema, csv_dir))

    # Process all years in parallel
    if all_tasks:
        logging.info(f"Processing {len(all_tasks)} year-datasets using {NUM_CORES} cores...")
        with Pool(NUM_CORES) as pool:
            results = pool.map(process_year_to_parquet, all_tasks)

        # Log results
        for result in results:
            if "Error" in result:
                logging.error(result)
            else:
                logging.info(result)


def move_parquet_files(data_sources):
    """Move all parquet files to the forecast_parquet_full_files directory."""
    logging.info("Moving parquet files to forecast_parquet_full_files...")

    # Create output directory
    PARQUET_OUTPUT_DIR.mkdir(exist_ok=True)

    parquet_count = 0

    for source in data_sources:
        source_path = BASE_DIR / source
        csv_dir = source_path / "CSV"

        if not csv_dir.exists():
            continue

        # Find all parquet files
        parquet_files = list(csv_dir.glob("*.parquet"))

        for parquet_file in parquet_files:
            try:
                dest_file = PARQUET_OUTPUT_DIR / parquet_file.name

                # Handle filename conflicts
                counter = 1
                while dest_file.exists():
                    stem = parquet_file.stem
                    dest_file = PARQUET_OUTPUT_DIR / f"{stem}_{counter}.parquet"
                    counter += 1

                shutil.copy2(str(parquet_file), str(dest_file))
                parquet_count += 1
                logging.info(f"Copied: {parquet_file.name}")

            except Exception as e:
                logging.error(f"Error copying {parquet_file}: {e}")

    logging.info(f"Copied {parquet_count} parquet files to {PARQUET_OUTPUT_DIR}")


def main():
    """Main execution function."""
    logging.info("=" * 80)
    logging.info("Starting ERCOT Critical Data Processing")
    logging.info("=" * 80)

    # Verify critical data sources exist
    existing_sources = []
    for source in CRITICAL_DATA_SOURCES:
        source_path = BASE_DIR / source
        if source_path.exists():
            existing_sources.append(source)
        else:
            logging.warning(f"Data source not found: {source}")

    logging.info(f"Found {len(existing_sources)}/{len(CRITICAL_DATA_SOURCES)} critical data sources")

    # Step 1: Create CSV directories
    create_csv_directories(existing_sources)

    # Step 2: Extract zip files
    extract_zip_files(existing_sources)

    # Step 3: Move CSV files to CSV subdirectories
    move_csv_files(existing_sources)

    # Step 4: Convert CSV to yearly parquet files
    csv_to_yearly_parquet(existing_sources)

    # Step 5: Move parquet files to central location
    move_parquet_files(existing_sources)

    logging.info("=" * 80)
    logging.info("ERCOT Critical Data Processing Complete!")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()

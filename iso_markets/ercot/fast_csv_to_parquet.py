#!/usr/bin/env python3
"""
Fast CSV to Parquet Converter - Works with already-extracted CSVs
This script skips the extraction phase and directly processes existing CSV files
into yearly parquet files, then moves them to forecast_parquet_full_files.
"""

import os
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from collections import defaultdict
import re
import logging
from multiprocessing import Pool
from glob import glob

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fast_parquet_conversion.log'),
        logging.StreamHandler()
    ]
)

# Critical data sources for ML forecasting
CRITICAL_DATA_SOURCES = [
    "Settlement_Point_Prices_at_Resource_Nodes,_Hubs_and_Load_Zones",
    "DAM_Settlement_Point_Prices",
    "LMPs_by_Resource_Nodes,_Load_Zones_and_Trading_Hubs",
    "Real-Time_ORDC_and_Reliability_Deployment_Price_Adders_and_Reserves_by_SCED_Interval",
    "SCED_System_Lambda",
    "SCED_Shadow_Prices_and_Binding_Transmission_Constraints",
    "Wind_Power_Production_-_Actual_5-Minute_Averaged_Values",
    "Solar_Power_Production_-_Actual_5-Minute_Averaged_Values",
    "Hourly_Resource_Outage_Capacity",
    "State_Estimator_Load_Report_-_Total_ERCOT_Generation",
    "Actual_System_Load_by_Forecast_Zone",
    "Actual_System_Load_by_Weather_Zone",
    "System-Wide_Demand",
    "Intra-Hour_Load_Forecast_by_Weather_Zone",
    "2-Day_DAM_and_SCED_Energy_Curves_Reports",
    "2-Day_Real_Time_Gen_and_Load_Data_Reports",
]

BASE_DIR = Path("/Users/enrico/data/ERCOT_data")
PARQUET_OUTPUT_DIR = BASE_DIR / "forecast_parquet_full_files"
NUM_CORES = 12


def extract_year_from_filename(filename):
    """Extract year from various ERCOT filename formats."""
    patterns = [
        r'\.(\d{4})\d{4}\.',  # .YYYYMMDD.
        r'_(\d{4})\d{4}_',    # _YYYYMMDD_
        r'_(\d{4})\d{4}\.',   # _YYYYMMDD.
        r'\.(\d{4})\d{4}_',   # .YYYYMMDD_
        r'/(\d{4})\d{4}/',    # /YYYYMMDD/
        r'(\d{4})-\d{2}-\d{2}',  # YYYY-MM-DD
        r'_(\d{4})_',         # _YYYY_
    ]

    for pattern in patterns:
        matches = re.findall(pattern, filename)
        for match in matches:
            year = int(match)
            if 2010 <= year <= 2030:
                return year

    # Fallback: search for any 4-digit number that looks like a year
    all_nums = re.findall(r'\b(\d{4})\b', filename)
    for num in all_nums:
        year = int(num)
        if 2010 <= year <= 2030:
            return year

    return None


def find_all_csvs(source_path):
    """Find all CSV files recursively in a data source directory."""
    csv_files = []

    # Look for CSVs in subdirectories (already extracted)
    for csv_file in source_path.rglob("*.csv"):
        csv_files.append(csv_file)

    return csv_files


def infer_unified_schema(csv_files, max_samples=20):
    """Infer unified schema from sample of CSV files."""
    all_columns = {}
    sample_files = csv_files[:max_samples] if len(csv_files) > max_samples else csv_files

    for csv_file in sample_files:
        try:
            df_sample = pd.read_csv(csv_file, nrows=100)
            for col in df_sample.columns:
                if col not in all_columns:
                    all_columns[col] = df_sample[col].dtype
                else:
                    existing_type = all_columns[col]
                    new_type = df_sample[col].dtype
                    if existing_type != new_type:
                        all_columns[col] = 'object'
        except Exception as e:
            logging.warning(f"Error reading {csv_file}: {e}")

    return all_columns


def process_year_to_parquet(args):
    """Process a single year of data to parquet."""
    source, year, year_files, unified_schema, output_dir = args

    try:
        logging.info(f"Processing {source} year {year}: {len(year_files)} CSV files")

        dfs = []
        for csv_file in year_files:
            try:
                df = pd.read_csv(csv_file, dtype=str, low_memory=False)

                # Add missing columns
                for col in unified_schema:
                    if col not in df.columns:
                        df[col] = None

                # Reorder columns
                df = df[list(unified_schema.keys())]
                dfs.append(df)

            except Exception as e:
                logging.warning(f"Error reading {csv_file}: {e}")

        if not dfs:
            return f"No data loaded for {source} year {year}"

        # Combine dataframes
        combined_df = pd.concat(dfs, ignore_index=True)

        # Try type conversion
        for col, dtype in unified_schema.items():
            if dtype != 'object':
                try:
                    combined_df[col] = pd.to_numeric(combined_df[col], errors='ignore')
                except:
                    pass

        # Create parquet file
        # Sanitize source name for filename
        safe_source = source.replace(',', '').replace(' ', '_')
        parquet_filename = f"{safe_source}_{year}.parquet"
        parquet_path = output_dir / parquet_filename

        combined_df.to_parquet(parquet_path, engine='pyarrow', compression='snappy', index=False)
        return f"✓ Created: {parquet_filename} ({len(combined_df):,} rows, {len(combined_df.columns)} cols)"

    except Exception as e:
        return f"✗ Error processing {source} year {year}: {e}"


def process_data_source(source):
    """Process a single data source."""
    source_path = BASE_DIR / source

    if not source_path.exists():
        return f"✗ Not found: {source}"

    # Find all CSVs
    csv_files = find_all_csvs(source_path)

    if not csv_files:
        return f"✗ No CSVs found in {source}"

    logging.info(f"Found {len(csv_files)} CSV files in {source}")

    # Group by year
    files_by_year = defaultdict(list)
    for csv_file in csv_files:
        year = extract_year_from_filename(csv_file.name)
        if year:
            files_by_year[year].append(csv_file)

    if not files_by_year:
        return f"✗ Could not extract years from filenames in {source}"

    # Infer schema
    unified_schema = infer_unified_schema(csv_files, max_samples=20)
    logging.info(f"{source}: {len(unified_schema)} columns, {len(files_by_year)} years")

    # Create tasks for each year
    tasks = []
    for year, year_files in sorted(files_by_year.items()):
        tasks.append((source, year, year_files, unified_schema, PARQUET_OUTPUT_DIR))

    # Process years in parallel
    with Pool(min(NUM_CORES, len(tasks))) as pool:
        results = pool.map(process_year_to_parquet, tasks)

    return results


def main():
    """Main execution."""
    logging.info("=" * 80)
    logging.info("Fast CSV to Parquet Conversion (Using Existing CSVs)")
    logging.info("=" * 80)

    # Create output directory
    PARQUET_OUTPUT_DIR.mkdir(exist_ok=True)
    logging.info(f"Output directory: {PARQUET_OUTPUT_DIR}")

    # Check which sources exist
    existing_sources = []
    for source in CRITICAL_DATA_SOURCES:
        source_path = BASE_DIR / source
        if source_path.exists():
            existing_sources.append(source)
        else:
            logging.warning(f"Not found: {source}")

    logging.info(f"Found {len(existing_sources)}/{len(CRITICAL_DATA_SOURCES)} data sources")
    logging.info("")

    # Process each data source
    all_results = []
    for i, source in enumerate(existing_sources, 1):
        logging.info(f"[{i}/{len(existing_sources)}] Processing: {source}")
        result = process_data_source(source)
        if isinstance(result, list):
            all_results.extend(result)
        else:
            all_results.append(result)
        logging.info("")

    # Summary
    logging.info("=" * 80)
    logging.info("SUMMARY")
    logging.info("=" * 80)

    successes = [r for r in all_results if "✓" in str(r)]
    failures = [r for r in all_results if "✗" in str(r)]

    logging.info(f"Successful conversions: {len(successes)}")
    logging.info(f"Failed conversions: {len(failures)}")

    if failures:
        logging.info("\nFailures:")
        for f in failures:
            logging.info(f"  {f}")

    # List output files
    parquet_files = list(PARQUET_OUTPUT_DIR.glob("*.parquet"))
    logging.info(f"\nTotal parquet files created: {len(parquet_files)}")
    logging.info(f"Output location: {PARQUET_OUTPUT_DIR}")

    logging.info("=" * 80)
    logging.info("COMPLETE!")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()

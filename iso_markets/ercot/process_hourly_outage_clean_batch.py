#!/usr/bin/env python3
"""
Process Hourly Resource Outage Capacity Dataset from ERCOT_clean_batch_dataset
1. Extracts all zip files to CSV subdirectory
2. Converts CSV files to yearly parquet files
3. Moves parquet files to parquet/Hourly Resource Outage Capacity directory
"""

import os
import zipfile
import shutil
import pandas as pd
from pathlib import Path
from collections import defaultdict
import re
import logging
from multiprocessing import Pool

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hourly_outage_processing.log'),
        logging.StreamHandler()
    ]
)

# Paths
DATA_SOURCE_DIR = Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_clean_batch_dataset/Hourly Resource Outage Capacity")
PARQUET_OUTPUT_DIR = Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_clean_batch_dataset/parquet/Hourly Resource Outage Capacity")
NUM_CORES = 12


def extract_year_from_filename(filename):
    """Extract year from ERCOT filename formats."""
    patterns = [
        r'\.(\d{4})\d{4}\.',  # .YYYYMMDD.
        r'_(\d{4})\d{4}_',    # _YYYYMMDD_
        r'_(\d{4})\d{4}\.',   # _YYYYMMDD.
        r'\.(\d{4})\d{4}_',   # .YYYYMMDD_
        r'(\d{4})-\d{2}-\d{2}',  # YYYY-MM-DD
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


def extract_single_zip(zip_file_path):
    """Extract a single zip file."""
    try:
        zip_file = Path(zip_file_path)
        csv_dir = DATA_SOURCE_DIR / "csv"

        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            # Extract all CSV files directly to csv directory
            for member in zip_ref.namelist():
                if member.endswith('.csv'):
                    # Extract to csv directory
                    filename = os.path.basename(member)
                    source = zip_ref.open(member)
                    target_path = csv_dir / filename

                    with open(target_path, 'wb') as target:
                        shutil.copyfileobj(source, target)

        return f"✓ Extracted: {zip_file.name}"

    except Exception as e:
        return f"✗ Error extracting {zip_file_path}: {e}"


def extract_zip_files():
    """Extract all zip files in the data directory."""
    logging.info("=" * 80)
    logging.info("STEP 1: EXTRACTING ZIP FILES")
    logging.info("=" * 80)

    # Create CSV directory
    csv_dir = DATA_SOURCE_DIR / "csv"
    csv_dir.mkdir(exist_ok=True)
    logging.info(f"CSV directory: {csv_dir}")

    # Find all zip files
    zip_files = list(DATA_SOURCE_DIR.glob("*.zip"))
    logging.info(f"Found {len(zip_files)} zip files")

    if not zip_files:
        logging.warning("No zip files found!")
        return

    # Extract in parallel
    logging.info(f"Extracting using {NUM_CORES} cores...")
    with Pool(NUM_CORES) as pool:
        results = pool.map(extract_single_zip, zip_files)

    # Log results
    successes = sum(1 for r in results if "✓" in r)
    failures = sum(1 for r in results if "✗" in r)

    logging.info(f"Extracted: {successes} successful, {failures} failed")

    for result in results:
        if "✗" in result:
            logging.error(result)


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
    year, year_files, unified_schema = args

    try:
        logging.info(f"Processing year {year}: {len(year_files)} CSV files")

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
            return f"✗ No data loaded for year {year}"

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
        parquet_filename = f"Hourly Resource Outage Capacity_{year}.parquet"
        parquet_path = PARQUET_OUTPUT_DIR / parquet_filename

        combined_df.to_parquet(parquet_path, engine='pyarrow', compression='snappy', index=False)

        file_size_mb = parquet_path.stat().st_size / (1024 * 1024)
        return f"✓ Created: {parquet_filename} ({len(combined_df):,} rows, {file_size_mb:.1f} MB)"

    except Exception as e:
        return f"✗ Error processing year {year}: {e}"


def csv_to_yearly_parquet():
    """Convert CSV files to yearly parquet files."""
    logging.info("=" * 80)
    logging.info("STEP 2: CONVERTING CSV TO YEARLY PARQUET FILES")
    logging.info("=" * 80)

    csv_dir = DATA_SOURCE_DIR / "csv"

    if not csv_dir.exists():
        logging.error(f"CSV directory not found: {csv_dir}")
        return

    # Create output directory
    PARQUET_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory: {PARQUET_OUTPUT_DIR}")

    # Find all CSV files
    csv_files = list(csv_dir.glob("*.csv"))

    if not csv_files:
        logging.error(f"No CSV files found in {csv_dir}")
        return

    logging.info(f"Found {len(csv_files)} CSV files")

    # Group files by year
    files_by_year = defaultdict(list)
    for csv_file in csv_files:
        year = extract_year_from_filename(csv_file.name)
        if year:
            files_by_year[year].append(csv_file)
        else:
            logging.warning(f"Could not extract year from: {csv_file.name}")

    if not files_by_year:
        logging.error("Could not extract years from any filenames!")
        return

    logging.info(f"Found data for {len(files_by_year)} years: {sorted(files_by_year.keys())}")

    # Infer schema
    logging.info("Inferring unified schema...")
    unified_schema = infer_unified_schema(csv_files, max_samples=20)
    logging.info(f"Schema has {len(unified_schema)} columns")

    # Create tasks for each year
    tasks = []
    for year, year_files in sorted(files_by_year.items()):
        tasks.append((year, year_files, unified_schema))

    # Process years in parallel
    logging.info(f"Processing {len(tasks)} years using {NUM_CORES} cores...")
    with Pool(min(NUM_CORES, len(tasks))) as pool:
        results = pool.map(process_year_to_parquet, tasks)

    # Log results
    successes = [r for r in results if "✓" in r]
    failures = [r for r in results if "✗" in r]

    logging.info(f"\nConversion results:")
    logging.info(f"  Successful: {len(successes)}")
    logging.info(f"  Failed: {len(failures)}")

    for result in results:
        if "✓" in result:
            logging.info(result)
        else:
            logging.error(result)


def main():
    """Main execution."""
    logging.info("=" * 80)
    logging.info("HOURLY RESOURCE OUTAGE CAPACITY PROCESSING")
    logging.info("=" * 80)
    logging.info(f"Data source: {DATA_SOURCE_DIR}")
    logging.info(f"Parquet output: {PARQUET_OUTPUT_DIR}")
    logging.info("")

    # Verify data source exists
    if not DATA_SOURCE_DIR.exists():
        logging.error(f"Data source directory not found: {DATA_SOURCE_DIR}")
        return

    # Step 1: Extract zip files to CSV
    extract_zip_files()

    # Step 2: Convert CSV to yearly parquet
    csv_to_yearly_parquet()

    # Summary
    logging.info("=" * 80)
    logging.info("COMPLETE!")
    logging.info("=" * 80)

    parquet_files = list(PARQUET_OUTPUT_DIR.glob("*.parquet"))
    logging.info(f"Total parquet files created: {len(parquet_files)}")

    total_size = sum(f.stat().st_size for f in parquet_files) / (1024 * 1024)
    logging.info(f"Total size: {total_size:.1f} MB")
    logging.info(f"Location: {PARQUET_OUTPUT_DIR}")


if __name__ == "__main__":
    main()

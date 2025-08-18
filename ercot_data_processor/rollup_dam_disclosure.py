#!/usr/bin/env python3
"""
Roll up 60-day DAM Disclosure Reports into annual CSV files.
Processes all distinct DAM file types and creates time-ordered annual rollups.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import re
from typing import Dict, List, Tuple
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
import os
import sys
from dotenv import load_dotenv
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get base directory from environment or use default
def get_ercot_data_dir():
    """Get ERCOT data directory from environment or platform-specific default."""
    data_dir = os.getenv("ERCOT_DATA_DIR")
    if data_dir:
        return Path(data_dir)
    # Default based on platform
    if sys.platform == "linux":
        return Path("/home/enrico/data/ERCOT_data")
    else:
        return Path("/Users/enrico/data/ERCOT_data")

# Base paths
BASE_DIR = get_ercot_data_dir()
DISCLOSURE_DIR = BASE_DIR / "60-Day_DAM_Disclosure_Reports"
ROLLUP_DIR = BASE_DIR / "rollup_dam_disclosure"

# Define DAM file type patterns based on the image
DAM_FILE_PATTERNS = {
    'EnergyBidAwards': r'60d_DAM_EnergyBidAwards-\d{2}-[A-Z]{3}-\d{2}\.csv',
    'EnergyBids': r'60d_DAM_EnergyBids-\d{2}-[A-Z]{3}-\d{2}\.csv',
    'EnergyOnlyOfferAwards': r'60d_DAM_EnergyOnlyOfferAwards-\d{2}-[A-Z]{3}-\d{2}\.csv',
    'EnergyOnlyOffers': r'60d_DAM_EnergyOnlyOffers-\d{2}-[A-Z]{3}-\d{2}\.csv',
    'Gen_Resource_Data': r'60d_DAM_Gen_Resource_Data-\d{2}-[A-Z]{3}-\d{2}\.csv',
    'Generation_Resource_ASOffers': r'60d_DAM_Generation_Resource_ASOffers-\d{2}-[A-Z]{3}-\d{2}\.csv',
    'Load_Resource_ASOffers': r'60d_DAM_Load_Resource_ASOffers-\d{2}-[A-Z]{3}-\d{2}\.csv',
    'Load_Resource_Data': r'60d_DAM_Load_Resource_Data-\d{2}-[A-Z]{3}-\d{2}\.csv',
    'PTP_Obligation_Option': r'60d_DAM_PTP_Obligation_Option-\d{2}-[A-Z]{3}-\d{2}\.csv',
    'PTPObligationBidAwards': r'60d_DAM_PTPObligationBidAwards-\d{2}-[A-Z]{3}-\d{2}\.csv',
    'QSE_Self_Arranged_AS': r'60d_DAM_QSE_Self_Arranged_AS-\d{2}-[A-Z]{3}-\d{2}\.csv',
    'PTP_Obligation_OptionAwards': r'60d_DAM_PTP_Obligation_OptionAwards-\d{2}-[A-Z]{3}-\d{2}\.csv',
    'PTPObligationBids': r'60d_DAM_PTPObligationBids-\d{2}-[A-Z]{3}-\d{2}\.csv',
}

def parse_date_from_filename(filename: str) -> datetime:
    """Extract date from filename like '60d_DAM_EnergyBids-15-AUG-25.csv'"""
    match = re.search(r'(\d{2})-([A-Z]{3})-(\d{2})', filename)
    if match:
        day = int(match.group(1))
        month_str = match.group(2)
        year = int(match.group(3))
        
        # Convert 2-digit year to 4-digit
        year = 2000 + year if year < 50 else 1900 + year
        
        # Month mapping
        months = {
            'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
            'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
        }
        month = months.get(month_str, 1)
        
        return datetime(year, month, day)
    return None

def get_files_by_type(file_type: str, pattern: str) -> Dict[int, List[Path]]:
    """Get all files of a specific type grouped by year."""
    csv_dir = DISCLOSURE_DIR / "csv"
    if not csv_dir.exists():
        csv_dir = DISCLOSURE_DIR
    
    files_by_year = {}
    
    for file_path in csv_dir.glob("*.csv"):
        if re.match(pattern, file_path.name):
            file_date = parse_date_from_filename(file_path.name)
            if file_date:
                year = file_date.year
                if year not in files_by_year:
                    files_by_year[year] = []
                files_by_year[year].append((file_date, file_path))
    
    # Sort files within each year by date
    for year in files_by_year:
        files_by_year[year].sort(key=lambda x: x[0])
        files_by_year[year] = [fp for _, fp in files_by_year[year]]
    
    return files_by_year

def read_dam_file(file_path: Path) -> pd.DataFrame:
    """Read a DAM disclosure CSV file with proper data types."""
    try:
        # First, detect the columns
        sample_df = pd.read_csv(file_path, nrows=5)
        
        # Define dtype mappings for common columns
        dtype_dict = {}
        for col in sample_df.columns:
            if 'Price' in col or 'MW' in col or 'Amount' in col or 'Quantity' in col:
                dtype_dict[col] = 'float64'
            elif 'Date' in col or 'Time' in col:
                continue  # Let pandas parse dates
            else:
                dtype_dict[col] = 'str'
        
        # Read the full file
        df = pd.read_csv(file_path, dtype=dtype_dict, low_memory=False)
        
        # Add source file for tracking
        df['source_file'] = file_path.name
        
        # Parse date columns
        for col in df.columns:
            if 'Date' in col and df[col].dtype == 'object':
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except:
                    pass
        
        return df
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return pd.DataFrame()

def process_file_type(file_type: str, pattern: str) -> Dict[int, int]:
    """Process all files of a specific type and create annual rollups."""
    logger.info(f"Processing {file_type} files...")
    
    files_by_year = get_files_by_type(file_type, pattern)
    
    if not files_by_year:
        logger.warning(f"No files found for {file_type}")
        return {}
    
    # Create output directory for this file type
    output_dir = ROLLUP_DIR / file_type
    output_dir.mkdir(parents=True, exist_ok=True)
    
    row_counts = {}
    
    for year, file_list in sorted(files_by_year.items()):
        logger.info(f"  Processing {year}: {len(file_list)} files")
        
        # Read all files for this year
        dfs = []
        for file_path in file_list:
            df = read_dam_file(file_path)
            if not df.empty:
                dfs.append(df)
        
        if dfs:
            # Combine all dataframes
            combined_df = pd.concat(dfs, ignore_index=True)
            
            # Sort by date columns if they exist
            date_cols = [col for col in combined_df.columns if 'Date' in col or 'Time' in col]
            if date_cols:
                # Try to sort by the first date column
                try:
                    combined_df = combined_df.sort_values(by=date_cols[0])
                except:
                    pass
            
            # Save to CSV
            output_file = output_dir / f"{file_type}_{year}.csv"
            combined_df.to_csv(output_file, index=False)
            
            row_counts[year] = len(combined_df)
            logger.info(f"    Saved {len(combined_df):,} rows to {output_file.name}")
            
            # Also save a parquet version for faster loading
            parquet_file = output_dir / f"{file_type}_{year}.parquet"
            combined_df.to_parquet(parquet_file, engine='pyarrow', compression='snappy')
    
    return row_counts

def generate_summary_report(results: Dict[str, Dict[int, int]]):
    """Generate a summary report of the rollup process."""
    report_path = ROLLUP_DIR / "rollup_summary.txt"
    
    with open(report_path, 'w') as f:
        f.write("60-Day DAM Disclosure Rollup Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for file_type, year_counts in results.items():
            f.write(f"\n{file_type}:\n")
            f.write("-" * 30 + "\n")
            
            if year_counts:
                total_rows = 0
                for year, count in sorted(year_counts.items()):
                    f.write(f"  {year}: {count:,} rows\n")
                    total_rows += count
                f.write(f"  Total: {total_rows:,} rows\n")
            else:
                f.write("  No data found\n")
    
    logger.info(f"Summary report saved to {report_path}")

def main():
    """Main execution function."""
    logger.info("Starting 60-Day DAM Disclosure Rollup")
    logger.info(f"Source directory: {DISCLOSURE_DIR}")
    logger.info(f"Output directory: {ROLLUP_DIR}")
    
    # Create rollup directory
    ROLLUP_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process each file type
    results = {}
    
    # Process files sequentially to avoid memory issues
    for file_type, pattern in DAM_FILE_PATTERNS.items():
        row_counts = process_file_type(file_type, pattern)
        results[file_type] = row_counts
    
    # Generate summary report
    generate_summary_report(results)
    
    logger.info("Rollup complete!")
    
    # Print summary to console
    print("\n" + "=" * 60)
    print("60-DAY DAM DISCLOSURE ROLLUP SUMMARY")
    print("=" * 60)
    
    for file_type, year_counts in results.items():
        if year_counts:
            print(f"\n{file_type}:")
            for year, count in sorted(year_counts.items()):
                print(f"  {year}: {count:,} rows")

if __name__ == "__main__":
    main()
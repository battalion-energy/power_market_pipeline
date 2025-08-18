#!/usr/bin/env python3
"""
Create hourly aggregated version of RT prices for easier comparison with DA prices.
This is a supplementary script - the main flatten_ercot_prices.py keeps 15-minute intervals.
"""

import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
import sys
from datetime import datetime
import numpy as np


def aggregate_rt_to_hourly(year: int, input_dir: Path, output_dir: Path):
    """Aggregate 15-minute RT prices to hourly averages."""
    input_file = input_dir / f"RT_prices_15min_{year}.parquet"
    if not input_file.exists():
        # Try original RT file if 15-min doesn't exist
        input_file = input_dir / f"RT_prices_{year}.parquet"
        if not input_file.exists():
            print(f"Skipping {year} - no RT file found")
            return
    
    print(f"Aggregating RT {year} to hourly...")
    
    # Read the 15-minute data
    df = pd.read_parquet(input_file)
    
    # Ensure datetime is proper type
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Create hour column
    df['hour'] = df['datetime'].dt.floor('h')
    
    # Get price columns (all except datetime, hour, and interval columns)
    exclude_cols = ['datetime', 'hour', 'DeliveryInterval', 'Interval']
    price_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Group by hour and take mean of all price columns
    df_hourly = df.groupby('hour')[price_cols].mean().reset_index()
    df_hourly.rename(columns={'hour': 'datetime'}, inplace=True)
    
    # Sort by datetime
    df_hourly = df_hourly.sort_values('datetime')
    
    # Save to parquet
    output_file = output_dir / f"RT_prices_hourly_{year}.parquet"
    df_hourly.to_parquet(output_file, index=False)
    print(f"  Saved {len(df_hourly)} hours to {output_file}")
    print(f"  Shape: {df_hourly.shape}")
    
    return df_hourly


def main():
    # Set up directories
    base_dir = Path("/home/enrico/data/ERCOT_data/rollup_files")
    input_dir = base_dir / "flattened"
    output_dir = base_dir / "flattened"
    
    # Get available years from 15-min RT files
    rt_15min_years = sorted([
        int(f.stem.replace('RT_prices_15min_', '')) 
        for f in input_dir.glob("RT_prices_15min_*.parquet")
    ])
    
    if not rt_15min_years:
        print("No 15-minute RT files found. Run flatten_ercot_prices.py first.")
        return
    
    print(f"Found 15-minute RT data for years: {rt_15min_years}")
    
    # Process each year
    print("\n=== Creating Hourly Aggregated RT Prices ===")
    for year in rt_15min_years:
        aggregate_rt_to_hourly(year, input_dir, output_dir)
    
    print("\n=== Processing Complete ===")
    print(f"Hourly RT files saved to: {output_dir}")
    print("\nYou now have:")
    print("  • RT_prices_15min_YYYY.parquet - Original 15-minute intervals (4 per hour)")
    print("  • RT_prices_hourly_YYYY.parquet - Hourly averages for DA comparison")


if __name__ == "__main__":
    main()
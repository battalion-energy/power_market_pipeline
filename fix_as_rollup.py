#!/usr/bin/env python3
"""
Fix AS price rollup to include all days from CSV files.
"""

import pandas as pd
from pathlib import Path
import sys

def process_as_prices(year: int):
    """Process all AS price CSV files for a given year."""
    
    csv_dir = Path("/home/enrico/data/ERCOT_data/DAM_Clearing_Prices_for_Capacity/csv")
    output_dir = Path("/home/enrico/data/ERCOT_data/rollup_files/AS_prices")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all CSV files for this year
    pattern = f"*{year}*.csv"
    year_files = sorted(csv_dir.glob(pattern))
    
    # Filter to only DAM capacity files
    year_files = [f for f in year_files if "00012300" in f.name or "DAMC" in f.name or "DAM" in f.name]
    
    print(f"Processing AS {year}: Found {len(year_files)} files")
    
    if not year_files:
        return
    
    # Process files in chunks to manage memory
    chunk_size = 50
    all_dfs = []
    
    for i in range(0, len(year_files), chunk_size):
        print(f"  Processing chunk {i//chunk_size + 1}/{(len(year_files) + chunk_size - 1)//chunk_size}")
        chunk_files = year_files[i:i+chunk_size]
        chunk_dfs = []
        
        for f in chunk_files:
            try:
                df = pd.read_csv(f)
                chunk_dfs.append(df)
            except Exception as e:
                print(f"Error reading {f.name}: {e}")
                continue
        
        if chunk_dfs:
            # Combine chunk
            chunk_combined = pd.concat(chunk_dfs, ignore_index=True)
            all_dfs.append(chunk_combined)
    
    if not all_dfs:
        print(f"No data for {year}")
        return
    
    # Combine all chunks
    print(f"Combining {len(all_dfs)} chunks...")
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Add datetime column
    combined_df['DeliveryDate'] = pd.to_datetime(combined_df['DeliveryDate'], format='%m/%d/%Y')
    combined_df['hour'] = combined_df['HourEnding'].str.extract(r'(\d+)').astype(int)
    combined_df['datetime'] = combined_df['DeliveryDate'] + pd.to_timedelta(combined_df['hour'] - 1, unit='h')
    
    # Convert back to string format for compatibility
    combined_df['DeliveryDate'] = combined_df['DeliveryDate'].dt.strftime('%Y-%m-%d')
    
    # Save to parquet
    output_file = output_dir / f"{year}.parquet"
    combined_df.to_parquet(output_file, index=False)
    
    print(f"✅ Saved {len(combined_df)} rows to {output_file}")
    
    # Check unique dates
    unique_dates = combined_df['datetime'].dt.date.nunique()
    print(f"   Unique dates: {unique_dates}")
    
    return combined_df


def main():
    """Process all years or specific year."""
    
    years = range(2010, 2026)
    
    if len(sys.argv) > 1:
        years = [int(sys.argv[1])]
    
    for year in years:
        process_as_prices(year)
    
    print("\n✅ AS price rollup complete!")


if __name__ == "__main__":
    main()
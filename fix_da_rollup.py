#!/usr/bin/env python3
"""
Fix DA price rollup to include all days from CSV files.
"""

import pandas as pd
from pathlib import Path
import sys
# Use simple progress instead of tqdm
import pyarrow.parquet as pq
import pyarrow as pa

def process_da_prices(year: int):
    """Process all DA price CSV files for a given year."""
    
    csv_dir = Path("/home/enrico/data/ERCOT_data/DAM_Settlement_Point_Prices/csv")
    output_dir = Path("/home/enrico/data/ERCOT_data/rollup_files/DA_prices")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all CSV files for this year - match the date pattern YYYYMMDD
    # The filename format is: cdr.00012331.0000000000000000.YYYYMMDD.HHMMSS.DAMSPNP4190.csv
    pattern = f"*.{year}*.csv"
    all_files = sorted(csv_dir.glob(pattern))
    
    # Filter to only DAMSPNP files with correct year in date position
    year_files = []
    for f in all_files:
        if "DAMSPNP" in f.name or "00012331" in f.name:
            # Extract the date part (4th component when split by '.')
            parts = f.name.split('.')
            if len(parts) >= 4:
                date_part = parts[3]
                if date_part.startswith(str(year)):
                    year_files.append(f)
    
    print(f"Processing {year}: Found {len(year_files)} files")
    
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
        process_da_prices(year)
    
    print("\n✅ DA price rollup complete!")


if __name__ == "__main__":
    main()
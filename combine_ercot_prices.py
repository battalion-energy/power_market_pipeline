#!/usr/bin/env python3
"""
Combine ERCOT price data: DA + AS, and DA + AS + RT
Also break down annual files into monthly files
"""

import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from pathlib import Path
import sys
from datetime import datetime
import numpy as np


def combine_da_as_prices(year: int, input_dir: Path, output_dir: Path):
    """Combine DA and AS prices for a given year."""
    da_file = input_dir / f"DA_prices_{year}.parquet"
    as_file = input_dir / f"AS_prices_{year}.parquet"
    
    if not da_file.exists():
        print(f"Skipping {year} - DA file not found")
        return None
    
    print(f"Combining DA + AS for {year}...")
    
    # Read DA prices
    df_da = pd.read_parquet(da_file)
    
    # Read AS prices if exists
    if as_file.exists():
        df_as = pd.read_parquet(as_file)
        # Both should have matching date columns, merge on all date columns
        merge_cols = ['DeliveryDate', 'DeliveryDateStr', 'datetime_ts']
        # Keep only columns that exist in both
        merge_cols = [col for col in merge_cols if col in df_da.columns and col in df_as.columns]
        if not merge_cols:
            # Fallback to old column name
            merge_cols = ['datetime'] if 'datetime' in df_da.columns else ['datetime_ts']
        
        # Remove duplicate columns from AS before merging
        as_cols_to_keep = [col for col in df_as.columns if col not in df_da.columns or col in merge_cols]
        df_as = df_as[as_cols_to_keep]
        
        df_combined = pd.merge(df_da, df_as, on=merge_cols, how='outer')
    else:
        print(f"  AS file not found for {year}, using DA only")
        df_combined = df_da
    
    # Sort by datetime
    sort_col = 'datetime_ts' if 'datetime_ts' in df_combined.columns else 'datetime'
    df_combined = df_combined.sort_values(sort_col)
    
    # Ensure Date32 format is preserved
    table = pa.Table.from_pandas(df_combined)
    new_fields = []
    for field in table.schema:
        if field.name == 'DeliveryDate' and str(field.type) != 'date32':
            new_fields.append(pa.field('DeliveryDate', pa.date32()))
        else:
            new_fields.append(field)
    
    if new_fields != list(table.schema):
        table = table.cast(pa.schema(new_fields))
    
    # Save combined file
    output_file = output_dir / f"DA_AS_combined_{year}.parquet"
    pq.write_table(table, output_file)
    print(f"  Saved {len(df_combined)} rows to {output_file}")
    print(f"  Shape: {df_combined.shape}")
    print(f"  Date format: Preserved Date32 format")
    
    return df_combined


def combine_da_as_rt_prices(year: int, input_dir: Path, output_dir: Path):
    """Combine DA, AS, and RT prices for a given year.
    
    Since RT is 15-minute data and DA/AS are hourly, we need special handling:
    - Option 1: Use hourly aggregated RT for alignment
    - Option 2: Keep 15-min RT and replicate DA/AS for each interval
    """
    da_file = input_dir / f"DA_prices_{year}.parquet"
    as_file = input_dir / f"AS_prices_{year}.parquet"
    rt_15min_file = input_dir / f"RT_prices_15min_{year}.parquet"
    rt_hourly_file = input_dir / f"RT_prices_hourly_{year}.parquet"
    
    if not da_file.exists():
        print(f"Skipping {year} - DA file not found")
        return None
    
    # Check which RT file to use
    use_15min = rt_15min_file.exists()
    use_hourly = rt_hourly_file.exists()
    
    if use_15min and not use_hourly:
        print(f"Creating hourly RT aggregation for {year}...")
        # Create hourly aggregation on the fly
        df_rt_15min = pd.read_parquet(rt_15min_file)
        # Handle datetime_ts column from flattened files
        if 'datetime_ts' in df_rt_15min.columns:
            df_rt_15min['datetime'] = pd.to_datetime(df_rt_15min['datetime_ts'])
        else:
            df_rt_15min['datetime'] = pd.to_datetime(df_rt_15min['datetime'])
        df_rt_15min['hour'] = df_rt_15min['datetime'].dt.floor('h')
        
        # Get price columns (exclude datetime, hour, interval columns)
        exclude_cols = ['datetime', 'datetime_ts', 'hour', 'DeliveryInterval', 'DeliveryDate', 'DeliveryDateStr', 'Interval']
        price_cols = [col for col in df_rt_15min.columns if col not in exclude_cols]
        
        # Aggregate to hourly
        df_rt = df_rt_15min.groupby('hour')[price_cols].mean().reset_index()
        df_rt.rename(columns={'hour': 'datetime'}, inplace=True)
    elif use_hourly:
        df_rt = pd.read_parquet(rt_hourly_file)
    else:
        df_rt = None
        print(f"  No RT file found for {year}")
    
    print(f"Combining DA + AS + RT (hourly aligned) for {year}...")
    
    # Read DA prices
    df_da = pd.read_parquet(da_file)
    # Handle datetime_ts column from flattened files
    if 'datetime_ts' in df_da.columns:
        df_da['datetime'] = pd.to_datetime(df_da['datetime_ts'])
    else:
        df_da['datetime'] = pd.to_datetime(df_da['datetime'])
    # Add prefix to DA columns (except datetime)
    df_da = df_da.rename(columns={col: f"DA_{col}" if col != 'datetime' else col for col in df_da.columns})
    
    # Read AS prices if exists
    if as_file.exists():
        df_as = pd.read_parquet(as_file)
        # Handle datetime_ts column from flattened files
        if 'datetime_ts' in df_as.columns:
            df_as['datetime'] = pd.to_datetime(df_as['datetime_ts'])
        else:
            df_as['datetime'] = pd.to_datetime(df_as['datetime'])
        # Add prefix to AS columns (except datetime)
        df_as = df_as.rename(columns={col: f"AS_{col}" if col != 'datetime' else col for col in df_as.columns})
        # Merge DA and AS
        df_combined = pd.merge(df_da, df_as, on='datetime', how='outer')
    else:
        print(f"  AS file not found for {year}")
        df_combined = df_da
    
    # Merge with RT if exists
    if df_rt is not None:
        # Handle datetime_ts column from flattened files
        if 'datetime_ts' in df_rt.columns:
            df_rt['datetime'] = pd.to_datetime(df_rt['datetime_ts'])
        else:
            df_rt['datetime'] = pd.to_datetime(df_rt['datetime'])
        # Add prefix to RT columns (except datetime)
        df_rt = df_rt.rename(columns={col: f"RT_{col}" if col != 'datetime' else col for col in df_rt.columns})
        # Merge with hourly RT
        df_combined = pd.merge(df_combined, df_rt, on='datetime', how='outer')
        print(f"  Merged {len(df_rt)} hours of RT data")
    
    # Sort by datetime
    df_combined = df_combined.sort_values('datetime')
    
    # Save combined file
    output_file = output_dir / f"DA_AS_RT_combined_{year}.parquet"
    df_combined.to_parquet(output_file, index=False)
    print(f"  Saved {len(df_combined)} rows to {output_file}")
    print(f"  Shape: {df_combined.shape}")
    
    return df_combined


def combine_da_as_rt_15min(year: int, input_dir: Path, output_dir: Path):
    """Create a 15-minute resolution combined file with DA/AS values repeated for each RT interval."""
    da_file = input_dir / f"DA_prices_{year}.parquet"
    as_file = input_dir / f"AS_prices_{year}.parquet"
    rt_file = input_dir / f"RT_prices_15min_{year}.parquet"
    
    if not rt_file.exists():
        print(f"Skipping 15-min combined for {year} - RT 15-min file not found")
        return None
    
    print(f"Creating 15-minute resolution combined file for {year}...")
    
    # Read RT 15-minute data as base
    df_rt = pd.read_parquet(rt_file)
    # Handle datetime_ts column from flattened files
    if 'datetime_ts' in df_rt.columns:
        df_rt['datetime'] = pd.to_datetime(df_rt['datetime_ts'])
    else:
        df_rt['datetime'] = pd.to_datetime(df_rt['datetime'])
    df_rt['hour'] = df_rt['datetime'].dt.floor('h')
    
    # Add RT prefix to price columns
    interval_cols = ['datetime', 'datetime_ts', 'hour', 'DeliveryInterval', 'DeliveryDate', 'DeliveryDateStr', 'Interval']
    rt_price_cols = [col for col in df_rt.columns if col not in interval_cols]
    for col in rt_price_cols:
        df_rt = df_rt.rename(columns={col: f"RT_{col}"})
    
    # Read DA prices
    if da_file.exists():
        df_da = pd.read_parquet(da_file)
        # Handle datetime_ts column from flattened files
        if 'datetime_ts' in df_da.columns:
            df_da['datetime'] = pd.to_datetime(df_da['datetime_ts'])
        else:
            df_da['datetime'] = pd.to_datetime(df_da['datetime'])
        # Rename datetime to hour for merging
        df_da = df_da.rename(columns={'datetime': 'hour'})
        # Add DA prefix
        df_da = df_da.rename(columns={col: f"DA_{col}" if col != 'hour' else col for col in df_da.columns})
        # Merge DA with RT on hour
        df_combined = pd.merge(df_rt, df_da, on='hour', how='left')
    else:
        df_combined = df_rt
    
    # Read AS prices
    if as_file.exists():
        df_as = pd.read_parquet(as_file)
        # Handle datetime_ts column from flattened files
        if 'datetime_ts' in df_as.columns:
            df_as['datetime'] = pd.to_datetime(df_as['datetime_ts'])
        else:
            df_as['datetime'] = pd.to_datetime(df_as['datetime'])
        # Rename datetime to hour for merging
        df_as = df_as.rename(columns={'datetime': 'hour'})
        # Add AS prefix
        df_as = df_as.rename(columns={col: f"AS_{col}" if col != 'hour' else col for col in df_as.columns})
        # Merge AS with combined
        df_combined = pd.merge(df_combined, df_as, on='hour', how='left')
    
    # Drop the hour column as we have datetime
    df_combined = df_combined.drop(columns=['hour'])
    
    # Sort by datetime
    df_combined = df_combined.sort_values('datetime')
    
    # Reorder columns to have datetime and interval first
    base_cols = ['datetime']
    if 'DeliveryInterval' in df_combined.columns:
        base_cols.append('DeliveryInterval')
    other_cols = [col for col in df_combined.columns if col not in base_cols]
    df_combined = df_combined[base_cols + sorted(other_cols)]
    
    # Save combined file
    output_file = output_dir / f"DA_AS_RT_15min_combined_{year}.parquet"
    df_combined.to_parquet(output_file, index=False)
    print(f"  Saved {len(df_combined)} 15-min intervals to {output_file}")
    print(f"  Shape: {df_combined.shape}")
    print(f"  DA/AS values repeated for each 15-min RT interval")
    
    return df_combined


def break_into_monthly_files(annual_file: Path, output_dir: Path, file_prefix: str):
    """Break an annual file into monthly parquet files."""
    if not annual_file.exists():
        print(f"File not found: {annual_file}")
        return
    
    year = annual_file.stem.split('_')[-1]
    print(f"Breaking {file_prefix}_{year} into monthly files...")
    
    # Read annual file
    df = pd.read_parquet(annual_file)
    
    # Handle datetime_ts column from flattened/combined files
    if 'datetime_ts' in df.columns and 'datetime' not in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime_ts'])
    elif 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
    else:
        raise ValueError(f"No datetime column found in {annual_file}")
    
    # Group by year-month
    df['year_month'] = df['datetime'].dt.to_period('M')
    
    # Create monthly subdirectory
    monthly_dir = output_dir / "monthly" / file_prefix
    monthly_dir.mkdir(parents=True, exist_ok=True)
    
    # Save each month
    for month, month_df in df.groupby('year_month'):
        # Drop the temporary year_month column
        month_df = month_df.drop(columns=['year_month'])
        
        # Format filename: prefix_YYYY_MM.parquet
        month_str = str(month).replace('-', '_')
        monthly_file = monthly_dir / f"{file_prefix}_{month_str}.parquet"
        
        month_df.to_parquet(monthly_file, index=False)
        print(f"  Saved {len(month_df)} rows to {monthly_file.name}")


def main():
    # Set up directories
    base_dir = Path("/home/enrico/data/ERCOT_data/rollup_files")
    input_dir = base_dir / "flattened"
    output_dir = base_dir / "combined"
    output_dir.mkdir(exist_ok=True)
    
    # Get available years from flattened directory
    da_years = sorted([int(f.stem.split('_')[-1]) for f in input_dir.glob("DA_prices_*.parquet")])
    print(f"Found DA price data for years: {da_years}")
    
    # Process combined DA + AS
    print("\n=== Combining DA + AS Prices ===")
    for year in da_years:
        df_combined = combine_da_as_prices(year, input_dir, output_dir)
        if df_combined is not None:
            # Break into monthly files
            annual_file = output_dir / f"DA_AS_combined_{year}.parquet"
            break_into_monthly_files(annual_file, output_dir, "DA_AS_combined")
    
    # Process combined DA + AS + RT (hourly aligned)
    print("\n=== Combining DA + AS + RT Prices (Hourly) ===")
    for year in da_years:
        df_combined = combine_da_as_rt_prices(year, input_dir, output_dir)
        if df_combined is not None:
            # Break into monthly files
            annual_file = output_dir / f"DA_AS_RT_combined_{year}.parquet"
            break_into_monthly_files(annual_file, output_dir, "DA_AS_RT_combined")
    
    # Check if we have 15-minute RT files
    rt_15min_years = sorted([
        int(f.stem.replace('RT_prices_15min_', '')) 
        for f in input_dir.glob("RT_prices_15min_*.parquet")
    ])
    
    if rt_15min_years:
        # Process 15-minute resolution combined files
        print("\n=== Creating 15-Minute Resolution Combined Files ===")
        for year in rt_15min_years:
            df_combined = combine_da_as_rt_15min(year, input_dir, output_dir)
            if df_combined is not None:
                # Break into monthly files
                annual_file = output_dir / f"DA_AS_RT_15min_combined_{year}.parquet"
                break_into_monthly_files(annual_file, output_dir, "DA_AS_RT_15min_combined")
    
    print("\n=== Processing Complete ===")
    print(f"Combined files saved to: {output_dir}")
    print(f"Monthly files saved to: {output_dir / 'monthly'}")
    print("\nFile types created:")
    print("  • DA_AS_combined_YYYY.parquet - Hourly DA + AS")
    print("  • DA_AS_RT_combined_YYYY.parquet - Hourly DA + AS + RT")
    print("  • DA_AS_RT_15min_combined_YYYY.parquet - 15-min RT with hourly DA/AS repeated")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Flatten ERCOT price data from long format to wide format.
Each row is an hour, each column is a settlement point (HBs, LZs, DCs).
"""

import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.compute as pc
from pathlib import Path
import sys
from datetime import datetime, date
import numpy as np


def flatten_da_prices(year: int, input_dir: Path, output_dir: Path):
    """Flatten DA Energy Prices for a given year."""
    input_file = input_dir / f"{year}.parquet"
    if not input_file.exists():
        print(f"Skipping {year} - file not found: {input_file}")
        return
    
    print(f"Processing DA Energy Prices for {year}...")
    
    # Read the data
    df = pd.read_parquet(input_file)
    
    # Handle DeliveryDate - could be date object or string
    if df['DeliveryDate'].dtype == 'object':
        # Check if it's already a date object
        first_val = df['DeliveryDate'].iloc[0] if len(df) > 0 else None
        if first_val and hasattr(first_val, 'year'):
            # It's a date object, convert to datetime
            df['DeliveryDate'] = pd.to_datetime(df['DeliveryDate'])
        else:
            # It's a string, try different formats
            # First try ISO format (YYYY-MM-DD)
            try:
                df['DeliveryDate'] = pd.to_datetime(df['DeliveryDate'], format='%Y-%m-%d')
            except:
                # Try MM/DD/YYYY format
                try:
                    df['DeliveryDate'] = pd.to_datetime(df['DeliveryDate'], format='%m/%d/%Y')
                except:
                    # Fall back to auto-detection
                    df['DeliveryDate'] = pd.to_datetime(df['DeliveryDate'])
    else:
        # Numeric or other type, try to convert
        df['DeliveryDate'] = pd.to_datetime(df['DeliveryDate'])
    
    # Extract hour from HourEnding (format: "HH:00" or "HH:00:00")
    df['hour'] = df['HourEnding'].str.extract(r'(\d+)').astype(int)
    
    # Create datetime (subtract 1 from hour since HourEnding represents end of interval)
    df['datetime'] = df['DeliveryDate'] + pd.to_timedelta(df['hour'] - 1, unit='h')
    
    # Filter for only HB_, LZ_, and DC_ settlement points
    settlement_points = ['HB_BUSAVG', 'HB_HOUSTON', 'HB_HUBAVG', 'HB_NORTH', 
                        'HB_PAN', 'HB_SOUTH', 'HB_WEST',
                        'LZ_AEN', 'LZ_CPS', 'LZ_HOUSTON', 'LZ_LCRA', 
                        'LZ_NORTH', 'LZ_RAYBN', 'LZ_SOUTH', 'LZ_WEST',
                        'DC_E', 'DC_L', 'DC_N', 'DC_R', 'DC_S']
    
    df_filtered = df[df['SettlementPoint'].isin(settlement_points)].copy()
    
    # Pivot the data
    df_pivot = df_filtered.pivot_table(
        index='datetime',
        columns='SettlementPoint',
        values='SettlementPointPrice',
        aggfunc='first'  # Use first value if duplicates exist
    )
    
    # Reset index to make datetime a column
    df_pivot = df_pivot.reset_index()
    
    # Sort by datetime
    df_pivot = df_pivot.sort_values('datetime')
    
    # Ensure all expected columns are present (fill missing with NaN)
    for sp in settlement_points:
        if sp not in df_pivot.columns:
            df_pivot[sp] = np.nan
    
    # Reorder columns: datetime first, then sorted settlement points
    column_order = ['datetime'] + sorted([col for col in df_pivot.columns if col != 'datetime'])
    df_pivot = df_pivot[column_order]
    
    # Convert datetime to Date32 (days since epoch) to match Rust format
    # First ensure it's datetime
    df_pivot['datetime'] = pd.to_datetime(df_pivot['datetime'])
    
    # Create Date32 column (days since Unix epoch)
    epoch = pd.Timestamp('1970-01-01')
    df_pivot['DeliveryDate'] = ((df_pivot['datetime'] - epoch).dt.total_seconds() / 86400).astype('int32')
    
    # Add timezone-aware string representation
    df_pivot['DeliveryDateStr'] = df_pivot['datetime'].dt.strftime('%Y-%m-%dT%H:%M:%S-06:00')
    
    # Keep datetime for compatibility but rename to datetime_ts
    df_pivot = df_pivot.rename(columns={'datetime': 'datetime_ts'})
    
    # Reorder columns: Date columns first
    date_cols = ['DeliveryDate', 'DeliveryDateStr', 'datetime_ts']
    other_cols = [col for col in df_pivot.columns if col not in date_cols]
    df_pivot = df_pivot[date_cols + sorted(other_cols)]
    
    # Save to parquet with proper schema
    table = pa.Table.from_pandas(df_pivot)
    # Update schema to set DeliveryDate as Date32
    new_fields = []
    for i, field in enumerate(table.schema):
        if field.name == 'DeliveryDate':
            new_fields.append(pa.field('DeliveryDate', pa.date32()))
        else:
            new_fields.append(field)
    new_schema = pa.schema(new_fields)
    
    # Cast the table to the new schema
    table = table.cast(new_schema)
    
    # Save to parquet
    output_file = output_dir / f"DA_prices_{year}.parquet"
    pq.write_table(table, output_file)
    print(f"  Saved {len(df_pivot)} hours to {output_file}")
    print(f"  Shape: {df_pivot.shape}")
    print(f"  Date format: Date32 (days since epoch) + string representation")
    
    return df_pivot


def flatten_as_prices(year: int, input_dir: Path, output_dir: Path):
    """Flatten Ancillary Services prices for a given year."""
    input_file = input_dir / f"{year}.parquet"
    if not input_file.exists():
        print(f"Skipping AS {year} - file not found: {input_file}")
        return
    
    print(f"Processing Ancillary Services for {year}...")
    
    # Read the data
    df = pd.read_parquet(input_file)
    
    # Get column names to identify AS types
    print(f"  AS Columns: {df.columns.tolist()}")
    
    # Check if data is empty
    if len(df) == 0:
        print(f"  Warning: No data in AS file for {year}")
        return
    
    # Always recreate datetime from DeliveryDate and HourEnding if they exist
    if 'DeliveryDate' in df.columns and 'HourEnding' in df.columns:
        # Handle DeliveryDate - could be date object or string
        if df['DeliveryDate'].dtype == 'object':
            first_val = df['DeliveryDate'].iloc[0] if len(df) > 0 else None
            if first_val and hasattr(first_val, 'year'):
                df['DeliveryDate'] = pd.to_datetime(df['DeliveryDate'])
            else:
                # Try different date formats
                try:
                    df['DeliveryDate'] = pd.to_datetime(df['DeliveryDate'], format='%Y-%m-%d')
                except:
                    try:
                        df['DeliveryDate'] = pd.to_datetime(df['DeliveryDate'], format='%m/%d/%Y')
                    except:
                        df['DeliveryDate'] = pd.to_datetime(df['DeliveryDate'])
        else:
            df['DeliveryDate'] = pd.to_datetime(df['DeliveryDate'])
        
        df['hour'] = df['HourEnding'].str.extract(r'(\d+)').astype(int)
        df['datetime'] = df['DeliveryDate'] + pd.to_timedelta(df['hour'] - 1, unit='h')
    elif 'datetime' in df.columns and df['datetime'].dtype == 'int64':
        # Convert from timestamp
        df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
    elif 'SCEDTimestamp' in df.columns:
        df['datetime'] = pd.to_datetime(df['SCEDTimestamp'])
    
    # Identify price columns (MCPC = Market Clearing Price for Capacity)
    price_cols = [col for col in df.columns if 'MCPC' in col or 'Price' in col.upper()]
    
    if not price_cols:
        print(f"  Warning: No price columns found in AS data for {year}")
        return
    
    # Create pivot based on available columns
    if 'AncillaryType' in df.columns and len(df['AncillaryType'].unique()) > 0:
        # If we have AncillaryType column, pivot on that
        df_pivot = df.pivot_table(
            index='datetime',
            columns='AncillaryType',
            values=price_cols[0] if price_cols else 'MCPC',
            aggfunc='first'
        )
    else:
        # Otherwise, just select datetime and price columns
        keep_cols = ['datetime'] + price_cols
        df_pivot = df[keep_cols].drop_duplicates(subset=['datetime']).set_index('datetime')
    
    # Check if we got any data
    if len(df_pivot) == 0:
        print(f"  Warning: No data after pivoting for {year}")
        return
    
    # Reset index and sort
    df_pivot = df_pivot.reset_index()
    df_pivot = df_pivot.sort_values('datetime')
    
    # Convert datetime to Date32 format to match Rust
    df_pivot['datetime'] = pd.to_datetime(df_pivot['datetime'])
    epoch = pd.Timestamp('1970-01-01')
    df_pivot['DeliveryDate'] = ((df_pivot['datetime'] - epoch).dt.total_seconds() / 86400).astype('int32')
    df_pivot['DeliveryDateStr'] = df_pivot['datetime'].dt.strftime('%Y-%m-%dT%H:%M:%S-06:00')
    df_pivot = df_pivot.rename(columns={'datetime': 'datetime_ts'})
    
    # Reorder columns
    date_cols = ['DeliveryDate', 'DeliveryDateStr', 'datetime_ts']
    other_cols = [col for col in df_pivot.columns if col not in date_cols]
    df_pivot = df_pivot[date_cols + sorted(other_cols)]
    
    # Save with Date32 schema
    table = pa.Table.from_pandas(df_pivot)
    new_fields = []
    for field in table.schema:
        if field.name == 'DeliveryDate':
            new_fields.append(pa.field('DeliveryDate', pa.date32()))
        else:
            new_fields.append(field)
    table = table.cast(pa.schema(new_fields))
    
    # Save to parquet
    output_file = output_dir / f"AS_prices_{year}.parquet"
    pq.write_table(table, output_file)
    print(f"  Saved {len(df_pivot)} hours to {output_file}")
    print(f"  Shape: {df_pivot.shape}")
    print(f"  Date format: Date32 (days since epoch) + string representation")
    
    return df_pivot


def flatten_rt_prices(year: int, input_dir: Path, output_dir: Path):
    """Flatten Real-Time prices for a given year keeping 15-minute intervals."""
    input_file = input_dir / f"{year}.parquet"
    if not input_file.exists():
        print(f"Skipping RT {year} - file not found: {input_file}")
        return
    
    print(f"Processing Real-Time Prices for {year} (keeping 15-min intervals)...")
    
    # Read the data
    df = pd.read_parquet(input_file)
    
    # Create datetime column if not exists or if it's not datetime type
    if 'datetime' not in df.columns or df['datetime'].dtype != 'datetime64[ns]':
        if 'datetime' in df.columns and df['datetime'].dtype == 'int64':
            # Convert from timestamp (milliseconds)
            df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
        elif 'SCEDTimestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['SCEDTimestamp'])
        elif 'DeliveryDate' in df.columns:
            # Handle DeliveryDate - could be date object or string
            if df['DeliveryDate'].dtype == 'object':
                first_val = df['DeliveryDate'].iloc[0] if len(df) > 0 else None
                if first_val and hasattr(first_val, 'year'):
                    df['datetime'] = pd.to_datetime(df['DeliveryDate'])
                else:
                    # Try different date formats
                    try:
                        df['datetime'] = pd.to_datetime(df['DeliveryDate'], format='%Y-%m-%d')
                    except:
                        try:
                            df['datetime'] = pd.to_datetime(df['DeliveryDate'], format='%m/%d/%Y')
                        except:
                            df['datetime'] = pd.to_datetime(df['DeliveryDate'])
            else:
                df['datetime'] = pd.to_datetime(df['DeliveryDate'])
            
            if 'DeliveryInterval' in df.columns and 'DeliveryHour' in df.columns:
                # Add hour and interval offset for exact 15-minute timestamp
                # Interval 1 = :00, Interval 2 = :15, Interval 3 = :30, Interval 4 = :45
                df['datetime'] = df['datetime'] + pd.to_timedelta((df['DeliveryHour'] - 1) * 60 + (df['DeliveryInterval'] - 1) * 15, unit='min')
    
    # Keep the interval column if it exists
    interval_col = None
    if 'DeliveryInterval' in df.columns:
        interval_col = 'DeliveryInterval'
    elif 'Interval' in df.columns:
        interval_col = 'Interval'
    
    # Identify settlement point column
    sp_col = None
    if 'SettlementPoint' in df.columns:
        sp_col = 'SettlementPoint'
    elif 'SettlementPointName' in df.columns:
        sp_col = 'SettlementPointName'
    
    # Identify price column
    price_col = None
    if 'LMP' in df.columns:
        price_col = 'LMP'
    elif 'SettlementPointPrice' in df.columns:
        price_col = 'SettlementPointPrice'
    else:
        # Look for any column with Price or LMP in name
        price_cols = [col for col in df.columns if 'Price' in col or 'LMP' in col]
        if price_cols:
            price_col = price_cols[0]
    
    if not sp_col or not price_col:
        print(f"  Warning: Could not identify settlement point or price columns")
        print(f"  Columns: {df.columns.tolist()}")
        return
    
    # Filter for only HB_, LZ_, and DC_ settlement points
    settlement_points = ['HB_BUSAVG', 'HB_HOUSTON', 'HB_HUBAVG', 'HB_NORTH', 
                        'HB_PAN', 'HB_SOUTH', 'HB_WEST',
                        'LZ_AEN', 'LZ_CPS', 'LZ_HOUSTON', 'LZ_LCRA', 
                        'LZ_NORTH', 'LZ_RAYBN', 'LZ_SOUTH', 'LZ_WEST',
                        'DC_E', 'DC_L', 'DC_N', 'DC_R', 'DC_S']
    
    df_filtered = df[df[sp_col].isin(settlement_points)].copy()
    
    # Pivot the data keeping 15-minute intervals
    # Use datetime and interval as index
    df_pivot = df_filtered.pivot_table(
        index=['datetime'] + ([interval_col] if interval_col else []),
        columns=sp_col,
        values=price_col,
        aggfunc='first'  # Use first value if duplicates exist
    )
    
    # Reset index to make datetime and interval columns
    df_pivot = df_pivot.reset_index()
    
    # Sort by datetime
    df_pivot = df_pivot.sort_values('datetime')
    
    # Ensure all expected columns are present (fill missing with NaN)
    for sp in settlement_points:
        if sp not in df_pivot.columns:
            df_pivot[sp] = np.nan
    
    # Reorder columns: datetime, interval (if exists), then sorted settlement points
    base_cols = ['datetime']
    if interval_col and interval_col in df_pivot.columns:
        base_cols.append(interval_col)
    column_order = base_cols + sorted([col for col in df_pivot.columns if col not in base_cols])
    df_pivot = df_pivot[column_order]
    
    # Convert datetime to Date32 format to match Rust
    df_pivot['datetime'] = pd.to_datetime(df_pivot['datetime'])
    epoch = pd.Timestamp('1970-01-01')
    # For RT, DeliveryDate is just the date part
    df_pivot['DeliveryDate'] = ((df_pivot['datetime'].dt.normalize() - epoch).dt.total_seconds() / 86400).astype('int32')
    df_pivot['DeliveryDateStr'] = df_pivot['datetime'].dt.strftime('%Y-%m-%dT%H:%M:%S-06:00')
    df_pivot = df_pivot.rename(columns={'datetime': 'datetime_ts'})
    
    # Reorder columns
    date_cols = ['DeliveryDate', 'DeliveryDateStr', 'datetime_ts']
    if interval_col and interval_col in df_pivot.columns:
        date_cols.append(interval_col)
    other_cols = [col for col in df_pivot.columns if col not in date_cols]
    df_pivot = df_pivot[date_cols + sorted(other_cols)]
    
    # Save with Date32 schema
    table = pa.Table.from_pandas(df_pivot)
    new_fields = []
    for field in table.schema:
        if field.name == 'DeliveryDate':
            new_fields.append(pa.field('DeliveryDate', pa.date32()))
        else:
            new_fields.append(field)
    table = table.cast(pa.schema(new_fields))
    
    # Save to parquet
    output_file = output_dir / f"RT_prices_15min_{year}.parquet"
    pq.write_table(table, output_file)
    print(f"  Saved {len(df_pivot)} 15-minute intervals to {output_file}")
    print(f"  Shape: {df_pivot.shape}")
    print(f"  Intervals per day: {len(df_pivot) / 365:.0f} (expected: 96)" if len(df_pivot) > 0 else "")
    print(f"  Date format: Date32 (days since epoch) + string representation")
    
    return df_pivot


def main():
    # Set up directories
    base_dir = Path("/home/enrico/data/ERCOT_data/rollup_files")
    
    # Input directories
    da_input_dir = base_dir / "DA_prices"
    as_input_dir = base_dir / "AS_prices"
    rt_input_dir = base_dir / "RT_prices"
    
    # Output directory - create flattened subdirectory
    output_dir = base_dir / "flattened"
    output_dir.mkdir(exist_ok=True)
    
    # Get available years from DA_prices directory
    da_years = sorted([int(f.stem) for f in da_input_dir.glob("*.parquet")])
    print(f"Found DA price data for years: {da_years}")
    
    # Process DA Energy Prices
    print("\n=== Processing Day Ahead Energy Prices ===")
    for year in da_years:
        flatten_da_prices(year, da_input_dir, output_dir)
    
    # Get available years from AS_prices directory
    as_years = sorted([int(f.stem) for f in as_input_dir.glob("*.parquet")])
    print(f"\nFound AS price data for years: {as_years}")
    
    # Process Ancillary Services
    print("\n=== Processing Ancillary Services ===")
    for year in as_years:
        flatten_as_prices(year, as_input_dir, output_dir)
    
    # Get available years from RT_prices directory
    rt_years = sorted([int(f.stem) for f in rt_input_dir.glob("*.parquet")])
    print(f"\nFound RT price data for years: {rt_years}")
    
    # Process Real-Time Prices
    print("\n=== Processing Real-Time Prices ===")
    for year in rt_years:
        flatten_rt_prices(year, rt_input_dir, output_dir)
    
    print("\n=== Processing Complete ===")
    print(f"All flattened files saved to: {output_dir}")


if __name__ == "__main__":
    main()
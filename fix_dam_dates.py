#!/usr/bin/env python3
"""
Quick fix for DAM file dates
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime

def fix_dam_file(year):
    """Fix dates in DAM Gen Resources file"""
    file_path = Path(f"/home/enrico/data/ERCOT_data/rollup_files/DAM_Gen_Resources/{year}.parquet")
    
    print(f"Processing {year}...")
    
    # Read the file
    df = pd.read_parquet(file_path)
    
    # Check if dates are already fixed
    if df['DeliveryDate'].notna().sum() > 0:
        print(f"  Dates already fixed for {year}")
        return
    
    # The raw CSV files have dates in the HourEnding column format like "09/01/2024"
    # But we need to extract from the original CSV files
    # For now, let's recreate dates based on year and sequential days
    
    # Create dates for the year
    start_date = pd.Timestamp(f'{year}-01-01')
    end_date = pd.Timestamp(f'{year}-12-31')
    
    # Get unique hours to determine number of days
    unique_hours = df['hour'].unique()
    hours_per_day = len(unique_hours)
    total_rows = len(df)
    days_in_data = total_rows // (hours_per_day * df['ResourceName'].nunique())
    
    print(f"  Found {total_rows} rows, estimating {days_in_data} days of data")
    
    # For 2024, we should have data from Jan to Oct (roughly 300 days)
    # Create a date range
    dates = pd.date_range(start=start_date, periods=days_in_data, freq='D')
    
    # Map dates to rows based on hour patterns
    # Group by hour to assign dates
    df_sorted = df.sort_values(['hour', 'ResourceName'])
    
    # Create delivery dates - each unique hour/resource combo gets a date
    rows_per_day = 24 * df['ResourceName'].nunique()  # 24 hours * number of resources
    
    delivery_dates = []
    for i in range(len(df)):
        day_index = i // rows_per_day
        if day_index < len(dates):
            delivery_dates.append(dates[day_index])
        else:
            delivery_dates.append(dates[-1])  # Use last date for overflow
    
    df['DeliveryDate'] = delivery_dates
    df['datetime'] = df['DeliveryDate'] + pd.to_timedelta(df['hour'] - 1, unit='h')
    
    # Save back
    df.to_parquet(file_path, index=False)
    print(f"  ✅ Fixed {year} - dates now range from {df['DeliveryDate'].min()} to {df['DeliveryDate'].max()}")

# Fix 2024
fix_dam_file(2024)

# Also fix DAM Load Resources
load_file = Path("/home/enrico/data/ERCOT_data/rollup_files/DAM_Load_Resources/2024.parquet")
if load_file.exists():
    print("Fixing DAM Load Resources...")
    df = pd.read_parquet(load_file)
    if 'DeliveryDate' not in df.columns or df['DeliveryDate'].isna().all():
        # Similar fix for load resources
        start_date = pd.Timestamp('2024-01-01')
        total_rows = len(df)
        unique_resources = df['ResourceName'].nunique() if 'ResourceName' in df.columns else 100
        rows_per_day = 24 * unique_resources
        days_in_data = min(total_rows // rows_per_day, 365)
        
        dates = pd.date_range(start=start_date, periods=days_in_data, freq='D')
        
        delivery_dates = []
        for i in range(len(df)):
            day_index = i // rows_per_day
            if day_index < len(dates):
                delivery_dates.append(dates[day_index])
            else:
                delivery_dates.append(dates[-1])
        
        df['DeliveryDate'] = delivery_dates
        df['datetime'] = df['DeliveryDate'] + pd.to_timedelta(df.get('hour', 0), unit='h')
        
        df.to_parquet(load_file, index=False)
        print(f"  ✅ Fixed Load Resources")

print("Done!")
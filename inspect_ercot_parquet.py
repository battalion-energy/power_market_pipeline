#!/usr/bin/env python3
import pandas as pd
import pyarrow.parquet as pq
import sys

def inspect_parquet(file_path):
    """Inspect the structure of a Parquet file."""
    # Read schema
    parquet_file = pq.ParquetFile(file_path)
    print(f"Schema for {file_path}:")
    print(parquet_file.schema)
    print("\n" + "="*80 + "\n")
    
    # Read first few rows
    df = pd.read_parquet(file_path)
    print(f"Shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    
    # Check for unique values in key columns
    if 'SettlementPoint' in df.columns:
        print(f"\nUnique Settlement Points: {df['SettlementPoint'].nunique()}")
        print(f"Sample Settlement Points: {df['SettlementPoint'].unique()[:10]}")
    
    if 'SettlementPointName' in df.columns:
        print(f"\nUnique Settlement Point Names: {df['SettlementPointName'].nunique()}")
        # Look for load zones
        load_zones = df[df['SettlementPointName'].str.contains('LZ_', na=False)]['SettlementPointName'].unique()
        print(f"\nLoad Zones found: {load_zones}")
    
    return df

def check_time_intervals(df):
    """Check time intervals in the data."""
    if 'SCEDTimestamp' in df.columns:
        # Convert to datetime
        df['timestamp'] = pd.to_datetime(df['SCEDTimestamp'])
        df_sorted = df.sort_values('timestamp')
        
        # Check intervals for a specific settlement point
        sample_point = df['SettlementPoint'].iloc[0]
        sample_df = df_sorted[df_sorted['SettlementPoint'] == sample_point].head(20)
        print(f"\nTime intervals for {sample_point}:")
        print(sample_df[['timestamp', 'LMP']])
        
        # Calculate typical interval
        time_diffs = sample_df['timestamp'].diff().dropna()
        if len(time_diffs) > 0:
            print(f"\nTypical interval: {time_diffs.mode()[0] if len(time_diffs.mode()) > 0 else time_diffs.iloc[0]}")

if __name__ == "__main__":
    # Inspect different types of files
    print("Checking Real-Time 5-minute data:")
    rt_file = "rt_rust_processor/annual_output/LMPs_by_Resource_Nodes__Load_Zones_and_Trading_Hubs/LMPs_by_Resource_Nodes__Load_Zones_and_Trading_Hubs_2023.parquet"
    try:
        df_rt = inspect_parquet(rt_file)
        check_time_intervals(df_rt)
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "="*80 + "\n")
    print("Checking Day-Ahead Hourly data:")
    dam_file = "rt_rust_processor/annual_output/Settlement_Point_Prices_at_Resource_Nodes__Hubs_and_Load_Zones/Settlement_Point_Prices_at_Resource_Nodes__Hubs_and_Load_Zones_2023.parquet"
    try:
        df_dam = inspect_parquet(dam_file)
    except Exception as e:
        print(f"Error: {e}")
        
    # Check for DAM files
    print("\n" + "="*80 + "\n")
    print("Looking for DAM files:")
    dam_bus_file = "rt_rust_processor/annual_output/DAM_Hourly_LMPs_BusLevel/DAM_Hourly_LMPs_BusLevel_2023.parquet"
    try:
        df_dam_bus = pd.read_parquet(dam_bus_file)
        print(f"DAM Bus Level columns: {list(df_dam_bus.columns)}")
        print(f"Shape: {df_dam_bus.shape}")
        print(f"First 5 rows:\n{df_dam_bus.head()}")
    except FileNotFoundError:
        print(f"DAM Bus Level file not found for 2023")
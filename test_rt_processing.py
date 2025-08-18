#!/usr/bin/env python3
"""
Test script to verify RT price processing maintains 15-minute intervals.
"""

import pandas as pd
from pathlib import Path


def test_rt_processing():
    base_dir = Path("/home/enrico/data/ERCOT_data/rollup_files")
    
    # Test year
    year = 2023
    
    print(f"Testing RT processing for {year}...")
    print("=" * 60)
    
    # 1. Check original RT data
    original_file = base_dir / "RT_prices" / f"{year}.parquet"
    if original_file.exists():
        df_orig = pd.read_parquet(original_file)
        print(f"\n1. Original RT data:")
        print(f"   Shape: {df_orig.shape}")
        print(f"   Columns: {df_orig.columns.tolist()}")
        print(f"   Sample:")
        print(df_orig.head(3))
    
    # 2. Check 15-minute flattened RT
    rt_15min_file = base_dir / "flattened" / f"RT_prices_15min_{year}.parquet"
    if rt_15min_file.exists():
        df_15min = pd.read_parquet(rt_15min_file)
        print(f"\n2. Flattened 15-minute RT data:")
        print(f"   Shape: {df_15min.shape}")
        print(f"   Columns (first 10): {df_15min.columns.tolist()[:10]}")
        print(f"   Expected intervals per day: {4 * 24} (4 per hour * 24 hours)")
        print(f"   Actual intervals in first day:")
        
        # Check intervals for first day
        df_15min['datetime'] = pd.to_datetime(df_15min['datetime'])
        first_day = df_15min['datetime'].dt.date.iloc[0]
        first_day_data = df_15min[df_15min['datetime'].dt.date == first_day]
        print(f"   Date: {first_day}, Count: {len(first_day_data)}")
        
        # Check if DeliveryInterval column exists
        if 'DeliveryInterval' in df_15min.columns:
            print(f"   DeliveryInterval range: {df_15min['DeliveryInterval'].min()} - {df_15min['DeliveryInterval'].max()}")
    
    # 3. Check hourly aggregated RT
    rt_hourly_file = base_dir / "flattened" / f"RT_prices_hourly_{year}.parquet"
    if rt_hourly_file.exists():
        df_hourly = pd.read_parquet(rt_hourly_file)
        print(f"\n3. Hourly aggregated RT data:")
        print(f"   Shape: {df_hourly.shape}")
        print(f"   Expected hours per year: ~8760")
        print(f"   Actual hours: {len(df_hourly)}")
    
    # 4. Check combined 15-minute file
    combined_15min_file = base_dir / "combined" / f"DA_AS_RT_15min_combined_{year}.parquet"
    if combined_15min_file.exists():
        df_combined_15min = pd.read_parquet(combined_15min_file)
        print(f"\n4. Combined 15-minute data (DA+AS+RT):")
        print(f"   Shape: {df_combined_15min.shape}")
        
        # Check column prefixes
        da_cols = [c for c in df_combined_15min.columns if c.startswith('DA_')]
        rt_cols = [c for c in df_combined_15min.columns if c.startswith('RT_')]
        as_cols = [c for c in df_combined_15min.columns if c.startswith('AS_')]
        
        print(f"   DA columns: {len(da_cols)}")
        print(f"   RT columns: {len(rt_cols)}")
        print(f"   AS columns: {len(as_cols)}")
        
        # Verify DA values are repeated for each 15-min interval
        if da_cols:
            sample_da_col = da_cols[0]
            df_combined_15min['datetime'] = pd.to_datetime(df_combined_15min['datetime'])
            df_combined_15min['hour'] = df_combined_15min['datetime'].dt.floor('h')
            
            # Check first hour
            first_hour = df_combined_15min['hour'].iloc[0]
            first_hour_data = df_combined_15min[df_combined_15min['hour'] == first_hour]
            
            print(f"\n   Verification - DA values repeated per hour:")
            print(f"   First hour: {first_hour}")
            print(f"   Intervals in hour: {len(first_hour_data)}")
            print(f"   Unique DA values in hour: {first_hour_data[sample_da_col].nunique()}")
            print(f"   Expected: 1 (same DA value for all 15-min intervals)")
    
    # 5. Check combined hourly file
    combined_hourly_file = base_dir / "combined" / f"DA_AS_RT_combined_{year}.parquet"
    if combined_hourly_file.exists():
        df_combined_hourly = pd.read_parquet(combined_hourly_file)
        print(f"\n5. Combined hourly data (DA+AS+RT):")
        print(f"   Shape: {df_combined_hourly.shape}")
        
        # Check for data alignment
        sample_row = df_combined_hourly.iloc[100]  # Pick a row in the middle
        print(f"\n   Sample row (hour 100):")
        print(f"   Datetime: {sample_row['datetime']}")
        if 'DA_HB_HOUSTON' in sample_row:
            print(f"   DA_HB_HOUSTON: {sample_row['DA_HB_HOUSTON']:.2f}" if pd.notna(sample_row['DA_HB_HOUSTON']) else "   DA_HB_HOUSTON: NaN")
        if 'RT_HB_HOUSTON' in sample_row:
            print(f"   RT_HB_HOUSTON: {sample_row['RT_HB_HOUSTON']:.2f}" if pd.notna(sample_row['RT_HB_HOUSTON']) else "   RT_HB_HOUSTON: NaN")
        if 'AS_REGUP' in sample_row:
            print(f"   AS_REGUP: {sample_row['AS_REGUP']:.2f}" if pd.notna(sample_row['AS_REGUP']) else "   AS_REGUP: NaN")
    
    print("\n" + "=" * 60)
    print("✅ RT processing test complete!")


if __name__ == "__main__":
    test_rt_processing()
#!/usr/bin/env python3
"""
Fix 2025 DAM Gen data date issues
The dates are likely encoded but not being parsed correctly
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

def fix_2025_dates():
    """Fix the date issues in 2025 DAM Gen data"""
    
    data_dir = Path('/home/enrico/data/ERCOT_data')
    file_2025 = data_dir / 'rollup_files/DAM_Gen_Resources/2025.parquet'
    
    print("Fixing 2025 DAM Gen date issues...")
    
    # Read the file
    df = pd.read_parquet(file_2025)
    print(f"Loaded {len(df):,} rows")
    
    # Check if we can infer dates from the data pattern
    # Since this is 2025 data, it should be from Jan 1, 2025 onwards
    
    # Method 1: If HourEnding exists, we can reconstruct dates
    if 'HourEnding' in df.columns:
        print("Attempting to reconstruct dates from data patterns...")
        
        # Group by unique combinations to find the pattern
        unique_hours = df.groupby(['ResourceName', 'HourEnding']).size().reset_index()
        print(f"Unique resource-hour combinations: {len(unique_hours)}")
        
        # Assuming data is ordered chronologically
        # Calculate total days of data
        hours_per_resource = df.groupby('ResourceName')['HourEnding'].count()
        avg_hours = hours_per_resource.mean()
        estimated_days = avg_hours / 24
        print(f"Estimated days of data per resource: {estimated_days:.1f}")
        
        # Create date sequence from Jan 1, 2025
        start_date = datetime(2025, 1, 1)
        
        # For each resource, assign dates sequentially
        df_fixed = []
        
        for resource in df['ResourceName'].unique()[:10]:  # Test with first 10
            resource_data = df[df['ResourceName'] == resource].copy()
            
            # Sort by hour to ensure chronological order
            resource_data = resource_data.sort_values('HourEnding')
            
            # Create date sequence
            num_records = len(resource_data)
            dates = []
            current_date = start_date
            
            for idx, row in resource_data.iterrows():
                hour = row['HourEnding']
                
                # Parse hour (format: "01:00" or just "1")
                if isinstance(hour, str) and ':' in hour:
                    hour_num = int(hour.split(':')[0])
                else:
                    try:
                        hour_num = int(hour)
                    except:
                        hour_num = 1
                
                # Set the datetime (hour 24 means end of day, so it's actually hour 0 of next day)
                if hour_num == 24:
                    dt = current_date.replace(hour=23)  # Use 23:00 for hour 24
                    current_date += timedelta(days=1)  # Move to next day
                else:
                    dt = current_date.replace(hour=hour_num-1 if hour_num > 0 else 0)
                
                dates.append(dt)
            
            resource_data['datetime_fixed'] = dates
            df_fixed.append(resource_data)
        
        if df_fixed:
            df_sample = pd.concat(df_fixed)
            print(f"\nFixed {len(df_sample)} records for sample resources")
            print(f"Date range: {df_sample['datetime_fixed'].min()} to {df_sample['datetime_fixed'].max()}")
            
            # Save sample
            output_file = data_dir / 'bess_analysis/dam_gen_2025_sample_fixed.parquet'
            df_sample.to_parquet(output_file)
            print(f"Saved sample to {output_file}")
            
            return df_sample
    
    return None

def analyze_2025_pattern():
    """Analyze the pattern in 2025 data to understand the structure"""
    
    data_dir = Path('/home/enrico/data/ERCOT_data')
    file_2025 = data_dir / 'rollup_files/DAM_Gen_Resources/2025.parquet'
    
    df = pd.read_parquet(file_2025)
    
    # Look for any column that might contain date information
    print("\nAnalyzing 2025 data structure:")
    print("="*60)
    
    for col in df.columns:
        if df[col].dtype in ['int32', 'int64', 'float64']:
            non_null = df[col].dropna()
            if len(non_null) > 0:
                min_val = non_null.min()
                max_val = non_null.max()
                
                # Check if it could be a date (days since epoch)
                if 15000 < min_val < 25000:  # Rough range for dates around 2025
                    print(f"\n{col} might be a date column:")
                    print(f"  Range: {min_val} to {max_val}")
                    
                    # Try converting to date
                    try:
                        test_date = pd.to_datetime(min_val, unit='D', origin='1970-01-01')
                        print(f"  As date: {test_date}")
                    except:
                        pass
    
    # Check if data is simply ordered by date
    print("\nChecking if data is chronologically ordered...")
    
    # Get a sample resource
    sample_resource = df[df['ResourceType'] == 'PWRSTR']['ResourceName'].iloc[0] if any(df['ResourceType'] == 'PWRSTR') else df['ResourceName'].iloc[0]
    sample_data = df[df['ResourceName'] == sample_resource].head(100)
    
    print(f"Sample resource: {sample_resource}")
    print(f"First 5 HourEnding values: {sample_data['HourEnding'].head().tolist()}")
    
    # Count unique days based on 24-hour cycles
    hour_sequence = sample_data['HourEnding'].tolist()
    days = len([h for h in hour_sequence if h == '24:00' or h == 24 or h == '24'])
    print(f"Estimated days in sample: {days}")

if __name__ == '__main__':
    # First analyze the pattern
    analyze_2025_pattern()
    
    # Then try to fix
    fixed_df = fix_2025_dates()
    
    if fixed_df is not None:
        print("\n✅ Successfully created fixed sample of 2025 data")
    else:
        print("\n❌ Could not fix 2025 dates - may need to reprocess from source")
#!/usr/bin/env python3
"""Check the schema of DAM disclosure files"""

import pandas as pd
import glob
import sys

def check_dam_file_schema():
    # Find a sample DAM file
    dam_files = glob.glob("/Users/enrico/data/ERCOT_data/60-Day_DAM_Disclosure_Reports/csv/*DAM_Gen_Resource_Data*.csv")
    
    if not dam_files:
        print("❌ No DAM files found")
        return
    
    # Take the first file
    sample_file = dam_files[0]
    print(f"📋 Checking schema of: {sample_file}")
    
    # Read first few rows to check schema
    df = pd.read_csv(sample_file, nrows=10)
    
    print(f"\nColumns ({len(df.columns)}):")
    for i, col in enumerate(df.columns):
        dtype = df[col].dtype
        sample_value = df[col].iloc[0] if len(df) > 0 else "N/A"
        print(f"  {i+1:3d}. {col:30s} {str(dtype):15s} Sample: {sample_value}")
    
    # Check for problematic column 'LSL' at position 30
    if len(df.columns) > 30:
        col_30 = df.columns[29]  # 0-indexed
        print(f"\n⚠️  Column 30 is: '{col_30}'")
        print(f"   Values: {df[col_30].head().tolist()}")
    
    # Also check PWRSTR rows
    print("\n🔋 Checking PWRSTR (BESS) rows:")
    if 'Resource Type' in df.columns:
        bess_df = df[df['Resource Type'] == 'PWRSTR']
        if len(bess_df) > 0:
            print(f"Found {len(bess_df)} PWRSTR rows")
            print("\nSample PWRSTR row:")
            print(bess_df.iloc[0].to_dict())
        else:
            print("No PWRSTR rows in first 10 rows")
            
            # Read more rows to find PWRSTR
            df_large = pd.read_csv(sample_file, nrows=1000)
            bess_df_large = df_large[df_large['Resource Type'] == 'PWRSTR']
            print(f"\nIn first 1000 rows, found {len(bess_df_large)} PWRSTR rows")

if __name__ == "__main__":
    check_dam_file_schema()
#!/usr/bin/env python3
"""Test different approaches to reading the problematic CSV"""

import pandas as pd
import glob

def test_csv_reading():
    # Find the problematic file
    dam_files = glob.glob("/Users/enrico/data/ERCOT_data/60-Day_DAM_Disclosure_Reports/csv/*DAM_Gen_Resource_Data*11.csv")
    
    if not dam_files:
        print("No 2011 DAM files found")
        return
    
    test_file = dam_files[0]
    print(f"Testing with: {test_file}")
    
    # Method 1: Let pandas infer types
    print("\nMethod 1: Default pandas reading")
    try:
        df1 = pd.read_csv(test_file, nrows=10)
        print(f"✅ Success! LSL column type: {df1['LSL'].dtype}")
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Method 2: Force all as string
    print("\nMethod 2: Force all columns as string")
    try:
        df2 = pd.read_csv(test_file, dtype=str, nrows=10)
        print(f"✅ Success! LSL column type: {df2['LSL'].dtype}")
        print(f"   LSL values: {df2['LSL'].unique()}")
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Method 3: Specify LSL as float
    print("\nMethod 3: Specify LSL as float")
    try:
        df3 = pd.read_csv(test_file, dtype={'LSL': float}, nrows=10)
        print(f"✅ Success! LSL column type: {df3['LSL'].dtype}")
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Find the problematic row
    print("\nSearching for the problematic value '84.6'...")
    df_full = pd.read_csv(test_file, dtype=str)
    problematic_rows = df_full[df_full['LSL'] == '84.6']
    print(f"Found {len(problematic_rows)} rows with LSL='84.6'")
    if len(problematic_rows) > 0:
        print(f"First occurrence at row {problematic_rows.index[0]}")
        print(f"Resource: {problematic_rows.iloc[0]['Resource Name']}")

if __name__ == "__main__":
    test_csv_reading()
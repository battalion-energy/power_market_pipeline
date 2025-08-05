#!/usr/bin/env python3
"""
Test script to check BESS data availability and structure
"""

import pandas as pd
import os
import glob
from datetime import datetime
from pathlib import Path

# Base directories
DISCLOSURE_DIR = "/Users/enrico/data/ERCOT_data"
PRICE_DATA_DIR = "/Users/enrico/data/ERCOT_data"

def check_bess_historical_data():
    """Check what BESS data is available historically"""
    
    print("Checking BESS data availability...\n")
    
    # 1. Check DAM disclosure files
    print("1. DAM Disclosure Files:")
    dam_pattern = f"{DISCLOSURE_DIR}/60-Day_DAM_Disclosure_Reports/csv/60d_DAM_Gen_Resource_Data-*.csv"
    dam_files = sorted(glob.glob(dam_pattern))
    
    if dam_files:
        print(f"   Found {len(dam_files)} DAM files")
        print(f"   Earliest: {Path(dam_files[0]).name}")
        print(f"   Latest: {Path(dam_files[-1]).name}")
        
        # Sample earliest file to find BESS
        print("\n   Checking earliest file for BESS resources...")
        try:
            df = pd.read_csv(dam_files[0], nrows=1000)
            print(f"   Columns: {', '.join(df.columns[:10])}...")
            
            if 'Resource Type' in df.columns:
                resource_types = df['Resource Type'].unique()
                print(f"   Resource Types found: {resource_types[:10]}")
                
                # Look for PWRSTR
                pwrstr_count = len(df[df['Resource Type'] == 'PWRSTR'])
                print(f"   PWRSTR resources in sample: {pwrstr_count}")
                
                # Also check for other potential BESS identifiers
                if pwrstr_count == 0:
                    # Check by name
                    bess_pattern = df['Resource Name'].str.contains('BESS|ESS|BATTERY', case=False, na=False)
                    bess_by_name = len(df[bess_pattern])
                    print(f"   BESS by name pattern: {bess_by_name}")
                    
        except Exception as e:
            print(f"   Error reading file: {e}")
    else:
        print("   No DAM files found!")
    
    # 2. Check SCED disclosure files
    print("\n2. SCED Disclosure Files:")
    sced_pattern = f"{DISCLOSURE_DIR}/60-Day_SCED_Disclosure_Reports/csv/60d_SCED_Gen_Resource_Data-*.csv"
    sced_files = sorted(glob.glob(sced_pattern))
    
    if sced_files:
        print(f"   Found {len(sced_files)} SCED files")
        print(f"   Earliest: {Path(sced_files[0]).name}")
        print(f"   Latest: {Path(sced_files[-1]).name}")
    else:
        print("   No SCED files found!")
    
    # 3. Check RT price files
    print("\n3. Real-Time Price Files:")
    rt_pattern = f"{PRICE_DATA_DIR}/Settlement_Point_Prices_at_Resource_Nodes/csv/*.csv"
    rt_files = glob.glob(rt_pattern)
    
    if rt_files:
        print(f"   Found {len(rt_files)} RT price files")
        # Sample file to check structure
        sample_file = rt_files[0]
        try:
            df = pd.read_csv(sample_file, nrows=5)
            print(f"   Sample columns: {', '.join(df.columns)}")
        except:
            pass
    else:
        print("   No RT price files found!")
    
    # 4. Check AS price files  
    print("\n4. AS Clearing Price Files:")
    as_pattern = f"{PRICE_DATA_DIR}/DAM_Clearing_Prices_for_Capacity/csv/*.csv"
    as_files = glob.glob(as_pattern)
    
    if as_files:
        print(f"   Found {len(as_files)} AS price files")
    else:
        print("   No AS price files found!")
    
    # 5. Try to find first BESS appearance
    print("\n5. Searching for first BESS appearance...")
    first_bess_date = None
    bess_count_by_year = {}
    
    for dam_file in dam_files[:50]:  # Check first 50 files
        try:
            date_str = Path(dam_file).stem.split('-')[-1]
            file_date = datetime.strptime(date_str, "%d-%b-%y")
            year = file_date.year
            
            df = pd.read_csv(dam_file, nrows=5000)
            
            if 'Resource Type' in df.columns:
                pwrstr_resources = df[df['Resource Type'] == 'PWRSTR']['Resource Name'].unique()
                
                if len(pwrstr_resources) > 0:
                    if first_bess_date is None or file_date < first_bess_date:
                        first_bess_date = file_date
                    
                    if year not in bess_count_by_year:
                        bess_count_by_year[year] = set()
                    bess_count_by_year[year].update(pwrstr_resources)
                    
        except Exception as e:
            continue
    
    if first_bess_date:
        print(f"   First BESS found: {first_bess_date.strftime('%Y-%m-%d')}")
        print("\n   BESS count by year:")
        for year in sorted(bess_count_by_year.keys()):
            print(f"   {year}: {len(bess_count_by_year[year])} unique BESS")


if __name__ == "__main__":
    check_bess_historical_data()
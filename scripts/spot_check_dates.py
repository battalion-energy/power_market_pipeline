#!/usr/bin/env python3
"""
Spot check date formats across all ERCOT data file types.
"""

import pandas as pd
import glob
import os
from pathlib import Path
import random

def check_file_dates(file_path, file_type, sample_size=3):
    """Check date formats in a specific file."""
    print(f"\n{'='*80}")
    print(f"FILE: {os.path.basename(file_path)}")
    print(f"TYPE: {file_type}")
    print(f"{'='*80}")
    
    try:
        # Read first 100 rows to check structure
        df = pd.read_csv(file_path, nrows=100)
        
        print(f"Columns: {df.columns.tolist()}")
        print(f"Shape: {df.shape}")
        
        # Identify date-related columns
        date_cols = []
        for col in df.columns:
            col_lower = col.lower()
            if any(term in col_lower for term in ['date', 'timestamp', 'time', 'hour', 'interval']):
                date_cols.append(col)
        
        print(f"\nDate-related columns found: {date_cols}")
        
        # Show samples for each date column
        for col in date_cols:
            print(f"\n{col}:")
            print(f"  Data type: {df[col].dtype}")
            # Get unique values if there are few, otherwise sample
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) <= 10:
                print(f"  Unique values: {unique_vals[:sample_size].tolist()}")
            else:
                print(f"  Sample values: {df[col].dropna().head(sample_size).tolist()}")
            
            # Try to infer date format
            if df[col].dtype == 'object':
                sample_val = str(df[col].dropna().iloc[0]) if len(df[col].dropna()) > 0 else None
                if sample_val:
                    if '/' in sample_val and ':' in sample_val:
                        print(f"  Likely format: MM/DD/YYYY HH:MM:SS")
                    elif '/' in sample_val:
                        parts = sample_val.split('/')
                        if len(parts) == 3:
                            if len(parts[2]) == 4:
                                print(f"  Likely format: MM/DD/YYYY")
                            else:
                                print(f"  Likely format: MM/DD/YY")
                    elif '-' in sample_val:
                        parts = sample_val.split('-')
                        if len(parts) == 3:
                            if len(parts[0]) == 4:
                                print(f"  Likely format: YYYY-MM-DD")
                            else:
                                print(f"  Likely format: MM-DD-YYYY or DD-MM-YYYY")
                    elif ':' in sample_val:
                        print(f"  Likely format: HH:MM or HH:MM:SS")
        
        return date_cols
        
    except Exception as e:
        print(f"Error reading file: {e}")
        return []

def main():
    base_dir = Path("/home/enrico/data/ERCOT_data")
    
    # Define file patterns for each data type
    file_patterns = {
        "DAM_Settlement_Point_Prices": {
            "path": base_dir / "DAM_Settlement_Point_Prices/csv",
            "pattern": "*.csv",
            "expected_dates": ["DeliveryDate", "HourEnding"]
        },
        "DAM_Clearing_Prices_for_Capacity": {
            "path": base_dir / "DAM_Clearing_Prices_for_Capacity/csv", 
            "pattern": "*.csv",
            "expected_dates": ["DeliveryDate", "HourEnding"]
        },
        "Settlement_Point_Prices_RT": {
            "path": base_dir / "Settlement_Point_Prices_at_Resource_Nodes,_Hubs_and_Load_Zones/csv",
            "pattern": "*.csv",
            "expected_dates": ["DeliveryDate", "DeliveryHour", "DeliveryInterval"]
        },
        "60-Day_DAM_Disclosure": {
            "path": base_dir / "60-Day_DAM_Disclosure_Reports",
            "pattern": "*DAM_Gen*.csv",
            "expected_dates": ["OperatingDate", "HourEnding"]
        },
        "60-Day_SCED_Disclosure": {
            "path": base_dir / "60-Day_SCED_Disclosure_Reports",
            "pattern": "*SCED_Gen*.csv",
            "expected_dates": ["SCEDTimestamp"]
        },
        "60-Day_COP_Snapshot": {
            "path": base_dir / "60-Day_COP_Adjustment_Period_Snapshot",
            "pattern": "*COP*.csv",
            "expected_dates": ["COPAdjustmentPeriodStartDate", "COPHourEnding"]
        }
    }
    
    all_results = {}
    
    for data_type, config in file_patterns.items():
        print(f"\n{'#'*80}")
        print(f"# {data_type}")
        print(f"{'#'*80}")
        
        # Find files
        pattern_path = config["path"] / config["pattern"]
        files = glob.glob(str(pattern_path))
        
        if not files:
            print(f"No files found matching: {pattern_path}")
            continue
            
        print(f"Found {len(files)} files")
        
        # Sample 3 files
        sample_files = random.sample(files, min(3, len(files)))
        
        results = []
        for file_path in sample_files:
            date_cols = check_file_dates(file_path, data_type)
            results.append({
                "file": os.path.basename(file_path),
                "date_columns": date_cols
            })
        
        all_results[data_type] = {
            "expected": config["expected_dates"],
            "found": results
        }
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY OF DATE FORMATS")
    print(f"{'='*80}")
    
    for data_type, info in all_results.items():
        print(f"\n{data_type}:")
        print(f"  Expected columns: {info['expected']}")
        for result in info['found']:
            print(f"  Found in {result['file']}: {result['date_columns']}")

if __name__ == "__main__":
    main()
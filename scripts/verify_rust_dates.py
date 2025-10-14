#!/usr/bin/env python3
"""
Verify that the Rust processor correctly parses dates from rollup files.
"""

import pandas as pd
from pathlib import Path

def verify_rollup_dates(data_type, year):
    """Verify dates in a rollup file."""
    rollup_dir = Path("/home/enrico/data/ERCOT_data/rollup_files")
    file_path = rollup_dir / data_type / f"{year}.parquet"
    
    if not file_path.exists():
        print(f"  File not found: {file_path}")
        return False
        
    df = pd.read_parquet(file_path)
    
    print(f"\n{data_type}/{year}.parquet:")
    print(f"  Total rows: {len(df):,}")
    
    # Check date columns
    date_cols = []
    for col in df.columns:
        col_lower = col.lower()
        if any(term in col_lower for term in ['date', 'timestamp', 'time']):
            date_cols.append(col)
    
    for col in date_cols:
        print(f"  {col}:")
        print(f"    Type: {df[col].dtype}")
        print(f"    Unique values: {df[col].nunique()}")
        print(f"    Sample: {df[col].dropna().head(3).tolist()}")
        
        # Check if dates are properly formatted
        if df[col].dtype == 'object':
            # String dates - check format
            sample = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
            if sample and '/' in str(sample):
                print(f"    Format: MM/DD/YYYY (string)")
                # Verify all dates are valid
                try:
                    parsed = pd.to_datetime(df[col].dropna().head(100), format='%m/%d/%Y')
                    print(f"    ✓ Date parsing successful")
                except Exception as e:
                    print(f"    ✗ Date parsing failed: {e}")
                    return False
        elif 'datetime64' in str(df[col].dtype):
            print(f"    Format: datetime64 (parsed)")
            # Check date range
            print(f"    Range: {df[col].min()} to {df[col].max()}")
    
    # Special checks for specific data types
    if data_type == "DA_prices":
        if "DeliveryDate" in df.columns:
            unique_dates = df["DeliveryDate"].nunique()
            expected_days = 365 if year % 4 != 0 or (year % 100 == 0 and year % 400 != 0) else 366
            if year == 2025:
                expected_days = min(expected_days, 230)  # Partial year
            print(f"  Date coverage: {unique_dates} unique days (expected ~{expected_days})")
            if unique_dates < expected_days - 10:
                print(f"    ⚠️ WARNING: Missing {expected_days - unique_dates} days")
    
    return True

def main():
    print("=" * 80)
    print("RUST PROCESSOR DATE VERIFICATION")
    print("=" * 80)
    
    # Test different data types and years
    test_cases = [
        ("DA_prices", 2023),
        ("DA_prices", 2024),
        ("AS_prices", 2023),
        ("AS_prices", 2024),
        ("RT_prices", 2023),
    ]
    
    all_good = True
    for data_type, year in test_cases:
        if not verify_rollup_dates(data_type, year):
            all_good = False
    
    # Check flattened files
    print("\n" + "=" * 80)
    print("FLATTENED FILES DATE VERIFICATION")
    print("=" * 80)
    
    flattened_dir = Path("/home/enrico/data/ERCOT_data/rollup_files/flattened")
    
    for year in [2023, 2024]:
        for prefix in ["DA_prices", "AS_prices", "RT_prices_15min"]:
            file_path = flattened_dir / f"{prefix}_{year}.parquet"
            if file_path.exists():
                df = pd.read_parquet(file_path)
                print(f"\n{prefix}_{year}.parquet:")
                print(f"  Rows: {len(df):,}")
                
                if "datetime" in df.columns:
                    print(f"  datetime column:")
                    print(f"    Type: {df['datetime'].dtype}")
                    print(f"    Range: {df['datetime'].min()} to {df['datetime'].max()}")
                    
                    # Check for gaps
                    if prefix == "DA_prices":
                        expected_hours = 8760 if year % 4 != 0 else 8784
                        expected_hours -= 1  # DST spring forward
                        actual_hours = len(df)
                        print(f"    Coverage: {actual_hours} hours (expected ~{expected_hours})")
                        if abs(actual_hours - expected_hours) > 2:
                            print(f"    ⚠️ WARNING: Hour count mismatch")
                            all_good = False
                    elif prefix == "RT_prices_15min":
                        expected_intervals = (8760 if year % 4 != 0 else 8784) * 4
                        expected_intervals -= 4  # DST spring forward
                        actual_intervals = len(df)
                        print(f"    Coverage: {actual_intervals} intervals (expected ~{expected_intervals})")
    
    print("\n" + "=" * 80)
    if all_good:
        print("✅ ALL DATE VERIFICATIONS PASSED")
    else:
        print("⚠️ SOME DATE ISSUES DETECTED")
    print("=" * 80)

if __name__ == "__main__":
    main()
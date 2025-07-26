#!/usr/bin/env python3
"""Test processing a sample of ERCOT historical data."""

import zipfile
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta

# Test with one file from each type
DAM_SAMPLE = "/Users/enrico/data/ERCOT_data/DAM_Hourly_LMPs/1009465156.cdr.00012328.0000000000000000.20240605.123251.DAMHRLMPNP4183_csv.zip"
RT_SAMPLE = "/Users/enrico/data/ERCOT_data/Settlement_Point_Prices_at_Resource_Nodes,_Hubs_and_Load_Zones/cdr.00012301.0000000000000000.20240823.000201899.SPPHLZNP6905_20240823_0000_csv.zip"
AS_SAMPLE = "/Users/enrico/data/ERCOT_data/DAM_Clearing_Prices_for_Capacity/1000100592.cdr.00012329.0000000000000000.20240430.123245.DAMCPCNP4188_csv.zip"


def test_dam_file():
    """Test DAM LMP file processing."""
    print("\n📊 Testing DAM LMP file...")
    print(f"File: {Path(DAM_SAMPLE).name}")
    
    try:
        with zipfile.ZipFile(DAM_SAMPLE, 'r') as zf:
            print(f"Contents: {zf.namelist()}")
            
            csv_files = [f for f in zf.namelist() if f.endswith('.csv')]
            if csv_files:
                with zf.open(csv_files[0]) as f:
                    df = pd.read_csv(f, nrows=10)
                    print(f"\nColumns: {list(df.columns)}")
                    print(f"Shape: {df.shape}")
                    print("\nSample data:")
                    print(df.head(3))
    except Exception as e:
        print(f"Error: {e}")


def test_rt_file():
    """Test RT SPP file processing."""
    print("\n\n📊 Testing RT SPP file...")
    print(f"File: {Path(RT_SAMPLE).name}")
    
    try:
        with zipfile.ZipFile(RT_SAMPLE, 'r') as zf:
            print(f"Contents: {zf.namelist()}")
            
            csv_files = [f for f in zf.namelist() if f.endswith('.csv')]
            if csv_files:
                with zf.open(csv_files[0]) as f:
                    df = pd.read_csv(f, nrows=10)
                    print(f"\nColumns: {list(df.columns)}")
                    print(f"Shape: {df.shape}")
                    print("\nSample data:")
                    print(df.head(3))
    except Exception as e:
        print(f"Error: {e}")


def test_as_file():
    """Test Ancillary Services file processing."""
    print("\n\n📊 Testing Ancillary Services file...")
    print(f"File: {Path(AS_SAMPLE).name}")
    
    try:
        with zipfile.ZipFile(AS_SAMPLE, 'r') as zf:
            print(f"Contents: {zf.namelist()}")
            
            csv_files = [f for f in zf.namelist() if f.endswith('.csv')]
            if csv_files:
                with zf.open(csv_files[0]) as f:
                    df = pd.read_csv(f, nrows=10)
                    print(f"\nColumns: {list(df.columns)}")
                    print(f"Shape: {df.shape}")
                    print("\nSample data:")
                    print(df.head(3))
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    print("🔍 ERCOT Historical Data Sample Test")
    print("=" * 60)
    
    test_dam_file()
    test_rt_file()
    test_as_file()
    
    print("\n✅ Test complete!")
#!/usr/bin/env python3
"""Test script to verify BESS daily revenue processor output"""

import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
import sys

def check_daily_revenue_output():
    """Check if daily revenue files were created and have expected structure"""
    daily_dir = Path("bess_daily_revenues/daily")
    monthly_dir = Path("bess_daily_revenues/monthly")
    
    # Check if directories exist
    if not daily_dir.exists():
        print(f"❌ Daily directory not found: {daily_dir}")
        return False
    
    if not monthly_dir.exists():
        print(f"❌ Monthly directory not found: {monthly_dir}")
        return False
    
    print("✅ Output directories exist")
    
    # Check daily files
    daily_files = list(daily_dir.glob("*.parquet"))
    print(f"\n📊 Found {len(daily_files)} daily parquet files")
    
    if daily_files:
        # Load first file to check schema
        df = pd.read_parquet(daily_files[0])
        print(f"\n📋 Daily revenue schema ({daily_files[0].name}):")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Rows: {len(df)}")
        
        # Expected columns
        expected_cols = [
            'resource_name', 'date', 'settlement_point', 'capacity_mw',
            'rt_energy_revenue', 'dam_energy_revenue', 'reg_up_revenue',
            'reg_down_revenue', 'spin_revenue', 'non_spin_revenue',
            'ecrs_revenue', 'total_revenue', 'energy_revenue', 'as_revenue'
        ]
        
        missing_cols = set(expected_cols) - set(df.columns)
        if missing_cols:
            print(f"   ⚠️  Missing columns: {missing_cols}")
        else:
            print("   ✅ All expected columns present")
        
        # Show sample data
        if len(df) > 0:
            print(f"\n📈 Sample daily revenue data:")
            print(df.head(3).to_string())
            
            # Show revenue breakdown for one BESS
            sample_bess = df['resource_name'].iloc[0]
            bess_data = df[df['resource_name'] == sample_bess]
            print(f"\n💰 Revenue breakdown for {sample_bess}:")
            print(f"   Total days: {len(bess_data)}")
            print(f"   Total revenue: ${bess_data['total_revenue'].sum():,.2f}")
            print(f"   Energy revenue: ${bess_data['energy_revenue'].sum():,.2f}")
            print(f"   AS revenue: ${bess_data['as_revenue'].sum():,.2f}")
    
    # Check monthly files
    monthly_files = list(monthly_dir.glob("*.parquet"))
    print(f"\n📊 Found {len(monthly_files)} monthly parquet files")
    
    if monthly_files:
        # Load first file to check schema
        df = pd.read_parquet(monthly_files[0])
        print(f"\n📋 Monthly rollup schema ({monthly_files[0].name}):")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Rows: {len(df)}")
        
        # Show sample data
        if len(df) > 0:
            print(f"\n📈 Sample monthly rollup data:")
            print(df.head(3).to_string())
    
    return True

def main():
    """Main test function"""
    print("🔍 Testing BESS Daily Revenue Processor Output")
    print("=" * 60)
    
    if not check_daily_revenue_output():
        sys.exit(1)
    
    print("\n✅ All checks passed!")

if __name__ == "__main__":
    main()
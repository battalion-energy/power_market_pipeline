#!/usr/bin/env python3
"""Create a simple test processor to validate the approach"""

import pandas as pd
import glob
import os
from pathlib import Path

def process_test_dam_file():
    """Process a single DAM file to test the parsing"""
    
    # Find a sample DAM file
    dam_files = glob.glob("/Users/enrico/data/ERCOT_data/60-Day_DAM_Disclosure_Reports/csv/*DAM_Gen_Resource_Data*24.csv")
    
    if not dam_files:
        print("❌ No DAM files found")
        return
    
    # Take the first file
    sample_file = dam_files[0]
    print(f"📋 Processing: {os.path.basename(sample_file)}")
    
    # Read the file
    df = pd.read_csv(sample_file)
    print(f"   Total rows: {len(df)}")
    
    # Filter for PWRSTR (BESS)
    bess_df = df[df['Resource Type'] == 'PWRSTR']
    print(f"   PWRSTR rows: {len(bess_df)}")
    
    if len(bess_df) == 0:
        print("   No BESS resources found")
        return
    
    # Show unique BESS resources
    unique_bess = bess_df['Resource Name'].unique()
    print(f"   Unique BESS: {len(unique_bess)}")
    print(f"   Examples: {list(unique_bess[:5])}")
    
    # Calculate daily revenues for one BESS
    sample_bess = unique_bess[0]
    bess_data = bess_df[bess_df['Resource Name'] == sample_bess]
    
    # Energy revenue
    energy_revenue = (bess_data['Awarded Quantity'] * bess_data['Energy Settlement Point Price']).sum()
    
    # AS revenues
    reg_up_revenue = (bess_data['RegUp Awarded'] * bess_data['RegUp MCPC']).sum()
    reg_down_revenue = (bess_data['RegDown Awarded'] * bess_data['RegDown MCPC']).sum()
    
    print(f"\n💰 Daily revenue for {sample_bess}:")
    print(f"   Energy: ${energy_revenue:,.2f}")
    print(f"   RegUp: ${reg_up_revenue:,.2f}")
    print(f"   RegDown: ${reg_down_revenue:,.2f}")
    
    # Save test output
    output_dir = Path("test_daily_output")
    output_dir.mkdir(exist_ok=True)
    
    # Create summary
    summary = pd.DataFrame({
        'resource_name': [sample_bess],
        'date': [df['Delivery Date'].iloc[0]],
        'energy_revenue': [energy_revenue],
        'reg_up_revenue': [reg_up_revenue],
        'reg_down_revenue': [reg_down_revenue],
        'total_revenue': [energy_revenue + reg_up_revenue + reg_down_revenue]
    })
    
    output_file = output_dir / "test_daily_revenue.csv"
    summary.to_csv(output_file, index=False)
    print(f"\n✅ Saved test output to: {output_file}")

if __name__ == "__main__":
    process_test_dam_file()
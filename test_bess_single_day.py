#!/usr/bin/env python3
"""
Test BESS revenue calculation for a single day
Verifies data availability and calculations
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import glob
from pathlib import Path

# Base data directory
BASE_DIR = "/Users/enrico/data/ERCOT_data"

def test_single_day(test_date: datetime):
    """Test revenue calculation for a single day"""
    
    print(f"\nTesting BESS revenue calculation for {test_date.strftime('%Y-%m-%d')}")
    print("="*80)
    
    # 1. Check disclosure data availability
    print("\n1. Checking 60-Day Disclosure Data:")
    
    date_str = test_date.strftime("%d-%b-%y").upper()
    
    # DAM disclosure
    dam_file = f"{BASE_DIR}/60-Day_DAM_Disclosure_Reports/csv/60d_DAM_Gen_Resource_Data-{date_str}.csv"
    if os.path.exists(dam_file):
        print(f"   ✓ DAM file found: {Path(dam_file).name}")
        dam_df = pd.read_csv(dam_file, nrows=100)
        bess_count = len(dam_df[dam_df['Resource Type'] == 'PWRSTR'])
        print(f"     PWRSTR resources in sample: {bess_count}")
    else:
        print(f"   ✗ DAM file not found: {date_str}")
        
    # SCED disclosure
    sced_file = f"{BASE_DIR}/60-Day_SCED_Disclosure_Reports/csv/60d_SCED_Gen_Resource_Data-{date_str}.csv"
    if os.path.exists(sced_file):
        print(f"   ✓ SCED file found: {Path(sced_file).name}")
    else:
        print(f"   ✗ SCED file not found: {date_str}")
    
    # 2. Check price data availability
    print("\n2. Checking Price Data:")
    
    price_date_str = test_date.strftime("%Y%m%d")
    
    # DAM SPP
    dam_spp_pattern = f"{BASE_DIR}/DAM_Settlement_Point_Prices/csv/cdr.*.{price_date_str}.*.DAMSPNP4190.csv"
    dam_spp_files = glob.glob(dam_spp_pattern)
    if dam_spp_files:
        print(f"   ✓ DAM SPP file found: {Path(dam_spp_files[0]).name}")
        df = pd.read_csv(dam_spp_files[0], nrows=5)
        print(f"     Columns: {', '.join(df.columns)}")
    else:
        print(f"   ✗ DAM SPP file not found for {price_date_str}")
    
    # RT SPP
    rt_spp_pattern = f"{BASE_DIR}/Settlement_Point_Prices_at_Resource_Nodes,_Hubs_and_Load_Zones/csv/cdr.*.{price_date_str}.*.SPPHLZNP6905_*.csv"
    rt_spp_files = glob.glob(rt_spp_pattern)
    print(f"   RT SPP files found: {len(rt_spp_files)} files")
    if rt_spp_files:
        print(f"     Sample: {Path(rt_spp_files[0]).name}")
        df = pd.read_csv(rt_spp_files[0], nrows=5)
        print(f"     Columns: {', '.join(df.columns)}")
    
    # AS MCPC
    as_mcpc_pattern = f"{BASE_DIR}/DAM_Clearing_Prices_for_Capacity/csv/cdr.*.{price_date_str}.*.DAMCPCNP4188.csv"
    as_mcpc_files = glob.glob(as_mcpc_pattern)
    if as_mcpc_files:
        print(f"   ✓ AS MCPC file found: {Path(as_mcpc_files[0]).name}")
        df = pd.read_csv(as_mcpc_files[0], nrows=10)
        print(f"     AS Types: {df['AncillaryType'].unique()}")
    else:
        print(f"   ✗ AS MCPC file not found for {price_date_str}")
    
    # 3. Sample calculation for one BESS
    if os.path.exists(dam_file) and dam_spp_files:
        print("\n3. Sample BESS Revenue Calculation:")
        
        # Load DAM data
        dam_df = pd.read_csv(dam_file)
        bess_df = dam_df[dam_df['Resource Type'] == 'PWRSTR']
        
        if not bess_df.empty:
            # Take first BESS
            sample_bess = bess_df.iloc[0]
            resource_name = sample_bess['Resource Name']
            settlement_point = sample_bess['Settlement Point Name']
            
            print(f"\n   Resource: {resource_name}")
            print(f"   Settlement Point: {settlement_point}")
            print(f"   QSE: {sample_bess['QSE']}")
            
            # Get all hours for this BESS
            bess_hours = dam_df[dam_df['Resource Name'] == resource_name]
            
            # Load DAM SPP prices
            spp_df = pd.read_csv(dam_spp_files[0])
            spp_df = spp_df[spp_df['SettlementPoint'] == settlement_point]
            
            print(f"\n   Hour  Award(MW)  SPP($/MWh)  Revenue($)")
            print("   " + "-"*40)
            
            total_revenue = 0
            for _, hour_data in bess_hours.iterrows():
                hour = int(hour_data['Hour Ending'])
                award = hour_data.get('Awarded Quantity', 0)
                
                # Find matching price
                price_row = spp_df[spp_df['HourEnding'] == f"{hour:02d}:00"]
                if not price_row.empty:
                    price = price_row['SettlementPointPrice'].iloc[0]
                    revenue = award * price
                    total_revenue += revenue
                    
                    if award > 0:  # Only show hours with awards
                        print(f"   {hour:4d}  {award:8.1f}  {price:10.2f}  {revenue:10.2f}")
            
            print("   " + "-"*40)
            print(f"   Total DAM Energy Revenue: ${total_revenue:,.2f}")
            
            # AS Revenue
            print(f"\n   Ancillary Services:")
            as_total = 0
            
            if as_mcpc_files:
                as_df = pd.read_csv(as_mcpc_files[0])
                
                for service in ['REGUP', 'REGDN', 'RRS', 'ECRS', 'NSPIN']:
                    service_col = {
                        'REGUP': 'RegUp Awarded',
                        'REGDN': 'RegDown Awarded',
                        'RRS': 'RRSFFR Awarded',
                        'ECRS': 'ECRSSD Awarded',
                        'NSPIN': 'NonSpin Awarded'
                    }.get(service, '')
                    
                    if service_col in bess_hours.columns:
                        mw = bess_hours[service_col].sum()
                        if mw > 0:
                            # Get MCPC
                            mcpc_row = as_df[as_df['AncillaryType'] == service]
                            if not mcpc_row.empty:
                                mcpc = mcpc_row['MCPC'].mean()
                                as_revenue = mw * mcpc
                                as_total += as_revenue
                                print(f"     {service:6s}: {mw:6.1f} MW × ${mcpc:5.2f}/MW = ${as_revenue:8.2f}")
            
            print(f"   Total AS Revenue: ${as_total:,.2f}")
            print(f"\n   TOTAL DAILY REVENUE: ${total_revenue + as_total:,.2f}")


# Test with a recent date
if __name__ == "__main__":
    # Try a date from late 2024 (should have 60-day disclosure)
    test_date = datetime(2024, 10, 1)
    test_single_day(test_date)
    
    # Also try an earlier date
    test_date2 = datetime(2023, 6, 1)
    test_single_day(test_date2)
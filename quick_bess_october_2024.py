#!/usr/bin/env python3
"""
Quick BESS revenue calculation for October 2024
Focus on getting results for one month
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import glob
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Base directories
BASE_DIR = "/Users/enrico/data/ERCOT_data"

def quick_october_analysis():
    """Analyze BESS revenues for October 2024"""
    
    print("BESS Revenue Analysis - October 2024")
    print("="*80)
    
    # 1. First identify BESS resources from October DAM files
    print("\n1. Identifying BESS resources...")
    
    bess_resources = {}
    
    # Just check October files
    for day in range(1, 8):  # First week of October
        date_str = f"{day:02d}-OCT-24"
        dam_file = f"{BASE_DIR}/60-Day_DAM_Disclosure_Reports/csv/60d_DAM_Gen_Resource_Data-{date_str}.csv"
        
        if os.path.exists(dam_file):
            df = pd.read_csv(dam_file)
            bess_df = df[df['Resource Type'] == 'PWRSTR']
            
            for _, row in bess_df.iterrows():
                resource = row['Resource Name']
                if resource not in bess_resources:
                    bess_resources[resource] = {
                        'settlement_point': row['Settlement Point Name'],
                        'qse': row['QSE'],
                        'capacity_mw': row.get('HSL', 0)
                    }
    
    print(f"Found {len(bess_resources)} BESS resources")
    
    # 2. Calculate revenues for first week of October
    print("\n2. Calculating revenues for Oct 1-7, 2024...")
    
    daily_results = []
    
    for day in range(1, 8):
        date = datetime(2024, 10, day)
        date_str = f"{day:02d}-OCT-24"
        print(f"\nProcessing {date_str}...")
        
        # Load DAM data
        dam_file = f"{BASE_DIR}/60-Day_DAM_Disclosure_Reports/csv/60d_DAM_Gen_Resource_Data-{date_str}.csv"
        if not os.path.exists(dam_file):
            continue
            
        dam_df = pd.read_csv(dam_file)
        dam_df = dam_df[dam_df['Resource Type'] == 'PWRSTR']
        
        # Load price data
        price_date_str = date.strftime("%Y%m%d")
        
        # DAM SPP prices
        dam_spp_pattern = f"{BASE_DIR}/DAM_Settlement_Point_Prices/csv/cdr.*.{price_date_str}.*.DAMSPNP4190.csv"
        dam_spp_files = glob.glob(dam_spp_pattern)
        
        dam_prices = {}
        if dam_spp_files:
            spp_df = pd.read_csv(dam_spp_files[0])
            for _, row in spp_df.iterrows():
                sp = row['SettlementPoint']
                hour = int(row['HourEnding'].split(':')[0])
                dam_prices[(sp, hour)] = row['SettlementPointPrice']
        
        # AS MCPC prices
        as_mcpc_pattern = f"{BASE_DIR}/DAM_Clearing_Prices_for_Capacity/csv/cdr.*.{price_date_str}.*.DAMCPCNP4188.csv"
        as_mcpc_files = glob.glob(as_mcpc_pattern)
        
        as_prices = {}
        if as_mcpc_files:
            mcpc_df = pd.read_csv(as_mcpc_files[0])
            for _, row in mcpc_df.iterrows():
                service = row['AncillaryType']
                hour = int(row['HourEnding'].split(':')[0])
                as_prices[(service, hour)] = row['MCPC']
        
        # Calculate daily revenues for each BESS
        for resource in bess_resources:
            resource_data = dam_df[dam_df['Resource Name'] == resource]
            if resource_data.empty:
                continue
                
            settlement_point = bess_resources[resource]['settlement_point']
            
            # Energy revenue
            energy_revenue = 0
            for _, hour_data in resource_data.iterrows():
                hour = int(hour_data['Hour Ending'])
                award = hour_data.get('Awarded Quantity', 0)
                price = dam_prices.get((settlement_point, hour), 0)
                energy_revenue += award * price
            
            # AS revenue
            regup_revenue = 0
            regdown_revenue = 0
            rrs_revenue = 0
            ecrs_revenue = 0
            nonspin_revenue = 0
            
            # RegUp
            regup_mw = resource_data['RegUp Awarded'].sum()
            if regup_mw > 0:
                avg_regup_price = np.mean([as_prices.get(('REGUP', h), 0) for h in range(1, 25)])
                regup_revenue = regup_mw * avg_regup_price
            
            # RegDown
            regdown_mw = resource_data['RegDown Awarded'].sum()
            if regdown_mw > 0:
                avg_regdown_price = np.mean([as_prices.get(('REGDN', h), 0) for h in range(1, 25)])
                regdown_revenue = regdown_mw * avg_regdown_price
            
            # RRS
            rrs_mw = resource_data[['RRSFFR Awarded', 'RRSPFR Awarded', 'RRSUFR Awarded']].sum().sum()
            if rrs_mw > 0:
                avg_rrs_price = np.mean([as_prices.get(('RRS', h), 0) for h in range(1, 25)])
                rrs_revenue = rrs_mw * avg_rrs_price
            
            # ECRS
            ecrs_mw = resource_data['ECRSSD Awarded'].sum()
            if ecrs_mw > 0:
                avg_ecrs_price = np.mean([as_prices.get(('ECRS', h), 0) for h in range(1, 25)])
                ecrs_revenue = ecrs_mw * avg_ecrs_price
            
            # NonSpin
            nonspin_mw = resource_data['NonSpin Awarded'].sum()
            if nonspin_mw > 0:
                avg_nonspin_price = np.mean([as_prices.get(('NSPIN', h), 0) for h in range(1, 25)])
                nonspin_revenue = nonspin_mw * avg_nonspin_price
            
            total_as_revenue = regup_revenue + regdown_revenue + rrs_revenue + ecrs_revenue + nonspin_revenue
            total_revenue = energy_revenue + total_as_revenue
            
            if total_revenue > 0:  # Only record if has revenue
                daily_results.append({
                    'date': date,
                    'resource_name': resource,
                    'settlement_point': settlement_point,
                    'qse': bess_resources[resource]['qse'],
                    'capacity_mw': bess_resources[resource]['capacity_mw'],
                    'energy_revenue': energy_revenue,
                    'regup_revenue': regup_revenue,
                    'regdown_revenue': regdown_revenue,
                    'rrs_revenue': rrs_revenue,
                    'ecrs_revenue': ecrs_revenue,
                    'nonspin_revenue': nonspin_revenue,
                    'total_as_revenue': total_as_revenue,
                    'total_revenue': total_revenue
                })
    
    # 3. Create summary
    if daily_results:
        df = pd.DataFrame(daily_results)
        
        # Save detailed results
        df.to_csv('bess_october_2024_daily.csv', index=False)
        print(f"\nSaved detailed results to bess_october_2024_daily.csv")
        
        # Weekly summary by resource
        weekly_summary = df.groupby('resource_name').agg({
            'capacity_mw': 'first',
            'qse': 'first',
            'energy_revenue': 'sum',
            'regup_revenue': 'sum',
            'regdown_revenue': 'sum',
            'rrs_revenue': 'sum',
            'ecrs_revenue': 'sum',
            'nonspin_revenue': 'sum',
            'total_as_revenue': 'sum',
            'total_revenue': 'sum'
        }).round(2)
        
        weekly_summary = weekly_summary.sort_values('total_revenue', ascending=False)
        
        print("\n" + "="*80)
        print("TOP 20 BESS BY WEEKLY REVENUE (Oct 1-7, 2024)")
        print("="*80)
        
        print(f"\n{'Resource':30s} {'Capacity':>8s} {'Total Rev':>12s} {'Energy':>12s} {'AS Rev':>12s} {'QSE':15s}")
        print("-"*95)
        
        for resource, row in weekly_summary.head(20).iterrows():
            energy_pct = row['energy_revenue'] / row['total_revenue'] * 100 if row['total_revenue'] > 0 else 0
            print(f"{resource:30s} {row['capacity_mw']:>7.1f}MW ${row['total_revenue']:>11,.0f} "
                  f"${row['energy_revenue']:>11,.0f} ${row['total_as_revenue']:>11,.0f} {row['qse']:15s}")
        
        # Overall statistics
        print("\n" + "-"*80)
        print("OVERALL STATISTICS")
        print("-"*80)
        
        total_bess = len(weekly_summary)
        active_bess = len(weekly_summary[weekly_summary['total_revenue'] > 0])
        
        print(f"\nTotal BESS resources: {total_bess}")
        print(f"Active BESS (with revenue): {active_bess}")
        print(f"\nTotal weekly revenue: ${weekly_summary['total_revenue'].sum():,.0f}")
        print(f"  Energy revenue: ${weekly_summary['energy_revenue'].sum():,.0f} ({weekly_summary['energy_revenue'].sum()/weekly_summary['total_revenue'].sum()*100:.1f}%)")
        print(f"  AS revenue: ${weekly_summary['total_as_revenue'].sum():,.0f} ({weekly_summary['total_as_revenue'].sum()/weekly_summary['total_revenue'].sum()*100:.1f}%)")
        
        # Revenue by AS type
        print(f"\nAS Revenue Breakdown:")
        print(f"  RegUp: ${weekly_summary['regup_revenue'].sum():,.0f}")
        print(f"  RegDown: ${weekly_summary['regdown_revenue'].sum():,.0f}")
        print(f"  RRS: ${weekly_summary['rrs_revenue'].sum():,.0f}")
        print(f"  ECRS: ${weekly_summary['ecrs_revenue'].sum():,.0f}")
        print(f"  NonSpin: ${weekly_summary['nonspin_revenue'].sum():,.0f}")
        
        # Save weekly summary
        weekly_summary.to_csv('bess_october_2024_weekly_summary.csv')
        print(f"\nSaved weekly summary to bess_october_2024_weekly_summary.csv")

if __name__ == "__main__":
    quick_october_analysis()
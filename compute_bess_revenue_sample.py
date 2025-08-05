#!/usr/bin/env python3
"""
BESS Revenue Calculator - Sample Analysis for January 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import glob
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Base data directory
BASE_DIR = "/Users/enrico/data/ERCOT_data"

def analyze_sample_days():
    """Analyze a few sample days to show BESS revenue patterns"""
    
    # Analyze January 7-9, 2025 (3 days)
    dates = [datetime(2025, 1, 7), datetime(2025, 1, 8), datetime(2025, 1, 9)]
    
    all_results = []
    
    for date in dates:
        print(f"\nAnalyzing {date.strftime('%Y-%m-%d')}")
        
        # Load DAM Gen Resource Data
        date_str = f"{date.day:02d}-{date.strftime('%b').upper()}-{date.strftime('%y')}"
        dam_file = f"{BASE_DIR}/60-Day_DAM_Disclosure_Reports/csv/60d_DAM_Gen_Resource_Data-{date_str}.csv"
        
        if not os.path.exists(dam_file):
            print(f"File not found: {dam_file}")
            continue
            
        # Read DAM data
        dam_df = pd.read_csv(dam_file)
        
        # Identify BESS resources - use Resource Type = PWRSTR
        bess_mask = dam_df['Resource Type'] == 'PWRSTR'
        bess_resources = dam_df[bess_mask]['Resource Name'].unique()
        
        print(f"Found {len(bess_resources)} BESS resources")
        
        # Get a sample of BESS resources for detailed analysis
        sample_bess = sorted(bess_resources)[:10]  # Top 10 alphabetically
        
        for resource in sample_bess:
            resource_data = dam_df[dam_df['Resource Name'] == resource]
            
            # Calculate basic metrics
            total_awards = resource_data['Awarded Quantity'].sum()
            avg_price = resource_data['Energy Settlement Point Price'].mean()
            
            # AS awards
            regup_awards = resource_data['RegUp Awarded'].sum()
            regdn_awards = resource_data['RegDown Awarded'].sum()
            rrs_awards = resource_data['RRSFFR Awarded'].sum()
            ecrs_awards = resource_data['ECRSSD Awarded'].sum()
            
            # Calculate revenues (simplified)
            energy_revenue = (resource_data['Awarded Quantity'] * 
                            resource_data['Energy Settlement Point Price']).sum()
            
            # AS revenues (using market prices when available)
            regup_revenue = regup_awards * resource_data['RegUp MCPC'].mean() if 'RegUp MCPC' in resource_data else 0
            regdn_revenue = regdn_awards * resource_data['RegDown MCPC'].mean() if 'RegDown MCPC' in resource_data else 0
            
            result = {
                'Date': date,
                'Resource': resource,
                'QSE': resource_data['QSE'].iloc[0] if len(resource_data) > 0 else '',
                'Total_MW_Awards': total_awards,
                'Avg_DAM_Price': avg_price,
                'Energy_Revenue': energy_revenue,
                'RegUp_MW': regup_awards,
                'RegDown_MW': regdn_awards,
                'RRS_MW': rrs_awards,
                'ECRS_MW': ecrs_awards,
                'RegUp_Revenue': regup_revenue,
                'RegDown_Revenue': regdn_revenue
            }
            
            all_results.append(result)
    
    # Create summary DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Print summary
    print("\n" + "="*80)
    print("BESS REVENUE SUMMARY - SAMPLE ANALYSIS")
    print("="*80)
    
    print("\nTop BESS by Energy Revenue (Daily Average):")
    energy_summary = results_df.groupby('Resource')['Energy_Revenue'].mean().sort_values(ascending=False)
    print(energy_summary.head(10))
    
    print("\nAncillary Services Participation:")
    as_summary = results_df.groupby('Resource')[['RegUp_MW', 'RegDown_MW', 'RRS_MW', 'ECRS_MW']].mean()
    as_summary = as_summary[as_summary.sum(axis=1) > 0].sort_values('RegUp_MW', ascending=False)
    print(as_summary.head(10))
    
    print("\nAverage DAM Prices Captured:")
    price_summary = results_df.groupby('Resource')['Avg_DAM_Price'].mean().sort_values(ascending=False)
    print(price_summary.head(10))
    
    # Save detailed results
    output_file = "bess_revenue_sample_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to: {output_file}")
    
    # Show sample of actual data
    print("\nSample of detailed results:")
    print(results_df.head(10))
    
    return results_df


if __name__ == "__main__":
    results = analyze_sample_days()
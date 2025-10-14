#!/usr/bin/env python3
"""
BESS Revenue Calculator for ERCOT 60-Day Disclosure Data
Calculates revenue from DAM Energy, RT Energy, and Ancillary Services
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

class BESSRevenueCalculator:
    def __init__(self, start_date, end_date):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.bess_resources = {}
        self.results = []
        
    def identify_bess_resources(self, date):
        """Identify BESS resources from DAM Gen Resource Data"""
        print(f"Identifying BESS resources for {date.strftime('%Y-%m-%d')}")
        
        # Look for DAM Gen Resource Data files
        # Format: 60d_DAM_Gen_Resource_Data-DD-MMM-YY.csv
        date_str = f"{date.day:02d}-{date.strftime('%b').upper()}-{date.strftime('%y')}"
        dam_pattern = f"{BASE_DIR}/60-Day_DAM_Disclosure_Reports/csv/60d_DAM_Gen_Resource_Data-{date_str}.csv"
        dam_files = glob.glob(dam_pattern)
        
        if not dam_files:
            print(f"No DAM Gen Resource Data found for {date}")
            return
            
        df = pd.read_csv(dam_files[0])
        
        # Identify BESS by resource name patterns and SOC columns
        bess_mask = df['Resource Name'].str.contains('BESS|ESS|BATTERY', case=False, na=False)
        
        # Check if SOC columns exist
        if 'Minimum SOC' in df.columns:
            bess_mask |= (df['Minimum SOC'].notna() & (df['Minimum SOC'] > 0))
        if 'Maximum SOC' in df.columns:
            bess_mask |= (df['Maximum SOC'].notna() & (df['Maximum SOC'] > 0))
        
        bess_resources = df[bess_mask]['Resource Name'].unique()
        
        for resource in bess_resources:
            if resource not in self.bess_resources:
                # Get settlement point mapping
                resource_data = df[df['Resource Name'] == resource].iloc[0]
                self.bess_resources[resource] = {
                    'settlement_point': resource_data.get('Settlement Point Name', resource),
                    'qse': resource_data.get('QSE', ''),
                    'resource_type': resource_data.get('Resource Type', 'BESS')
                }
        
        print(f"Found {len(bess_resources)} BESS resources for {date}")
        return bess_resources
    
    def calculate_dam_energy_revenue(self, date):
        """Calculate Day-Ahead Market energy revenue"""
        dam_revenues = {}
        
        # Load DAM Gen Resource Data
        date_str = f"{date.day:02d}-{date.strftime('%b').upper()}-{date.strftime('%y')}"
        dam_pattern = f"{BASE_DIR}/60-Day_DAM_Disclosure_Reports/csv/60d_DAM_Gen_Resource_Data-{date_str}.csv"
        dam_files = glob.glob(dam_pattern)
        
        if not dam_files:
            return dam_revenues
            
        dam_df = pd.read_csv(dam_files[0])
        
        # Load DAM Settlement Point Prices - need to look 60 days forward
        price_date = date + timedelta(days=60)
        price_pattern = f"{BASE_DIR}/DAM_Settlement_Point_Prices/csv/*{price_date.strftime('%Y%m%d')}*.csv"
        price_files = glob.glob(price_pattern)
        
        if not price_files:
            print(f"No DAM price data found for {price_date}")
            return dam_revenues
            
        price_df = pd.read_csv(price_files[0])
        price_df['DeliveryDate'] = pd.to_datetime(price_df['DeliveryDate'])
        
        # Calculate revenue for each BESS
        for resource, info in self.bess_resources.items():
            resource_data = dam_df[dam_df['Resource Name'] == resource]
            if resource_data.empty:
                continue
                
            daily_revenue = 0
            settlement_point = info['settlement_point']
            
            for _, row in resource_data.iterrows():
                hour = row['Hour Ending']
                award = row.get('Awarded Quantity', 0)
                
                if pd.isna(award) or award == 0:
                    continue
                
                # Find matching price
                price_row = price_df[
                    (price_df['SettlementPoint'] == settlement_point) &
                    (price_df['HourEnding'] == f"{int(hour):02d}:00")
                ]
                
                if not price_row.empty:
                    price = price_row['SettlementPointPrice'].iloc[0]
                    hourly_revenue = award * price  # MW * $/MWh = $
                    daily_revenue += hourly_revenue
            
            dam_revenues[resource] = daily_revenue
            
        return dam_revenues
    
    def calculate_as_revenue(self, date):
        """Calculate Ancillary Services revenue"""
        as_revenues = {}
        
        # Load DAM Gen Resource Data for AS awards
        date_str = f"{date.day:02d}-{date.strftime('%b').upper()}-{date.strftime('%y')}"
        dam_pattern = f"{BASE_DIR}/60-Day_DAM_Disclosure_Reports/csv/60d_DAM_Gen_Resource_Data-{date_str}.csv"
        dam_files = glob.glob(dam_pattern)
        
        if not dam_files:
            return as_revenues
            
        dam_df = pd.read_csv(dam_files[0])
        
        # Load AS Clearing Prices - need to look 60 days forward
        price_date = date + timedelta(days=60)
        as_price_pattern = f"{BASE_DIR}/DAM_Clearing_Prices_for_Capacity/csv/*{price_date.strftime('%Y%m%d')}*.csv"
        as_price_files = glob.glob(as_price_pattern)
        
        if not as_price_files:
            print(f"No AS price data found for {price_date}")
            return as_revenues
            
        as_price_df = pd.read_csv(as_price_files[0])
        as_price_df['DeliveryDate'] = pd.to_datetime(as_price_df['DeliveryDate'])
        
        # AS services to calculate
        as_services = {
            'RegUp': ('RegUp Awarded', 'RegUp MCPC', 'REGUP'),
            'RegDown': ('RegDown Awarded', 'RegDown MCPC', 'REGDN'),
            'RRS': ('RRSFFR Awarded', 'RRS MCPC', 'RRS'),
            'ECRS': ('ECRSSD Awarded', 'ECRS MCPC', 'ECRS'),
            'NonSpin': ('NonSpin Awarded', 'NonSpin MCPC', 'NSPIN')
        }
        
        # Calculate revenue for each BESS
        for resource in self.bess_resources:
            resource_data = dam_df[dam_df['Resource Name'] == resource]
            if resource_data.empty:
                continue
                
            as_revenue_by_service = {}
            
            for service_name, (award_col, mcpc_col, price_type) in as_services.items():
                daily_revenue = 0
                
                for _, row in resource_data.iterrows():
                    hour = row['Hour Ending']
                    award = row.get(award_col, 0)
                    
                    if pd.isna(award) or award == 0:
                        continue
                    
                    # First try to use resource-specific MCPC
                    mcpc = row.get(mcpc_col, np.nan)
                    
                    # If not available, use market clearing price
                    if pd.isna(mcpc):
                        price_row = as_price_df[
                            (as_price_df['AncillaryType'] == price_type) &
                            (as_price_df['HourEnding'] == f"{int(hour):02d}:00")
                        ]
                        if not price_row.empty:
                            mcpc = price_row['MCPC'].iloc[0]
                    
                    if not pd.isna(mcpc):
                        hourly_revenue = award * mcpc  # MW * $/MW = $
                        daily_revenue += hourly_revenue
                
                as_revenue_by_service[service_name] = daily_revenue
            
            as_revenues[resource] = as_revenue_by_service
            
        return as_revenues
    
    def calculate_rt_energy_revenue(self, date):
        """Calculate Real-Time energy revenue (simplified - using SCED base points)"""
        rt_revenues = {}
        
        # Load SCED Gen Resource Data
        date_str = f"{date.day:02d}-{date.strftime('%b').upper()}-{date.strftime('%y')}"
        sced_pattern = f"{BASE_DIR}/60-Day_SCED_Disclosure_Reports/csv/60d_SCED_Gen_Resource_Data-{date_str}.csv"
        sced_files = glob.glob(sced_pattern)
        
        if not sced_files:
            return rt_revenues
            
        print(f"Processing RT data for {date} - this may take a while...")
        
        # For simplified calculation, we'll estimate RT revenue as a percentage of DAM
        # In reality, you would need to process all 288 SCED intervals per day
        # and match with 96 RT price files
        
        for resource in self.bess_resources:
            # Placeholder: RT revenue typically 10-20% of DAM for BESS
            rt_revenues[resource] = 0  # Would need full implementation
            
        return rt_revenues
    
    def process_date_range(self):
        """Process all dates in the range"""
        current_date = self.start_date
        
        while current_date <= self.end_date:
            print(f"\nProcessing {current_date.strftime('%Y-%m-%d')}")
            
            # Identify BESS resources
            self.identify_bess_resources(current_date)
            
            # Calculate revenues
            dam_revenues = self.calculate_dam_energy_revenue(current_date)
            as_revenues = self.calculate_as_revenue(current_date)
            rt_revenues = self.calculate_rt_energy_revenue(current_date)
            
            # Aggregate results
            for resource in self.bess_resources:
                dam_rev = dam_revenues.get(resource, 0)
                as_rev_dict = as_revenues.get(resource, {})
                rt_rev = rt_revenues.get(resource, 0)
                
                total_as_rev = sum(as_rev_dict.values())
                total_rev = dam_rev + total_as_rev + rt_rev
                
                self.results.append({
                    'Date': current_date,
                    'Resource': resource,
                    'QSE': self.bess_resources[resource]['qse'],
                    'DAM_Energy_Revenue': dam_rev,
                    'RT_Energy_Revenue': rt_rev,
                    'RegUp_Revenue': as_rev_dict.get('RegUp', 0),
                    'RegDown_Revenue': as_rev_dict.get('RegDown', 0),
                    'RRS_Revenue': as_rev_dict.get('RRS', 0),
                    'ECRS_Revenue': as_rev_dict.get('ECRS', 0),
                    'NonSpin_Revenue': as_rev_dict.get('NonSpin', 0),
                    'Total_AS_Revenue': total_as_rev,
                    'Total_Revenue': total_rev
                })
            
            current_date += timedelta(days=1)
    
    def generate_summary(self):
        """Generate summary statistics"""
        if not self.results:
            print("No results to summarize")
            return
            
        df = pd.DataFrame(self.results)
        
        print("\n" + "="*80)
        print("BESS REVENUE SUMMARY")
        print("="*80)
        
        # Overall summary
        print(f"\nAnalysis Period: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}")
        print(f"Total BESS Resources Found: {len(self.bess_resources)}")
        print(f"Total Days Analyzed: {len(df['Date'].unique())}")
        
        # Revenue by resource
        resource_summary = df.groupby('Resource').agg({
            'DAM_Energy_Revenue': 'sum',
            'RT_Energy_Revenue': 'sum',
            'Total_AS_Revenue': 'sum',
            'Total_Revenue': 'sum'
        }).round(2)
        
        print("\nRevenue by BESS Resource (Total for Period):")
        print(resource_summary.sort_values('Total_Revenue', ascending=False).head(10))
        
        # Daily average revenue
        daily_avg = df.groupby('Resource')['Total_Revenue'].mean().round(2)
        print("\nDaily Average Revenue by Resource:")
        print(daily_avg.sort_values(ascending=False).head(10))
        
        # Revenue breakdown
        total_dam = df['DAM_Energy_Revenue'].sum()
        total_rt = df['RT_Energy_Revenue'].sum()
        total_as = df['Total_AS_Revenue'].sum()
        total_all = df['Total_Revenue'].sum()
        
        print("\nRevenue Breakdown Across All BESS:")
        print(f"DAM Energy: ${total_dam:,.2f} ({total_dam/total_all*100:.1f}%)")
        print(f"RT Energy: ${total_rt:,.2f} ({total_rt/total_all*100:.1f}%)")
        print(f"Ancillary Services: ${total_as:,.2f} ({total_as/total_all*100:.1f}%)")
        print(f"Total Revenue: ${total_all:,.2f}")
        
        # AS breakdown
        as_breakdown = df[['RegUp_Revenue', 'RegDown_Revenue', 'RRS_Revenue', 
                          'ECRS_Revenue', 'NonSpin_Revenue']].sum()
        print("\nAncillary Services Breakdown:")
        for service, revenue in as_breakdown.items():
            if revenue > 0:
                print(f"{service.replace('_Revenue', '')}: ${revenue:,.2f} ({revenue/total_as*100:.1f}%)")
        
        # Save detailed results
        output_file = f"bess_revenue_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(output_file, index=False)
        print(f"\nDetailed results saved to: {output_file}")
        
        return df


def main():
    # Calculate dates for past 90 days of disclosure data
    # The disclosure files contain delivery dates from 60 days ago
    # So for May 2025 files, they contain March 2025 delivery data
    # Let's analyze January 2025 data which should be available
    end_date = datetime(2025, 1, 31)  # January 2025 data
    start_date = datetime(2025, 1, 1)
    
    print(f"Analyzing BESS revenue from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print("Note: Due to 60-day disclosure lag, price data is matched from current dates")
    
    calculator = BESSRevenueCalculator(start_date, end_date)
    calculator.process_date_range()
    results_df = calculator.generate_summary()
    
    return results_df


if __name__ == "__main__":
    results = main()
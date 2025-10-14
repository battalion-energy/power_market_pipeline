#!/usr/bin/env python3
"""
TBX Calculator V2 - Battery Arbitrage Revenue Analysis
Calculates TB2 (2-hour) and TB4 (4-hour) battery arbitrage values for all nodes
Handles the actual ERCOT data structure properly
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

class TBXCalculator:
    def __init__(self, efficiency=0.9, data_dir=None, output_dir=None):
        self.efficiency = efficiency  # 90% round-trip efficiency
        self.data_dir = Path(data_dir or "/home/enrico/data/ERCOT_data/rollup_files/flattened")
        self.output_dir = Path(output_dir or "/home/enrico/data/ERCOT_data/tbx_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def calculate_tbx_for_day(self, prices, hours):
        """Calculate TBX for a single day's prices"""
        if len(prices) < 24 or np.isnan(prices).all():
            return 0.0, [], []
        
        # Take first 24 hours if more than 24
        prices = np.array(prices[:24])
        
        # Handle NaNs
        if np.isnan(prices).any():
            return 0.0, [], []
        
        # Get indices sorted by price
        sorted_indices = np.argsort(prices)
        
        # Charge during lowest price hours
        charge_hours = sorted_indices[:hours]
        # Discharge during highest price hours  
        discharge_hours = sorted_indices[-hours:]
        
        # Calculate revenue with efficiency losses
        charge_cost = prices[charge_hours].sum()
        discharge_revenue = prices[discharge_hours].sum()
        
        # Apply efficiency to discharge (we lose 10% on round trip)
        net_revenue = discharge_revenue * self.efficiency - charge_cost
        
        return net_revenue, charge_hours.tolist(), discharge_hours.tolist()
    
    def process_year(self, year):
        """Process all nodes for a given year"""
        print(f"\n📅 Processing year {year}")
        
        # Load price data
        da_path = self.data_dir / f"DA_prices_{year}.parquet"
        rt_path = self.data_dir / f"RT_prices_hourly_{year}.parquet"  # Use hourly RT
        
        da_df = pd.read_parquet(da_path) if da_path.exists() else pd.DataFrame()
        rt_df = pd.read_parquet(rt_path) if rt_path.exists() else pd.DataFrame()
        
        if da_df.empty and rt_df.empty:
            print(f"  ❌ No price data available for {year}")
            return
        
        # Identify price columns (not date/metadata columns)
        da_price_cols = []
        rt_price_cols = []
        
        if not da_df.empty:
            # Skip date and metadata columns
            skip_cols = ['DeliveryDate', 'DeliveryDateStr', 'datetime_ts', 'datetime', 
                        'DeliveryInterval', 'DSTFlag', 'RepeatedHourFlag']
            da_price_cols = [col for col in da_df.columns if col not in skip_cols]
            # Add date column for grouping
            if 'DeliveryDate' in da_df.columns:
                da_df['date'] = pd.to_datetime(da_df['DeliveryDate']).dt.date
            elif 'datetime' in da_df.columns:
                da_df['date'] = pd.to_datetime(da_df['datetime']).dt.date
        
        if not rt_df.empty:
            skip_cols = ['datetime', 'timestamp', 'date']
            rt_price_cols = [col for col in rt_df.columns if col not in skip_cols]
            # Add date column for grouping
            if 'datetime' in rt_df.columns:
                rt_df['date'] = pd.to_datetime(rt_df['datetime']).dt.date
        
        # Get all unique nodes
        nodes = sorted(list(set(da_price_cols + rt_price_cols)))
        print(f"  📊 Found {len(nodes)} price nodes")
        
        # Process each node
        all_results = []
        
        for idx, node in enumerate(nodes):
            if (idx + 1) % 5 == 0:
                print(f"    Processing node {idx+1}/{len(nodes)}...")
            
            # Process DA prices
            if node in da_price_cols and not da_df.empty:
                # Group by date
                for date, group in da_df.groupby('date'):
                    prices = group[node].values
                    tb2_revenue, _, _ = self.calculate_tbx_for_day(prices, 2)
                    tb4_revenue, _, _ = self.calculate_tbx_for_day(prices, 4)
                    
                    all_results.append({
                        'date': date,
                        'node': node,
                        'tb2_da_revenue': tb2_revenue,
                        'tb2_rt_revenue': 0,
                        'tb4_da_revenue': tb4_revenue,
                        'tb4_rt_revenue': 0
                    })
            
            # Process RT prices
            if node in rt_price_cols and not rt_df.empty:
                # Group by date
                for date, group in rt_df.groupby('date'):
                    prices = group[node].values
                    tb2_revenue, _, _ = self.calculate_tbx_for_day(prices, 2)
                    tb4_revenue, _, _ = self.calculate_tbx_for_day(prices, 4)
                    
                    # Find matching DA result or create new
                    found = False
                    for result in all_results:
                        if result['date'] == date and result['node'] == node:
                            result['tb2_rt_revenue'] = tb2_revenue
                            result['tb4_rt_revenue'] = tb4_revenue
                            found = True
                            break
                    
                    if not found:
                        all_results.append({
                            'date': date,
                            'node': node,
                            'tb2_da_revenue': 0,
                            'tb2_rt_revenue': tb2_revenue,
                            'tb4_da_revenue': 0,
                            'tb4_rt_revenue': tb4_revenue
                        })
        
        # Convert to DataFrame
        daily_df = pd.DataFrame(all_results)
        
        if daily_df.empty:
            print(f"  ❌ No results for {year}")
            return
            
        # Save daily results
        daily_path = self.output_dir / f"tbx_daily_{year}.parquet"
        daily_df.to_parquet(daily_path)
        print(f"  💾 Saved {len(daily_df)} daily records to {daily_path}")
        
        # Aggregate monthly
        monthly_df = daily_df.copy()
        monthly_df['month'] = pd.to_datetime(monthly_df['date']).dt.month
        monthly_df = monthly_df.groupby(['node', 'month']).agg({
            'tb2_da_revenue': 'sum',
            'tb2_rt_revenue': 'sum',
            'tb4_da_revenue': 'sum',
            'tb4_rt_revenue': 'sum',
            'date': 'count'
        }).rename(columns={'date': 'days_count'}).reset_index()
        
        monthly_path = self.output_dir / f"tbx_monthly_{year}.parquet"
        monthly_df.to_parquet(monthly_path)
        print(f"  💾 Saved {len(monthly_df)} monthly records to {monthly_path}")
        
        # Aggregate annual
        annual_df = daily_df.groupby('node').agg({
            'tb2_da_revenue': 'sum',
            'tb2_rt_revenue': 'sum',
            'tb4_da_revenue': 'sum',
            'tb4_rt_revenue': 'sum',
            'date': 'count'
        }).rename(columns={'date': 'days_count'}).reset_index()
        
        annual_path = self.output_dir / f"tbx_annual_{year}.parquet"
        annual_df.to_parquet(annual_path)
        print(f"  💾 Saved {len(annual_df)} annual records to {annual_path}")
        
        # Print top 10 nodes by TB2 DA revenue
        if not annual_df.empty and 'tb2_da_revenue' in annual_df.columns:
            top_nodes = annual_df.nlargest(10, 'tb2_da_revenue')
            print(f"\n  🏆 Top 10 Nodes by TB2 Day-Ahead Revenue for {year}:")
            print(f"  {'Node':<20} {'TB2 DA ($)':>15} {'TB2 RT ($)':>15} {'TB4 DA ($)':>15} {'Days':>8}")
            print(f"  {'-'*78}")
            for _, row in top_nodes.iterrows():
                print(f"  {row['node']:<20} {row['tb2_da_revenue']:>15,.2f} {row['tb2_rt_revenue']:>15,.2f} {row['tb4_da_revenue']:>15,.2f} {row['days_count']:>8}")
        
        # Also show top RT nodes if different
        if not annual_df.empty and 'tb2_rt_revenue' in annual_df.columns:
            top_rt_nodes = annual_df.nlargest(10, 'tb2_rt_revenue')
            print(f"\n  🏆 Top 10 Nodes by TB2 Real-Time Revenue for {year}:")
            print(f"  {'Node':<20} {'TB2 RT ($)':>15} {'TB4 RT ($)':>15} {'Days':>8}")
            print(f"  {'-'*58}")
            for _, row in top_rt_nodes.iterrows():
                print(f"  {row['node']:<20} {row['tb2_rt_revenue']:>15,.2f} {row['tb4_rt_revenue']:>15,.2f} {row['days_count']:>8}")
        
        print(f"\n  ✅ Year {year} complete!")

def main():
    parser = argparse.ArgumentParser(description='Calculate TBX battery arbitrage values')
    parser.add_argument('--efficiency', type=float, default=0.9, 
                       help='Round-trip efficiency (default: 0.9 for 90%)')
    parser.add_argument('--years', type=int, nargs='+', default=[2021, 2022, 2023, 2024, 2025],
                       help='Years to process')
    parser.add_argument('--data-dir', type=str, 
                       default='/home/enrico/data/ERCOT_data/rollup_files/flattened',
                       help='Input data directory')
    parser.add_argument('--output-dir', type=str,
                       default='/home/enrico/data/ERCOT_data/tbx_results',
                       help='Output directory')
    
    args = parser.parse_args()
    
    print("⚡ TBX Calculator V2 - Battery Arbitrage Analysis")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Efficiency: {args.efficiency*100:.1f}%")
    print(f"  TB2: 2-hour battery arbitrage")
    print(f"  TB4: 4-hour battery arbitrage")
    print(f"  Years: {args.years}")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Output directory: {args.output_dir}")
    
    calculator = TBXCalculator(
        efficiency=args.efficiency,
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    
    for year in args.years:
        calculator.process_year(year)
    
    print("\n✅ TBX calculation complete!")
    print(f"📁 Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
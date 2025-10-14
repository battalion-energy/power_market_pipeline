#!/usr/bin/env python3
"""
TBX Calculator - Battery Arbitrage Revenue Analysis
Calculates TB2 (2-hour) and TB4 (4-hour) battery arbitrage values for all nodes
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
        if len(prices) < 24:
            return 0.0, [], []
        
        # Take first 24 hours if more than 24
        prices = prices[:24]
        
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
    
    def process_node_year(self, node, year, da_df, rt_df):
        """Process a single node for a year"""
        results = []
        
        # Get the node's prices
        if node not in da_df.columns and node not in rt_df.columns:
            return results
            
        da_prices = da_df[node].values if node in da_df.columns else None
        rt_prices = rt_df[node].values if node in rt_df.columns else None
        
        # Get datetime index
        datetime_index = da_df.index if da_prices is not None else rt_df.index
        
        # Group by day
        dates = pd.to_datetime(datetime_index).normalize().unique()
        
        for date in dates:
            day_mask = pd.to_datetime(datetime_index).normalize() == date
            
            # Get prices for this day
            da_day = da_prices[day_mask] if da_prices is not None else np.array([])
            rt_day = rt_prices[day_mask] if rt_prices is not None else np.array([])
            
            # For RT, convert 15-min to hourly by averaging
            if len(rt_day) > 24:
                rt_hourly = []
                for i in range(0, len(rt_day), 4):
                    rt_hourly.append(np.mean(rt_day[i:i+4]))
                rt_day = np.array(rt_hourly[:24])
            
            # Calculate TB2 and TB4
            tb2_da, _, _ = self.calculate_tbx_for_day(da_day, 2) if len(da_day) >= 24 else (0, [], [])
            tb4_da, _, _ = self.calculate_tbx_for_day(da_day, 4) if len(da_day) >= 24 else (0, [], [])
            tb2_rt, _, _ = self.calculate_tbx_for_day(rt_day, 2) if len(rt_day) >= 24 else (0, [], [])
            tb4_rt, _, _ = self.calculate_tbx_for_day(rt_day, 4) if len(rt_day) >= 24 else (0, [], [])
            
            results.append({
                'date': date,
                'node': node,
                'tb2_da_revenue': tb2_da,
                'tb2_rt_revenue': tb2_rt,
                'tb4_da_revenue': tb4_da,
                'tb4_rt_revenue': tb4_rt
            })
        
        return results
    
    def process_year(self, year):
        """Process all nodes for a given year"""
        print(f"\n📅 Processing year {year}")
        
        # Load price data
        da_path = self.data_dir / f"DA_prices_{year}.parquet"
        rt_path = self.data_dir / f"RT_prices_15min_{year}.parquet"
        
        da_df = pd.read_parquet(da_path) if da_path.exists() else pd.DataFrame()
        rt_df = pd.read_parquet(rt_path) if rt_path.exists() else pd.DataFrame()
        
        if da_df.empty and rt_df.empty:
            print(f"  ❌ No price data available for {year}")
            return
        
        # Get all unique nodes
        nodes = set()
        if not da_df.empty:
            nodes.update([col for col in da_df.columns if col != 'datetime'])
        if not rt_df.empty:
            nodes.update([col for col in rt_df.columns if col != 'datetime'])
        
        nodes = sorted(list(nodes))
        print(f"  📊 Found {len(nodes)} unique nodes")
        
        # Set datetime as index
        if not da_df.empty and 'datetime' in da_df.columns:
            da_df.set_index('datetime', inplace=True)
        if not rt_df.empty and 'datetime' in rt_df.columns:
            rt_df.set_index('datetime', inplace=True)
        
        # Process nodes in parallel
        all_results = []
        with ProcessPoolExecutor(max_workers=24) as executor:
            futures = {
                executor.submit(self.process_node_year, node, year, da_df, rt_df): node 
                for node in nodes
            }
            
            completed = 0
            for future in as_completed(futures):
                completed += 1
                if completed % 100 == 0:
                    print(f"    Processed {completed}/{len(nodes)} nodes...")
                try:
                    results = future.result()
                    all_results.extend(results)
                except Exception as e:
                    node = futures[future]
                    print(f"    ⚠️  Error processing node {node}: {e}")
        
        # Convert to DataFrame
        daily_df = pd.DataFrame(all_results)
        
        if daily_df.empty:
            print(f"  ❌ No results for {year}")
            return
            
        # Save daily results
        daily_path = self.output_dir / f"tbx_daily_{year}.parquet"
        daily_df.to_parquet(daily_path)
        print(f"  💾 Saved daily results to {daily_path}")
        
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
        print(f"  💾 Saved monthly results to {monthly_path}")
        
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
        print(f"  💾 Saved annual results to {annual_path}")
        
        # Print top 10 nodes
        top_nodes = annual_df.nlargest(10, 'tb2_da_revenue')
        print(f"\n  🏆 Top 10 Nodes by TB2 Day-Ahead Revenue for {year}:")
        print(f"  {'Node':<30} {'TB2 DA ($)':>15} {'TB4 DA ($)':>15} {'Days':>10}")
        print(f"  {'-'*70}")
        for _, row in top_nodes.iterrows():
            print(f"  {row['node']:<30} {row['tb2_da_revenue']:>15,.2f} {row['tb4_da_revenue']:>15,.2f} {row['days_count']:>10}")
        
        print(f"  ✅ Year {year} complete!")

def main():
    parser = argparse.ArgumentParser(description='Calculate TBX battery arbitrage values')
    parser.add_argument('--efficiency', type=float, default=0.9, 
                       help='Round-trip efficiency (default: 0.9 for 90%)')
    parser.add_argument('--years', type=int, nargs='+', default=[2021, 2022, 2023, 2024],
                       help='Years to process')
    parser.add_argument('--data-dir', type=str, 
                       default='/home/enrico/data/ERCOT_data/rollup_files/flattened',
                       help='Input data directory')
    parser.add_argument('--output-dir', type=str,
                       default='/home/enrico/data/ERCOT_data/tbx_results',
                       help='Output directory')
    
    args = parser.parse_args()
    
    print("⚡ TBX Calculator - Battery Arbitrage Analysis")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Efficiency: {args.efficiency*100:.1f}%")
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

if __name__ == "__main__":
    main()
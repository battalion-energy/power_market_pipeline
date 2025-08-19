#!/usr/bin/env python3
"""
TBX Calculator for ALL Settlement Points
Processes all 900+ ERCOT settlement points for TB2/TB4 arbitrage analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import pyarrow.parquet as pq
import pyarrow as pa
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

class TBXCalculatorAllNodes:
    def __init__(self, efficiency=0.9, data_dir=None, output_dir=None):
        self.efficiency = efficiency
        self.data_dir = data_dir or Path("/home/enrico/data/ERCOT_data/rollup_files")
        self.output_dir = output_dir or Path("/home/enrico/data/ERCOT_data/tbx_results_all_nodes")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
    def load_da_prices(self, year):
        """Load raw DA prices with all settlement points"""
        file_path = self.data_dir / "DA_prices" / f"{year}.parquet"
        if not file_path.exists():
            print(f"⚠️  DA prices not found for {year}")
            return None
            
        print(f"  📂 Loading DA prices from {file_path}")
        df = pd.read_parquet(file_path)
        
        # Convert to wide format (pivot)
        print(f"  🔄 Pivoting data for {df['SettlementPoint'].nunique()} settlement points...")
        
        # Handle different HourEnding formats
        if df['HourEnding'].dtype == 'object' and ':' in str(df['HourEnding'].iloc[0]):
            # Format is "HH:00" - handle "24:00" as hour 0 of next day
            df['HourEnding'] = df['HourEnding'].str.replace('24:00', '00:00')
            df['hour_int'] = pd.to_datetime(df['HourEnding'], format='%H:%M').dt.hour
            # Adjust for HourEnding convention (1-24 becomes 0-23)
            df.loc[df['hour_int'] == 0, 'hour_int'] = 24
            df['hour_int'] = df['hour_int'] - 1
            df['datetime'] = pd.to_datetime(df['DeliveryDate']) + pd.to_timedelta(df['hour_int'], unit='h')
        elif 'hour' in df.columns and df['hour'].dtype != 'object':
            # Already has numeric hour column
            df['datetime'] = pd.to_datetime(df['DeliveryDate']) + pd.to_timedelta(df['hour']-1, unit='h')
        else:
            # Format is integer in HourEnding
            df['hour_int'] = df['HourEnding'].astype(int) - 1
            df['datetime'] = pd.to_datetime(df['DeliveryDate']) + pd.to_timedelta(df['hour_int'], unit='h')
        
        # Pivot to wide format
        df_wide = df.pivot_table(
            index='datetime',
            columns='SettlementPoint',
            values='SettlementPointPrice',
            aggfunc='first'
        ).reset_index()
        
        print(f"  ✅ Loaded {len(df_wide)} hours × {len(df_wide.columns)-1} settlement points")
        return df_wide
    
    def calculate_arbitrage(self, prices, hours):
        """Calculate arbitrage for given number of hours"""
        if len(prices) < hours * 2:
            return 0.0
        
        # Find lowest hours for charging and highest for discharging
        sorted_indices = np.argsort(prices)
        charge_hours = sorted_indices[:hours]
        discharge_hours = sorted_indices[-hours:]
        
        # Calculate revenues
        charge_cost = prices[charge_hours].sum()
        discharge_revenue = prices[discharge_hours].sum()
        
        # Apply efficiency
        net_revenue = discharge_revenue * self.efficiency - charge_cost
        return net_revenue
    
    def process_node_year(self, node, year, da_prices):
        """Process a single node for a year"""
        if node not in da_prices.columns:
            return []
        
        prices = da_prices[node].values
        datetime_col = da_prices['datetime']
        
        results = []
        
        # Group by day
        da_prices['date'] = datetime_col.dt.date
        
        for date, day_data in da_prices.groupby('date'):
            day_prices = day_data[node].values
            
            # Skip days with missing data
            if np.isnan(day_prices).any() or len(day_prices) != 24:
                continue
            
            tb2_revenue = self.calculate_arbitrage(day_prices, 2)
            tb4_revenue = self.calculate_arbitrage(day_prices, 4)
            
            results.append({
                'date': date,
                'year': year,
                'month': date.month,
                'node': node,
                'tb2_revenue': tb2_revenue,
                'tb4_revenue': tb4_revenue,
                'price_mean': np.mean(day_prices),
                'price_std': np.std(day_prices),
                'price_min': np.min(day_prices),
                'price_max': np.max(day_prices)
            })
        
        return results
    
    def process_year(self, year):
        """Process all nodes for a given year"""
        print(f"\n📅 Processing year {year}")
        
        # Load DA prices
        da_prices = self.load_da_prices(year)
        if da_prices is None:
            return pd.DataFrame()
        
        # Get all settlement points (excluding datetime)
        nodes = [col for col in da_prices.columns if col != 'datetime' and col != 'date']
        print(f"  📍 Processing {len(nodes)} settlement points...")
        
        all_results = []
        
        # Process in batches for memory efficiency
        batch_size = 50
        num_batches = (len(nodes) + batch_size - 1) // batch_size
        for batch_num, i in enumerate(range(0, len(nodes), batch_size), 1):
            print(f"  Processing batch {batch_num}/{num_batches}...")
            batch_nodes = nodes[i:i+batch_size]
            
            for node in batch_nodes:
                node_results = self.process_node_year(node, year, da_prices)
                all_results.extend(node_results)
        
        if all_results:
            df = pd.DataFrame(all_results)
            
            # Save daily results
            daily_file = self.output_dir / f"tbx_daily_{year}_all_nodes.parquet"
            df.to_parquet(daily_file, index=False)
            print(f"  💾 Saved {len(df)} daily records to {daily_file.name}")
            
            # Create monthly aggregation
            monthly = df.groupby(['year', 'month', 'node']).agg({
                'tb2_revenue': 'sum',
                'tb4_revenue': 'sum',
                'price_mean': 'mean',
                'price_std': 'mean',
                'date': 'count'
            }).rename(columns={'date': 'days'}).reset_index()
            
            monthly_file = self.output_dir / f"tbx_monthly_{year}_all_nodes.parquet"
            monthly.to_parquet(monthly_file, index=False)
            print(f"  💾 Saved {len(monthly)} monthly records")
            
            # Create annual summary
            annual = df.groupby(['year', 'node']).agg({
                'tb2_revenue': 'sum',
                'tb4_revenue': 'sum',
                'price_mean': 'mean',
                'price_std': 'mean',
                'date': 'count'
            }).rename(columns={'date': 'days'}).reset_index()
            
            annual_file = self.output_dir / f"tbx_annual_{year}_all_nodes.parquet"
            annual.to_parquet(annual_file, index=False)
            print(f"  💾 Saved {len(annual)} annual records")
            
            return annual
        
        return pd.DataFrame()
    
    def create_leaderboard(self, years):
        """Create leaderboard from all years"""
        print("\n🏆 Creating All-Nodes Leaderboard...")
        
        all_annual = []
        for year in years:
            annual_file = self.output_dir / f"tbx_annual_{year}_all_nodes.parquet"
            if annual_file.exists():
                df = pd.read_parquet(annual_file)
                all_annual.append(df)
        
        if not all_annual:
            print("No data found for leaderboard")
            return
        
        # Combine all years
        combined = pd.concat(all_annual, ignore_index=True)
        
        # Sum across all years
        leaderboard = combined.groupby('node').agg({
            'tb2_revenue': 'sum',
            'tb4_revenue': 'sum',
            'price_mean': 'mean',
            'price_std': 'mean',
            'days': 'sum'
        }).reset_index()
        
        # Add revenue per MW metrics
        leaderboard['tb2_per_mw_year'] = leaderboard['tb2_revenue'] / len(years)
        leaderboard['tb4_per_mw_year'] = leaderboard['tb4_revenue'] / len(years)
        
        # Sort by TB4 revenue
        leaderboard = leaderboard.sort_values('tb4_revenue', ascending=False)
        
        # Save leaderboard
        leaderboard_file = self.output_dir / "tbx_leaderboard_all_nodes.parquet"
        leaderboard.to_parquet(leaderboard_file, index=False)
        
        # Also save as CSV for easy viewing
        leaderboard.to_csv(self.output_dir / "tbx_leaderboard_all_nodes.csv", index=False)
        
        # Print top 50
        print("\n🏆 TOP 50 SETTLEMENT POINTS BY TB4 REVENUE (All Years):")
        print("=" * 100)
        print(f"{'Rank':<5} {'Settlement Point':<25} {'TB2 Revenue':<15} {'TB4 Revenue':<15} {'Days':<10}")
        print("-" * 100)
        
        for i, row in leaderboard.head(50).iterrows():
            print(f"{i+1:<5} {row['node'][:25]:<25} ${row['tb2_revenue']:>13,.2f} ${row['tb4_revenue']:>13,.2f} {row['days']:>10.0f}")
        
        print(f"\n📊 Total settlement points analyzed: {len(leaderboard)}")
        print(f"💾 Leaderboard saved to: {leaderboard_file}")
        
        return leaderboard
    
    def run(self, years=None):
        """Run TBX calculation for all nodes"""
        if years is None:
            years = [2021, 2022, 2023, 2024, 2025]
        
        print("⚡ TBX Calculator - ALL SETTLEMENT POINTS")
        print("=" * 60)
        print(f"Configuration:")
        print(f"  Efficiency: {self.efficiency * 100:.0f}%")
        print(f"  Years: {years}")
        print(f"  Data directory: {self.data_dir}")
        print(f"  Output directory: {self.output_dir}")
        
        # Process each year
        for year in years:
            self.process_year(year)
        
        # Create leaderboard
        self.create_leaderboard(years)
        
        print("\n✅ TBX calculation complete for ALL nodes!")

if __name__ == "__main__":
    calculator = TBXCalculatorAllNodes(efficiency=0.9)
    calculator.run(years=[2024, 2025])  # Start with 2 years for testing
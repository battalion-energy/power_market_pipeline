#!/usr/bin/env python3
"""
Direct TBX Calculator - Processes TB2/TB4 spreads without flattening
Loads entire parquet files into memory and iterates by Settlement Point
Supports both Day-Ahead and Real-Time prices
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import pyarrow.parquet as pq
import pyarrow as pa
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class DirectTBXCalculator:
    def __init__(self, 
                 efficiency: float = 0.9,
                 data_dir: Optional[Path] = None,
                 output_dir: Optional[Path] = None):
        """
        Initialize TBX calculator
        
        Args:
            efficiency: Round-trip efficiency (default 0.9 = 90%)
            data_dir: Base directory for ERCOT data
            output_dir: Directory for output files
        """
        self.efficiency = efficiency
        self.data_dir = data_dir or Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/rollup_files")
        self.output_dir = output_dir or Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/tbx_results")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
    def load_prices(self, year: int, price_type: str = "DA") -> pd.DataFrame:
        """
        Load price data for a given year
        
        Args:
            year: Year to process
            price_type: "DA" for Day-Ahead or "RT" for Real-Time
            
        Returns:
            DataFrame with all price data for the year
        """
        if price_type == "DA":
            file_path = self.data_dir / "DA_prices" / f"{year}.parquet"
        else:
            file_path = self.data_dir / "RT_prices" / f"{year}.parquet"
            
        if not file_path.exists():
            print(f"⚠️  {price_type} prices not found for {year} at {file_path}")
            return pd.DataFrame()
            
        print(f"📂 Loading {price_type} prices from {file_path}")
        df = pd.read_parquet(file_path)
        
        # Standardize hour column
        if 'HourEnding' in df.columns and df['HourEnding'].dtype == 'object':
            # Handle "HH:00" format
            df['hour'] = df['HourEnding'].str.replace('24:00', '00:00')
            df['hour'] = pd.to_datetime(df['hour'], format='%H:%M').dt.hour
            df.loc[df['hour'] == 0, 'hour'] = 24
        elif 'hour' not in df.columns:
            df['hour'] = df['HourEnding'].astype(int)
            
        # Create date column for grouping
        df['date'] = pd.to_datetime(df['DeliveryDate']).dt.date
        
        print(f"✅ Loaded {len(df):,} rows for {df['SettlementPoint'].nunique()} settlement points")
        return df
    
    def calculate_daily_arbitrage(self, daily_prices: np.ndarray, hours: int) -> float:
        """
        Calculate arbitrage revenue for a single day
        
        Args:
            daily_prices: Array of 24 hourly prices
            hours: Number of hours for charge/discharge (2 or 4)
            
        Returns:
            Net arbitrage revenue for the day
        """
        if len(daily_prices) != 24 or np.any(np.isnan(daily_prices)):
            return 0.0
        
        # Find best hours to charge (lowest prices) and discharge (highest prices)
        sorted_indices = np.argsort(daily_prices)
        charge_hours = sorted_indices[:hours]
        discharge_hours = sorted_indices[-hours:]
        
        # Calculate costs and revenues
        charge_cost = daily_prices[charge_hours].sum() / self.efficiency
        discharge_revenue = daily_prices[discharge_hours].sum() * self.efficiency
        
        # Net revenue
        return discharge_revenue - charge_cost
    
    def process_settlement_point(self, 
                                sp_data: pd.DataFrame, 
                                settlement_point: str) -> Dict:
        """
        Process all days for a single settlement point
        
        Args:
            sp_data: DataFrame with all data for one settlement point
            settlement_point: Name of the settlement point
            
        Returns:
            Dictionary with TB2 and TB4 results
        """
        # Pivot to get hourly prices by date
        daily_data = sp_data.pivot_table(
            index='date',
            columns='hour',
            values='SettlementPointPrice',
            aggfunc='first'
        )
        
        # Ensure we have all 24 hours
        for hour in range(1, 25):
            if hour not in daily_data.columns:
                daily_data[hour] = np.nan
        
        # Sort columns to ensure hour order
        daily_data = daily_data[[h for h in range(1, 25)]]
        
        # Calculate TB2 and TB4 for each day
        tb2_revenues = []
        tb4_revenues = []
        
        for date, prices in daily_data.iterrows():
            prices_array = prices.values
            if not np.any(np.isnan(prices_array)):
                tb2_revenues.append(self.calculate_daily_arbitrage(prices_array, 2))
                tb4_revenues.append(self.calculate_daily_arbitrage(prices_array, 4))
        
        # Calculate statistics
        results = {
            'settlement_point': settlement_point,
            'num_days': len(tb2_revenues),
            'tb2_daily_avg': np.mean(tb2_revenues) if tb2_revenues else 0,
            'tb2_daily_max': np.max(tb2_revenues) if tb2_revenues else 0,
            'tb2_daily_min': np.min(tb2_revenues) if tb2_revenues else 0,
            'tb2_annual_total': np.sum(tb2_revenues) if tb2_revenues else 0,
            'tb4_daily_avg': np.mean(tb4_revenues) if tb4_revenues else 0,
            'tb4_daily_max': np.max(tb4_revenues) if tb4_revenues else 0,
            'tb4_daily_min': np.min(tb4_revenues) if tb4_revenues else 0,
            'tb4_annual_total': np.sum(tb4_revenues) if tb4_revenues else 0,
        }
        
        return results
    
    def process_year(self, 
                    year: int, 
                    price_type: str = "DA",
                    settlement_points: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Process all settlement points for a given year
        
        Args:
            year: Year to process
            price_type: "DA" or "RT"
            settlement_points: Optional list of specific settlement points to process
            
        Returns:
            DataFrame with results for all settlement points
        """
        print(f"\n{'='*60}")
        print(f"💰 Processing TB2/TB4 for {year} using {price_type} prices")
        print(f"{'='*60}")
        
        # Load all data into memory
        df = self.load_prices(year, price_type)
        if df.empty:
            return pd.DataFrame()
        
        # Get list of settlement points to process
        all_sps = df['SettlementPoint'].unique()
        if settlement_points:
            sps_to_process = [sp for sp in settlement_points if sp in all_sps]
            print(f"📍 Processing {len(sps_to_process)} specified settlement points")
        else:
            sps_to_process = all_sps
            print(f"📍 Processing all {len(sps_to_process)} settlement points")
        
        # Process each settlement point
        results = []
        for i, sp in enumerate(sps_to_process):
            if i % 100 == 0:
                print(f"  Progress: {i}/{len(sps_to_process)} settlement points...")
            
            sp_data = df[df['SettlementPoint'] == sp]
            sp_results = self.process_settlement_point(sp_data, sp)
            sp_results['year'] = year
            sp_results['price_type'] = price_type
            results.append(sp_results)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Sort by TB2 annual total
        results_df = results_df.sort_values('tb2_annual_total', ascending=False)
        
        # Save results
        output_file = self.output_dir / f"tbx_{price_type}_{year}.parquet"
        results_df.to_parquet(output_file)
        print(f"\n💾 Saved results to {output_file}")
        
        # Print top performers
        print(f"\n🏆 Top 10 Settlement Points by TB2 Annual Revenue:")
        print(f"{'Settlement Point':<20} {'TB2 Annual ($)':<15} {'TB4 Annual ($)':<15}")
        print("-" * 50)
        for _, row in results_df.head(10).iterrows():
            print(f"{row['settlement_point']:<20} ${row['tb2_annual_total']:>13,.2f} ${row['tb4_annual_total']:>13,.2f}")
        
        return results_df
    
    def process_multiple_years(self,
                             start_year: int,
                             end_year: int,
                             price_type: str = "DA",
                             settlement_points: Optional[List[str]] = None):
        """
        Process multiple years and create summary
        
        Args:
            start_year: First year to process
            end_year: Last year to process (inclusive)
            price_type: "DA" or "RT"
            settlement_points: Optional list of specific settlement points
        """
        all_results = []
        
        for year in range(start_year, end_year + 1):
            results = self.process_year(year, price_type, settlement_points)
            if not results.empty:
                all_results.append(results)
        
        if all_results:
            # Combine all years
            combined_df = pd.concat(all_results, ignore_index=True)
            
            # Calculate multi-year averages
            summary = combined_df.groupby('settlement_point').agg({
                'tb2_annual_total': ['mean', 'std', 'min', 'max'],
                'tb4_annual_total': ['mean', 'std', 'min', 'max'],
                'num_days': 'mean'
            }).round(2)
            
            # Save summary
            summary_file = self.output_dir / f"tbx_summary_{price_type}_{start_year}_{end_year}.parquet"
            summary.to_parquet(summary_file)
            
            print(f"\n📊 Multi-Year Summary ({start_year}-{end_year}):")
            print(f"{'='*60}")
            print(summary.head(20))
            print(f"\n💾 Summary saved to {summary_file}")

def main():
    """Main function to run TBX calculations"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate TB2/TB4 spreads from ERCOT price data')
    parser.add_argument('--year', type=int, help='Single year to process')
    parser.add_argument('--start-year', type=int, default=2024, help='Start year for multi-year processing')
    parser.add_argument('--end-year', type=int, default=2024, help='End year for multi-year processing')
    parser.add_argument('--price-type', choices=['DA', 'RT'], default='DA', help='Price type to use')
    parser.add_argument('--efficiency', type=float, default=0.9, help='Round-trip efficiency (0-1)')
    parser.add_argument('--settlement-points', nargs='+', help='Specific settlement points to process')
    parser.add_argument('--output-dir', type=str, help='Output directory for results')
    
    args = parser.parse_args()
    
    # Initialize calculator
    output_dir = Path(args.output_dir) if args.output_dir else None
    calculator = DirectTBXCalculator(
        efficiency=args.efficiency,
        output_dir=output_dir
    )
    
    # Process data
    if args.year:
        # Single year
        calculator.process_year(args.year, args.price_type, args.settlement_points)
    else:
        # Multiple years
        calculator.process_multiple_years(
            args.start_year, 
            args.end_year, 
            args.price_type,
            args.settlement_points
        )
    
    print("\n✅ TBX calculation complete!")

if __name__ == "__main__":
    main()
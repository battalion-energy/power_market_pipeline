#!/usr/bin/env python3
"""
Comprehensive TBX Calculator - Computes 6 different TB values:
- TB2_DA: Day-Ahead only (hourly)
- TB2_RT: Real-Time only (15-minute)
- TB2_DART: Combined Day-Ahead (charge) + Real-Time (discharge)
- TB4_DA: Day-Ahead only (hourly)
- TB4_RT: Real-Time only (15-minute)  
- TB4_DART: Combined Day-Ahead (charge) + Real-Time (discharge)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import pyarrow.parquet as pq
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveTBXCalculator:
    def __init__(self, 
                 efficiency: float = 0.9,
                 data_dir: Optional[Path] = None,
                 output_dir: Optional[Path] = None):
        """
        Initialize comprehensive TBX calculator
        
        Args:
            efficiency: Round-trip efficiency (default 0.9 = 90%)
            data_dir: Base directory for ERCOT data
            output_dir: Directory for output files
        """
        self.efficiency = efficiency
        self.data_dir = data_dir or Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/rollup_files")
        self.output_dir = output_dir or Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/tbx_results")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
    def load_da_prices(self, year: int) -> pd.DataFrame:
        """Load Day-Ahead prices (hourly)"""
        file_path = self.data_dir / "DA_prices" / f"{year}.parquet"
        
        if not file_path.exists():
            print(f"⚠️  DA prices not found for {year} at {file_path}")
            return pd.DataFrame()
            
        print(f"📂 Loading DA prices from {file_path}")
        df = pd.read_parquet(file_path)
        
        # Standardize hour column (1-24)
        if 'HourEnding' in df.columns and df['HourEnding'].dtype == 'object':
            df['hour'] = df['HourEnding'].str.replace('24:00', '00:00')
            df['hour'] = pd.to_datetime(df['hour'], format='%H:%M').dt.hour
            df.loc[df['hour'] == 0, 'hour'] = 24
        elif 'hour' not in df.columns:
            df['hour'] = df['HourEnding'].astype(int)
            
        df['date'] = pd.to_datetime(df['DeliveryDate']).dt.date
        
        print(f"✅ Loaded {len(df):,} DA price rows for {df['SettlementPoint'].nunique()} settlement points")
        return df
    
    def load_rt_prices(self, year: int) -> pd.DataFrame:
        """Load Real-Time prices (15-minute intervals)"""
        file_path = self.data_dir / "RT_prices" / f"{year}.parquet"
        
        if not file_path.exists():
            print(f"⚠️  RT prices not found for {year} at {file_path}")
            return pd.DataFrame()
            
        print(f"📂 Loading RT prices from {file_path}")
        df = pd.read_parquet(file_path)
        
        # Convert DeliveryDate to date
        df['date'] = pd.to_datetime(df['DeliveryDate'], format='%m/%d/%Y').dt.date
        
        # Create interval index (0-95 for 96 15-minute intervals per day)
        df['interval_index'] = (df['DeliveryHour'].astype(int) - 1) * 4 + df['DeliveryInterval'].astype(int) - 1
        
        # Rename column for consistency
        df = df.rename(columns={'SettlementPointName': 'SettlementPoint'})
        
        print(f"✅ Loaded {len(df):,} RT price rows for {df['SettlementPoint'].nunique()} settlement points")
        return df
    
    def calculate_tb2_da(self, hourly_prices: np.ndarray) -> float:
        """Calculate TB2 using Day-Ahead hourly prices only"""
        if len(hourly_prices) != 24 or np.any(np.isnan(hourly_prices)):
            return 0.0
        
        # Find 2 best hours to charge and 2 best to discharge
        sorted_indices = np.argsort(hourly_prices)
        charge_hours = sorted_indices[:2]
        discharge_hours = sorted_indices[-2:]
        
        charge_cost = hourly_prices[charge_hours].sum() / self.efficiency
        discharge_revenue = hourly_prices[discharge_hours].sum() * self.efficiency
        
        return discharge_revenue - charge_cost
    
    def calculate_tb4_da(self, hourly_prices: np.ndarray) -> float:
        """Calculate TB4 using Day-Ahead hourly prices only"""
        if len(hourly_prices) != 24 or np.any(np.isnan(hourly_prices)):
            return 0.0
        
        sorted_indices = np.argsort(hourly_prices)
        charge_hours = sorted_indices[:4]
        discharge_hours = sorted_indices[-4:]
        
        charge_cost = hourly_prices[charge_hours].sum() / self.efficiency
        discharge_revenue = hourly_prices[discharge_hours].sum() * self.efficiency
        
        return discharge_revenue - charge_cost
    
    def calculate_tb2_rt(self, interval_prices: np.ndarray) -> float:
        """Calculate TB2 using Real-Time 15-minute prices only"""
        if len(interval_prices) != 96 or np.any(np.isnan(interval_prices)):
            return 0.0
        
        # 2 hours = 8 fifteen-minute intervals
        sorted_indices = np.argsort(interval_prices)
        charge_intervals = sorted_indices[:8]
        discharge_intervals = sorted_indices[-8:]
        
        # Scale by 0.25 since prices are per MWh but intervals are 15 minutes
        charge_cost = (interval_prices[charge_intervals].sum() * 0.25) / self.efficiency
        discharge_revenue = (interval_prices[discharge_intervals].sum() * 0.25) * self.efficiency
        
        return discharge_revenue - charge_cost
    
    def calculate_tb4_rt(self, interval_prices: np.ndarray) -> float:
        """Calculate TB4 using Real-Time 15-minute prices only"""
        if len(interval_prices) != 96 or np.any(np.isnan(interval_prices)):
            return 0.0
        
        # 4 hours = 16 fifteen-minute intervals
        sorted_indices = np.argsort(interval_prices)
        charge_intervals = sorted_indices[:16]
        discharge_intervals = sorted_indices[-16:]
        
        # Scale by 0.25 since prices are per MWh but intervals are 15 minutes
        charge_cost = (interval_prices[charge_intervals].sum() * 0.25) / self.efficiency
        discharge_revenue = (interval_prices[discharge_intervals].sum() * 0.25) * self.efficiency
        
        return discharge_revenue - charge_cost
    
    def calculate_tb2_dart(self, da_hourly: np.ndarray, rt_intervals: np.ndarray) -> float:
        """Calculate TB2 using DA for charging and RT for discharging"""
        if len(da_hourly) != 24 or len(rt_intervals) != 96:
            return 0.0
        if np.any(np.isnan(da_hourly)) or np.any(np.isnan(rt_intervals)):
            return 0.0
        
        # Charge during 2 lowest DA hours
        da_sorted = np.argsort(da_hourly)
        charge_hours = da_sorted[:2]
        charge_cost = da_hourly[charge_hours].sum() / self.efficiency
        
        # Discharge during 8 highest RT intervals (2 hours worth)
        rt_sorted = np.argsort(rt_intervals)
        discharge_intervals = rt_sorted[-8:]
        discharge_revenue = (rt_intervals[discharge_intervals].sum() * 0.25) * self.efficiency
        
        return discharge_revenue - charge_cost
    
    def calculate_tb4_dart(self, da_hourly: np.ndarray, rt_intervals: np.ndarray) -> float:
        """Calculate TB4 using DA for charging and RT for discharging"""
        if len(da_hourly) != 24 or len(rt_intervals) != 96:
            return 0.0
        if np.any(np.isnan(da_hourly)) or np.any(np.isnan(rt_intervals)):
            return 0.0
        
        # Charge during 4 lowest DA hours
        da_sorted = np.argsort(da_hourly)
        charge_hours = da_sorted[:4]
        charge_cost = da_hourly[charge_hours].sum() / self.efficiency
        
        # Discharge during 16 highest RT intervals (4 hours worth)
        rt_sorted = np.argsort(rt_intervals)
        discharge_intervals = rt_sorted[-16:]
        discharge_revenue = (rt_intervals[discharge_intervals].sum() * 0.25) * self.efficiency
        
        return discharge_revenue - charge_cost
    
    def process_settlement_point(self, 
                                sp: str,
                                da_sp_data: pd.DataFrame,
                                rt_sp_data: pd.DataFrame) -> Dict:
        """Process all TB calculations for a single settlement point"""
        
        # Get unique dates present in both datasets
        da_dates = set(da_sp_data['date'].unique()) if not da_sp_data.empty else set()
        rt_dates = set(rt_sp_data['date'].unique()) if not rt_sp_data.empty else set()
        common_dates = sorted(da_dates & rt_dates)
        
        results = {
            'settlement_point': sp,
            'num_days': len(common_dates),
            'tb2_da_daily_avg': 0,
            'tb2_da_annual_total': 0,
            'tb2_rt_daily_avg': 0,
            'tb2_rt_annual_total': 0,
            'tb2_dart_daily_avg': 0,
            'tb2_dart_annual_total': 0,
            'tb4_da_daily_avg': 0,
            'tb4_da_annual_total': 0,
            'tb4_rt_daily_avg': 0,
            'tb4_rt_annual_total': 0,
            'tb4_dart_daily_avg': 0,
            'tb4_dart_annual_total': 0,
        }
        
        if not common_dates:
            return results
        
        # Prepare DA data (hourly)
        da_pivot = da_sp_data.pivot_table(
            index='date',
            columns='hour',
            values='SettlementPointPrice',
            aggfunc='first'
        )
        
        # Ensure all 24 hours
        for hour in range(1, 25):
            if hour not in da_pivot.columns:
                da_pivot[hour] = np.nan
        da_pivot = da_pivot[[h for h in range(1, 25)]]
        
        # Prepare RT data (15-minute intervals)
        rt_pivot = rt_sp_data.pivot_table(
            index='date',
            columns='interval_index',
            values='SettlementPointPrice',
            aggfunc='first'
        )
        
        # Ensure all 96 intervals
        for interval in range(96):
            if interval not in rt_pivot.columns:
                rt_pivot[interval] = np.nan
        rt_pivot = rt_pivot[[i for i in range(96)]]
        
        # Calculate TB values for each day
        tb2_da_values = []
        tb2_rt_values = []
        tb2_dart_values = []
        tb4_da_values = []
        tb4_rt_values = []
        tb4_dart_values = []
        
        for date in common_dates:
            # Get DA hourly prices
            if date in da_pivot.index:
                da_prices = da_pivot.loc[date].values
            else:
                da_prices = np.full(24, np.nan)
            
            # Get RT interval prices
            if date in rt_pivot.index:
                rt_prices = rt_pivot.loc[date].values
            else:
                rt_prices = np.full(96, np.nan)
            
            # Calculate all TB values
            if not np.any(np.isnan(da_prices)):
                tb2_da_values.append(self.calculate_tb2_da(da_prices))
                tb4_da_values.append(self.calculate_tb4_da(da_prices))
            
            if not np.any(np.isnan(rt_prices)):
                tb2_rt_values.append(self.calculate_tb2_rt(rt_prices))
                tb4_rt_values.append(self.calculate_tb4_rt(rt_prices))
            
            if not np.any(np.isnan(da_prices)) and not np.any(np.isnan(rt_prices)):
                tb2_dart_values.append(self.calculate_tb2_dart(da_prices, rt_prices))
                tb4_dart_values.append(self.calculate_tb4_dart(da_prices, rt_prices))
        
        # Calculate statistics
        if tb2_da_values:
            results['tb2_da_daily_avg'] = np.mean(tb2_da_values)
            results['tb2_da_annual_total'] = np.sum(tb2_da_values)
        
        if tb2_rt_values:
            results['tb2_rt_daily_avg'] = np.mean(tb2_rt_values)
            results['tb2_rt_annual_total'] = np.sum(tb2_rt_values)
        
        if tb2_dart_values:
            results['tb2_dart_daily_avg'] = np.mean(tb2_dart_values)
            results['tb2_dart_annual_total'] = np.sum(tb2_dart_values)
        
        if tb4_da_values:
            results['tb4_da_daily_avg'] = np.mean(tb4_da_values)
            results['tb4_da_annual_total'] = np.sum(tb4_da_values)
        
        if tb4_rt_values:
            results['tb4_rt_daily_avg'] = np.mean(tb4_rt_values)
            results['tb4_rt_annual_total'] = np.sum(tb4_rt_values)
        
        if tb4_dart_values:
            results['tb4_dart_daily_avg'] = np.mean(tb4_dart_values)
            results['tb4_dart_annual_total'] = np.sum(tb4_dart_values)
        
        return results
    
    def process_year(self, 
                    year: int,
                    settlement_points: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Process all settlement points for a given year
        
        Args:
            year: Year to process
            settlement_points: Optional list of specific settlement points
            
        Returns:
            DataFrame with all 6 TB calculations for each settlement point
        """
        print(f"\n{'='*80}")
        print(f"💰 Processing Comprehensive TB2/TB4 for {year}")
        print(f"   Computing 6 variants: TB2_DA, TB2_RT, TB2_DART, TB4_DA, TB4_RT, TB4_DART")
        print(f"{'='*80}")
        
        # Load both DA and RT data
        da_df = self.load_da_prices(year)
        rt_df = self.load_rt_prices(year)
        
        if da_df.empty and rt_df.empty:
            print("❌ No data available for processing")
            return pd.DataFrame()
        
        # Get settlement points
        da_sps = set(da_df['SettlementPoint'].unique()) if not da_df.empty else set()
        rt_sps = set(rt_df['SettlementPoint'].unique()) if not rt_df.empty else set()
        all_sps = da_sps | rt_sps
        
        if settlement_points:
            sps_to_process = [sp for sp in settlement_points if sp in all_sps]
            print(f"📍 Processing {len(sps_to_process)} specified settlement points")
        else:
            sps_to_process = sorted(all_sps)
            print(f"📍 Processing all {len(sps_to_process)} settlement points")
        
        # Process each settlement point
        results = []
        for i, sp in enumerate(sps_to_process):
            if i % 100 == 0:
                print(f"  Progress: {i}/{len(sps_to_process)} settlement points...")
            
            da_sp_data = da_df[da_df['SettlementPoint'] == sp] if not da_df.empty else pd.DataFrame()
            rt_sp_data = rt_df[rt_df['SettlementPoint'] == sp] if not rt_df.empty else pd.DataFrame()
            
            sp_results = self.process_settlement_point(sp, da_sp_data, rt_sp_data)
            sp_results['year'] = year
            results.append(sp_results)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Sort by TB2_DART (combined strategy) as primary metric
        results_df = results_df.sort_values('tb2_dart_annual_total', ascending=False)
        
        # Save results
        output_file = self.output_dir / f"tbx_comprehensive_{year}.parquet"
        results_df.to_parquet(output_file)
        print(f"\n💾 Saved results to {output_file}")
        
        # Print top performers
        print(f"\n🏆 Top 10 Settlement Points by TB2_DART Annual Revenue:")
        print(f"{'Settlement Point':<20} {'TB2_DA':<12} {'TB2_RT':<12} {'TB2_DART':<12} {'TB4_DART':<12}")
        print("-" * 80)
        for _, row in results_df.head(10).iterrows():
            print(f"{row['settlement_point']:<20} "
                  f"${row['tb2_da_annual_total']:>10,.0f} "
                  f"${row['tb2_rt_annual_total']:>10,.0f} "
                  f"${row['tb2_dart_annual_total']:>10,.0f} "
                  f"${row['tb4_dart_annual_total']:>10,.0f}")
        
        # Summary statistics
        print(f"\n📊 Summary Statistics (Annual Totals):")
        print(f"  TB2_DA:   Mean=${results_df['tb2_da_annual_total'].mean():,.0f}, "
              f"Max=${results_df['tb2_da_annual_total'].max():,.0f}")
        print(f"  TB2_RT:   Mean=${results_df['tb2_rt_annual_total'].mean():,.0f}, "
              f"Max=${results_df['tb2_rt_annual_total'].max():,.0f}")
        print(f"  TB2_DART: Mean=${results_df['tb2_dart_annual_total'].mean():,.0f}, "
              f"Max=${results_df['tb2_dart_annual_total'].max():,.0f}")
        print(f"  TB4_DA:   Mean=${results_df['tb4_da_annual_total'].mean():,.0f}, "
              f"Max=${results_df['tb4_da_annual_total'].max():,.0f}")
        print(f"  TB4_RT:   Mean=${results_df['tb4_rt_annual_total'].mean():,.0f}, "
              f"Max=${results_df['tb4_rt_annual_total'].max():,.0f}")
        print(f"  TB4_DART: Mean=${results_df['tb4_dart_annual_total'].mean():,.0f}, "
              f"Max=${results_df['tb4_dart_annual_total'].max():,.0f}")
        
        return results_df

def main():
    """Main function to run comprehensive TBX calculations"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate comprehensive TB2/TB4 spreads (6 variants)')
    parser.add_argument('--year', type=int, default=2024, help='Year to process')
    parser.add_argument('--efficiency', type=float, default=0.9, help='Round-trip efficiency (0-1)')
    parser.add_argument('--settlement-points', nargs='+', help='Specific settlement points to process')
    parser.add_argument('--output-dir', type=str, help='Output directory for results')
    
    args = parser.parse_args()
    
    # Initialize calculator
    output_dir = Path(args.output_dir) if args.output_dir else None
    calculator = ComprehensiveTBXCalculator(
        efficiency=args.efficiency,
        output_dir=output_dir
    )
    
    # Process data
    calculator.process_year(args.year, args.settlement_points)
    
    print("\n✅ Comprehensive TBX calculation complete!")

if __name__ == "__main__":
    main()
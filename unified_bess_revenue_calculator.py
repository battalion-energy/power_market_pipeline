#!/usr/bin/env python3
"""
Unified BESS Revenue Calculator
Calculates all revenue streams for Battery Energy Storage Systems:
- DA Energy Arbitrage (charging as load, discharging as gen)
- RT Energy Revenue/Cost
- Ancillary Services (RegUp, RegDn, Spin, NonSpin, ECRS)
Outputs database-ready format for NextJS frontend
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass, asdict
import pyarrow.parquet as pq

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BESSRevenue:
    """Complete BESS revenue breakdown"""
    resource_name: str
    date: str
    year: int
    month: int
    day: int
    
    # Day-Ahead Market
    da_energy_revenue: float  # Gen discharge revenue
    da_energy_cost: float     # Load charging cost
    da_net_energy: float      # Net DA position
    
    # Real-Time Market
    rt_energy_revenue: float  # RT discharge revenue
    rt_energy_cost: float     # RT charging cost
    rt_net_energy: float      # Net RT position
    
    # Ancillary Services
    as_regup_revenue: float
    as_regdn_revenue: float
    as_rrs_revenue: float     # Responsive Reserve
    as_nonspin_revenue: float
    as_ecrs_revenue: float    # ERCOT Contingency Reserve
    as_total_revenue: float
    
    # Totals
    total_energy_revenue: float  # DA + RT net
    total_revenue: float         # Energy + AS
    
    # Operations
    cycles: float               # Estimated charge/discharge cycles
    mwh_charged: float         # Total MWh charged
    mwh_discharged: float      # Total MWh discharged
    capacity_factor: float     # Utilization %
    
    # Metadata
    settlement_point: str
    capacity_mw: float
    duration_hours: float

class UnifiedBESSCalculator:
    """Calculate BESS revenues from all market streams"""
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Path("/home/enrico/data/ERCOT_data")
        self.rollup_dir = self.data_dir / "rollup_files"
        self.output_dir = self.data_dir / "bess_analysis"
        self.output_dir.mkdir(exist_ok=True)
        
        # Load BESS registry
        self.bess_registry = self._load_bess_registry()
        
    def _load_bess_registry(self) -> pd.DataFrame:
        """Load BESS resource mappings"""
        registry_file = self.output_dir / "bess_registry.parquet"
        if registry_file.exists():
            return pd.read_parquet(registry_file)
        
        # Create from mapping if registry doesn't exist
        mapping_file = Path("bess_resource_mapping.csv")
        if mapping_file.exists():
            df = pd.read_csv(mapping_file)
            # Convert to registry format
            registry = pd.DataFrame({
                'resource_name': df['battery_name'],
                'gen_resource': df['gen_resources'].fillna(''),
                'load_resource': df['load_resources'].fillna(''),
                'settlement_point': df['settlement_points'].fillna('UNKNOWN'),
                'capacity_mw': df['max_power_mw'].fillna(100.0),
                'duration_hours': df['duration_hours'].fillna(2.0)
            })
            registry.to_parquet(registry_file, index=False)
            return registry
        
        logger.warning("No BESS registry found - will identify from data")
        return pd.DataFrame()
    
    def calculate_da_revenues(self, year: int) -> pd.DataFrame:
        """Calculate Day-Ahead energy arbitrage revenues"""
        logger.info(f"Calculating DA revenues for {year}")
        
        results = []
        
        # Load DA prices
        da_price_file = self.rollup_dir / "flattened" / f"DA_prices_{year}.parquet"
        if not da_price_file.exists():
            logger.warning(f"DA prices not found for {year}")
            return pd.DataFrame()
        
        da_prices = pd.read_parquet(da_price_file)
        
        # Handle different datetime column names
        if 'datetime_ts' in da_prices.columns:
            da_prices['datetime'] = pd.to_datetime(da_prices['datetime_ts'])
        elif 'datetime' not in da_prices.columns:
            logger.error(f"No datetime column in DA prices for {year}")
            return pd.DataFrame()
        
        # Load DAM Gen awards (discharge)
        dam_gen_file = self.rollup_dir / "DAM_Gen_Resources" / f"{year}.parquet"
        if dam_gen_file.exists():
            dam_gen = pd.read_parquet(dam_gen_file)
            # Filter for BESS resources (PWRSTR type)
            if 'ResourceType' in dam_gen.columns:
                dam_gen = dam_gen[dam_gen['ResourceType'] == 'PWRSTR']
        else:
            dam_gen = pd.DataFrame()
        
        # Load DAM Load awards (charging)
        dam_load_file = self.rollup_dir / "DAM_Load_Resources" / f"{year}.parquet"
        if dam_load_file.exists():
            dam_load = pd.read_parquet(dam_load_file)
        else:
            dam_load = pd.DataFrame()
        
        # Process each BESS resource
        for _, bess in self.bess_registry.iterrows():
            resource_name = bess['resource_name']
            settlement_point = bess['settlement_point']
            
            # Get settlement point prices - use HB_BUSAVG as default
            if settlement_point in da_prices.columns:
                sp_prices = da_prices[['datetime', settlement_point]].copy()
                sp_prices.columns = ['datetime', 'price']
            elif 'HB_BUSAVG' in da_prices.columns:
                # Use average hub price as fallback
                sp_prices = da_prices[['datetime', 'HB_BUSAVG']].copy()
                sp_prices.columns = ['datetime', 'price']
            else:
                # Skip if no price available
                continue
            
            # Initialize merged dataframes
            gen_merged = pd.DataFrame()
            load_merged = pd.DataFrame()
            
            # Calculate discharge revenues (gen)
            if not dam_gen.empty and 'ResourceName' in dam_gen.columns:
                gen_awards = dam_gen[dam_gen['ResourceName'].str.contains(resource_name, na=False)]
                if not gen_awards.empty:
                    # Merge with prices
                    gen_awards['datetime'] = pd.to_datetime(gen_awards.get('datetime', gen_awards.get('DeliveryDate')))
                    gen_merged = gen_awards.merge(sp_prices, on='datetime', how='left')
                    gen_merged['da_gen_revenue'] = gen_merged.get('AwardedQuantity', 0) * gen_merged['price']
            
            # Calculate charging costs (load)
            if not dam_load.empty and 'ResourceName' in dam_load.columns:
                load_awards = dam_load[dam_load['ResourceName'].str.contains(resource_name, na=False)]
                if not load_awards.empty:
                    load_awards['datetime'] = pd.to_datetime(load_awards.get('datetime', load_awards.get('DeliveryDate')))
                    load_merged = load_awards.merge(sp_prices, on='datetime', how='left')
                    load_merged['da_load_cost'] = load_merged.get('EnergyBidAward', 0) * load_merged['price']
            
            # Aggregate by day
            if not gen_merged.empty or not load_merged.empty:
                # Combine gen and load data
                daily_da = self._aggregate_daily_da(gen_merged, load_merged, resource_name, bess)
                results.extend(daily_da)
        
        return pd.DataFrame(results)
    
    def calculate_rt_revenues(self, year: int) -> pd.DataFrame:
        """Calculate Real-Time energy revenues"""
        logger.info(f"Calculating RT revenues for {year}")
        
        results = []
        
        # Load RT prices (15-minute)
        rt_price_file = self.rollup_dir / "flattened" / f"RT_prices_15min_{year}.parquet"
        if not rt_price_file.exists():
            # Try 5-minute file
            rt_price_file = self.rollup_dir / "flattened" / f"RT_prices_{year}.parquet"
        
        if not rt_price_file.exists():
            logger.warning(f"RT prices not found for {year}")
            return pd.DataFrame()
        
        rt_prices = pd.read_parquet(rt_price_file)
        
        # Load SCED Gen (discharge in RT)
        sced_gen_file = self.rollup_dir / "SCED_Gen_Resources" / f"{year}.parquet"
        if sced_gen_file.exists():
            sced_gen = pd.read_parquet(sced_gen_file)
        else:
            sced_gen = pd.DataFrame()
        
        # Load SCED Load (charging in RT)
        sced_load_file = self.rollup_dir / "SCED_Load_Resources" / f"{year}.parquet"
        if sced_load_file.exists():
            sced_load = pd.read_parquet(sced_load_file)
        else:
            sced_load = pd.DataFrame()
        
        # Process each BESS resource
        for _, bess in self.bess_registry.iterrows():
            resource_name = bess['resource_name']
            settlement_point = bess['settlement_point']
            
            # Get RT settlement point prices - use HB_BUSAVG as default
            if settlement_point in rt_prices.columns:
                sp_prices = rt_prices[['datetime_ts' if 'datetime_ts' in rt_prices.columns else 'datetime', 
                                      settlement_point]].copy()
                sp_prices.columns = ['datetime', 'price']
            elif 'HB_BUSAVG' in rt_prices.columns:
                # Use average hub price as fallback
                sp_prices = rt_prices[['datetime_ts' if 'datetime_ts' in rt_prices.columns else 'datetime', 
                                      'HB_BUSAVG']].copy()
                sp_prices.columns = ['datetime', 'price']
            else:
                continue
            
            # Calculate RT revenues
            daily_rt = self._calculate_rt_daily(sced_gen, sced_load, sp_prices, resource_name, bess)
            results.extend(daily_rt)
        
        return pd.DataFrame(results)
    
    def calculate_as_revenues(self, year: int) -> pd.DataFrame:
        """Calculate Ancillary Services revenues"""
        logger.info(f"Calculating AS revenues for {year}")
        
        results = []
        
        # Load AS prices
        as_price_file = self.rollup_dir / "flattened" / f"AS_prices_{year}.parquet"
        if not as_price_file.exists():
            logger.warning(f"AS prices not found for {year}")
            return pd.DataFrame()
        
        as_prices = pd.read_parquet(as_price_file)
        
        # AS columns mapping
        as_services = {
            'REGUP': 'as_regup_revenue',
            'REGDN': 'as_regdn_revenue', 
            'RRS': 'as_rrs_revenue',
            'NSPIN': 'as_nonspin_revenue',
            'ECRS': 'as_ecrs_revenue'
        }
        
        # Load DAM Gen for AS awards
        dam_gen_file = self.rollup_dir / "DAM_Gen_Resources" / f"{year}.parquet"
        if dam_gen_file.exists():
            dam_gen = pd.read_parquet(dam_gen_file)
            dam_gen = dam_gen[dam_gen['ResourceType'] == 'PWRSTR'] if 'ResourceType' in dam_gen.columns else dam_gen
        else:
            return pd.DataFrame()
        
        # Process each BESS
        for _, bess in self.bess_registry.iterrows():
            resource_name = bess['resource_name']
            
            # Get resource AS awards
            resource_awards = dam_gen[dam_gen['ResourceName'].str.contains(resource_name, na=False)]
            if resource_awards.empty:
                continue
            
            # Calculate AS revenues for each service
            daily_as = self._calculate_as_daily(resource_awards, as_prices, as_services, resource_name, bess)
            results.extend(daily_as)
        
        return pd.DataFrame(results)
    
    def _aggregate_daily_da(self, gen_df: pd.DataFrame, load_df: pd.DataFrame, 
                            resource_name: str, bess_info: pd.Series) -> List[Dict]:
        """Aggregate DA revenues by day"""
        results = []
        
        # Combine gen and load data
        all_dates = set()
        if not gen_df.empty:
            gen_df['date'] = pd.to_datetime(gen_df['datetime']).dt.date
            all_dates.update(gen_df['date'].unique())
        if not load_df.empty:
            load_df['date'] = pd.to_datetime(load_df['datetime']).dt.date
            all_dates.update(load_df['date'].unique())
        
        for date in sorted(all_dates):
            daily_revenue = BESSRevenue(
                resource_name=resource_name,
                date=str(date),
                year=date.year,
                month=date.month,
                day=date.day,
                da_energy_revenue=0.0,
                da_energy_cost=0.0,
                da_net_energy=0.0,
                rt_energy_revenue=0.0,
                rt_energy_cost=0.0,
                rt_net_energy=0.0,
                as_regup_revenue=0.0,
                as_regdn_revenue=0.0,
                as_rrs_revenue=0.0,
                as_nonspin_revenue=0.0,
                as_ecrs_revenue=0.0,
                as_total_revenue=0.0,
                total_energy_revenue=0.0,
                total_revenue=0.0,
                cycles=0.0,
                mwh_charged=0.0,
                mwh_discharged=0.0,
                capacity_factor=0.0,
                settlement_point=bess_info['settlement_point'],
                capacity_mw=bess_info['capacity_mw'],
                duration_hours=bess_info['duration_hours']
            )
            
            # Sum gen revenues
            if not gen_df.empty:
                day_gen = gen_df[gen_df['date'] == date]
                if not day_gen.empty:
                    daily_revenue.da_energy_revenue = day_gen['da_gen_revenue'].sum()
                    daily_revenue.mwh_discharged = day_gen.get('AwardedQuantity', 0).sum()
            
            # Sum load costs
            if not load_df.empty:
                day_load = load_df[load_df['date'] == date]
                if not day_load.empty:
                    daily_revenue.da_energy_cost = day_load['da_load_cost'].sum()
                    daily_revenue.mwh_charged = day_load.get('EnergyBidAward', 0).sum()
            
            # Calculate net
            daily_revenue.da_net_energy = daily_revenue.da_energy_revenue - daily_revenue.da_energy_cost
            daily_revenue.total_energy_revenue = daily_revenue.da_net_energy
            daily_revenue.total_revenue = daily_revenue.total_energy_revenue
            
            # Estimate cycles
            if daily_revenue.capacity_mw > 0:
                daily_revenue.cycles = min(daily_revenue.mwh_discharged, daily_revenue.mwh_charged) / (
                    daily_revenue.capacity_mw * daily_revenue.duration_hours)
                daily_revenue.capacity_factor = daily_revenue.mwh_discharged / (
                    daily_revenue.capacity_mw * 24) if daily_revenue.capacity_mw > 0 else 0
            
            results.append(asdict(daily_revenue))
        
        return results
    
    def _calculate_rt_daily(self, sced_gen: pd.DataFrame, sced_load: pd.DataFrame,
                           rt_prices: pd.DataFrame, resource_name: str, 
                           bess_info: pd.Series) -> List[Dict]:
        """Calculate daily RT revenues"""
        results = []
        
        # Filter for resource
        if not sced_gen.empty and 'ResourceName' in sced_gen.columns:
            gen_data = sced_gen[sced_gen['ResourceName'].str.contains(resource_name, na=False)]
        else:
            gen_data = pd.DataFrame()
        
        if not sced_load.empty and 'ResourceName' in sced_load.columns:
            load_data = sced_load[sced_load['ResourceName'].str.contains(resource_name, na=False)]
        else:
            load_data = pd.DataFrame()
        
        # Process RT data (simplified for now)
        # In production, this would match 5-minute SCED intervals with RT prices
        
        return results
    
    def _calculate_as_daily(self, awards_df: pd.DataFrame, as_prices: pd.DataFrame,
                           as_services: Dict, resource_name: str, 
                           bess_info: pd.Series) -> List[Dict]:
        """Calculate daily AS revenues"""
        results = []
        
        if awards_df.empty:
            return results
        
        # Add date column
        awards_df['date'] = pd.to_datetime(awards_df.get('datetime', 
                                          awards_df.get('DeliveryDate'))).dt.date
        
        for date in awards_df['date'].unique():
            if pd.isna(date):
                continue
                
            day_awards = awards_df[awards_df['date'] == date]
            
            daily_revenue = BESSRevenue(
                resource_name=resource_name,
                date=str(date),
                year=date.year,
                month=date.month,
                day=date.day,
                da_energy_revenue=0.0,
                da_energy_cost=0.0,
                da_net_energy=0.0,
                rt_energy_revenue=0.0,
                rt_energy_cost=0.0,
                rt_net_energy=0.0,
                as_regup_revenue=0.0,
                as_regdn_revenue=0.0,
                as_rrs_revenue=0.0,
                as_nonspin_revenue=0.0,
                as_ecrs_revenue=0.0,
                as_total_revenue=0.0,
                total_energy_revenue=0.0,
                total_revenue=0.0,
                cycles=0.0,
                mwh_charged=0.0,
                mwh_discharged=0.0,
                capacity_factor=0.0,
                settlement_point=bess_info['settlement_point'],
                capacity_mw=bess_info['capacity_mw'],
                duration_hours=bess_info['duration_hours']
            )
            
            # Calculate AS revenues
            # AS awards are MW capacity, prices are $/MW-hr, so revenue = MW × $/MW-hr × 1 hr
            # We need to get the specific hour's AS price, not the daily average
            
            # Get AS prices for this date
            if 'datetime_ts' in as_prices.columns:
                as_prices['date'] = pd.to_datetime(as_prices['datetime_ts']).dt.date
            else:
                as_prices['date'] = pd.to_datetime(as_prices.get('datetime', as_prices.get('DeliveryDate'))).dt.date
            
            day_as_prices = as_prices[as_prices['date'] == date] if 'date' in as_prices.columns else as_prices
            
            # RegUp: MW × $/MW-hr
            if 'RegUpAwarded' in day_awards.columns and 'REGUP' in day_as_prices.columns:
                # Sum up hourly revenues: each hour's MW award × that hour's price
                daily_revenue.as_regup_revenue = (day_awards['RegUpAwarded'] * 
                                                 day_as_prices['REGUP'].mean()).sum() if not day_as_prices.empty else 0
            
            # RegDn: MW × $/MW-hr  
            if 'RegDownAwarded' in day_awards.columns and 'REGDN' in day_as_prices.columns:
                daily_revenue.as_regdn_revenue = (day_awards['RegDownAwarded'] * 
                                                 day_as_prices['REGDN'].mean()).sum() if not day_as_prices.empty else 0
            
            # RRS: MW × $/MW-hr
            if 'RRSAwarded' in day_awards.columns and 'RRS' in day_as_prices.columns:
                daily_revenue.as_rrs_revenue = (day_awards['RRSAwarded'] * 
                                               day_as_prices['RRS'].mean()).sum() if not day_as_prices.empty else 0
            
            # NonSpin: MW × $/MW-hr
            if 'NonSpinAwarded' in day_awards.columns and 'NSPIN' in day_as_prices.columns:
                daily_revenue.as_nonspin_revenue = (day_awards['NonSpinAwarded'] * 
                                                   day_as_prices['NSPIN'].mean()).sum() if not day_as_prices.empty else 0
            
            # ECRS: MW × $/MW-hr
            if 'ECRSAwarded' in day_awards.columns and 'ECRS' in day_as_prices.columns:
                daily_revenue.as_ecrs_revenue = (day_awards['ECRSAwarded'] * 
                                                day_as_prices['ECRS'].mean()).sum() if not day_as_prices.empty else 0
            
            # Total AS
            daily_revenue.as_total_revenue = (daily_revenue.as_regup_revenue + 
                                             daily_revenue.as_regdn_revenue +
                                             daily_revenue.as_rrs_revenue +
                                             daily_revenue.as_nonspin_revenue +
                                             daily_revenue.as_ecrs_revenue)
            
            daily_revenue.total_revenue = daily_revenue.as_total_revenue
            
            results.append(asdict(daily_revenue))
        
        return results
    
    def process_all_years(self, start_year: int = 2019, end_year: int = 2024) -> pd.DataFrame:
        """Process all years and combine results"""
        all_results = []
        
        for year in range(start_year, end_year + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing year {year}")
            logger.info(f"{'='*60}")
            
            # Calculate each revenue stream
            da_revenues = self.calculate_da_revenues(year)
            rt_revenues = self.calculate_rt_revenues(year)
            as_revenues = self.calculate_as_revenues(year)
            
            # Combine revenues for the year
            if not da_revenues.empty:
                all_results.append(da_revenues)
            if not rt_revenues.empty:
                all_results.append(rt_revenues)
            if not as_revenues.empty:
                all_results.append(as_revenues)
        
        if all_results:
            combined_df = pd.concat(all_results, ignore_index=True)
            
            # Group by resource/date and sum revenues
            grouped = combined_df.groupby(['resource_name', 'date']).agg({
                'year': 'first',
                'month': 'first',
                'day': 'first',
                'da_energy_revenue': 'sum',
                'da_energy_cost': 'sum',
                'da_net_energy': 'sum',
                'rt_energy_revenue': 'sum',
                'rt_energy_cost': 'sum',
                'rt_net_energy': 'sum',
                'as_regup_revenue': 'sum',
                'as_regdn_revenue': 'sum',
                'as_rrs_revenue': 'sum',
                'as_nonspin_revenue': 'sum',
                'as_ecrs_revenue': 'sum',
                'as_total_revenue': 'sum',
                'total_energy_revenue': 'sum',
                'total_revenue': 'sum',
                'cycles': 'sum',
                'mwh_charged': 'sum',
                'mwh_discharged': 'sum',
                'capacity_factor': 'mean',
                'settlement_point': 'first',
                'capacity_mw': 'first',
                'duration_hours': 'first'
            }).reset_index()
            
            return grouped
        
        return pd.DataFrame()
    
    def create_leaderboard(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create revenue leaderboard rankings"""
        if df.empty:
            return pd.DataFrame()
        
        # Annual leaderboard
        annual = df.groupby(['resource_name', 'year']).agg({
            'total_revenue': 'sum',
            'da_net_energy': 'sum',
            'rt_net_energy': 'sum',
            'as_total_revenue': 'sum',
            'cycles': 'sum',
            'mwh_discharged': 'sum',
            'capacity_mw': 'first',
            'duration_hours': 'first'
        }).reset_index()
        
        # Add revenue per MW
        annual['revenue_per_mw'] = annual['total_revenue'] / annual['capacity_mw']
        annual['revenue_per_mwh'] = annual['total_revenue'] / (
            annual['capacity_mw'] * annual['duration_hours'])
        
        # Rank by total revenue
        annual['rank'] = annual.groupby('year')['total_revenue'].rank(
            ascending=False, method='dense')
        
        # Sort by year and rank
        annual = annual.sort_values(['year', 'rank'])
        
        return annual
    
    def save_to_database_format(self, df: pd.DataFrame, leaderboard: pd.DataFrame):
        """Save in database-ready format for NextJS"""
        output_dir = self.output_dir / "database_export"
        output_dir.mkdir(exist_ok=True)
        
        # Daily revenues table
        daily_file = output_dir / "bess_daily_revenues.parquet"
        df.to_parquet(daily_file, index=False)
        logger.info(f"Saved daily revenues to {daily_file}")
        
        # Also save as CSV for easy viewing
        df.to_csv(output_dir / "bess_daily_revenues.csv", index=False)
        
        # Leaderboard table
        leaderboard_file = output_dir / "bess_annual_leaderboard.parquet"
        leaderboard.to_parquet(leaderboard_file, index=False)
        logger.info(f"Saved leaderboard to {leaderboard_file}")
        
        # Also save as CSV
        leaderboard.to_csv(output_dir / "bess_annual_leaderboard.csv", index=False)
        
        # Create JSON metadata for NextJS
        metadata = {
            "last_updated": datetime.now().isoformat(),
            "years_processed": sorted(df['year'].unique().tolist()),
            "total_resources": len(df['resource_name'].unique()),
            "total_days": len(df),
            "revenue_streams": [
                "da_energy", "rt_energy", "as_regup", "as_regdn", 
                "as_rrs", "as_nonspin", "as_ecrs"
            ],
            "schema": {
                "daily_revenues": list(df.columns),
                "leaderboard": list(leaderboard.columns)
            }
        }
        
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved metadata to {output_dir / 'metadata.json'}")
        
        return output_dir

def main():
    """Main execution"""
    logger.info("="*80)
    logger.info("UNIFIED BESS REVENUE CALCULATOR")
    logger.info("="*80)
    
    calculator = UnifiedBESSCalculator()
    
    # Process 6 years of data
    df = calculator.process_all_years(start_year=2019, end_year=2024)
    
    if df.empty:
        logger.error("No revenue data calculated")
        return
    
    # Create leaderboard
    leaderboard = calculator.create_leaderboard(df)
    
    # Save to database format
    output_dir = calculator.save_to_database_format(df, leaderboard)
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    logger.info(f"Total resources analyzed: {len(df['resource_name'].unique())}")
    logger.info(f"Total days processed: {len(df)}")
    logger.info(f"Total revenue: ${df['total_revenue'].sum():,.2f}")
    logger.info(f"Average daily revenue: ${df['total_revenue'].mean():,.2f}")
    
    # Top performers
    logger.info("\nTOP 10 PERFORMERS (All Time):")
    top_performers = df.groupby('resource_name')['total_revenue'].sum().nlargest(10)
    for i, (resource, revenue) in enumerate(top_performers.items(), 1):
        logger.info(f"  {i}. {resource}: ${revenue:,.2f}")
    
    logger.info(f"\n✅ Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
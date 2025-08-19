#!/usr/bin/env python3
"""
BESS Revenue Calculator - Parquet-based Implementation
Phase 2: Revenue Calculation using Parquet files
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, date, timedelta
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Base paths
BASE_DATA_DIR = Path("/home/enrico/data/ERCOT_data")
ROLLUP_DIR = BASE_DATA_DIR / "rollup_files"
OUTPUT_DIR = BASE_DATA_DIR / "bess_analysis"

@dataclass
class RevenueResult:
    """Revenue calculation result for a BESS resource"""
    resource_name: str
    period_start: date
    period_end: date
    energy_revenue_dam: float
    energy_revenue_rt: float
    as_revenue_regup: float
    as_revenue_regdown: float
    as_revenue_rrs: float
    as_revenue_ecrs: float
    as_revenue_nspin: float
    total_energy_revenue: float
    total_as_revenue: float
    total_revenue: float
    hours_operated: int
    avg_soc: float = 0.0
    capacity_factor: float = 0.0

class BESSRevenueCalculator:
    """Calculate BESS revenues using Parquet files"""
    
    def __init__(self, rollup_dir: Path):
        self.rollup_dir = rollup_dir
        self.bess_registry = self._load_bess_registry()
        
    def _load_bess_registry(self) -> pd.DataFrame:
        """Load BESS registry from Phase 1"""
        registry_path = OUTPUT_DIR / "bess_registry.parquet"
        if registry_path.exists():
            return pd.read_parquet(registry_path)
        else:
            logger.warning("BESS registry not found. Run Phase 1 first.")
            return pd.DataFrame()
    
    @lru_cache(maxsize=128)
    def _load_price_data(self, price_type: str, year: int) -> pd.DataFrame:
        """Load and cache price data for a year"""
        if price_type == 'DA':
            file_path = self.rollup_dir / 'DA_prices' / f"{year}.parquet"
        elif price_type == 'RT':
            file_path = self.rollup_dir / 'RT_prices' / f"{year}.parquet"
        elif price_type == 'AS':
            file_path = self.rollup_dir / 'AS_prices' / f"{year}.parquet"
        else:
            raise ValueError(f"Unknown price type: {price_type}")
        
        if not file_path.exists():
            logger.error(f"Price file not found: {file_path}")
            return pd.DataFrame()
        
        return pd.read_parquet(file_path)
    
    def calculate_energy_revenue(self, resource_name: str, 
                                start_date: date, end_date: date) -> Tuple[float, float]:
        """Calculate energy revenue (DAM + RT) for a BESS resource"""
        
        # Get settlement point for this resource
        resource_info = self.bess_registry[self.bess_registry['resource_name'] == resource_name]
        if resource_info.empty:
            logger.warning(f"Resource {resource_name} not found in registry")
            return 0.0, 0.0
        
        settlement_point = resource_info.iloc[0]['settlement_point']
        
        dam_revenue = 0.0
        rt_revenue = 0.0
        
        # Process each year in the date range
        for year in range(start_date.year, end_date.year + 1):
            
            # Calculate year-specific date range
            year_start = max(start_date, date(year, 1, 1))
            year_end = min(end_date, date(year, 12, 31))
            
            # Load DAM awards
            dam_file = self.rollup_dir / 'DAM_Gen_Resources' / f"{year}.parquet"
            if dam_file.exists():
                # Read with filters for efficiency
                dam_awards = pq.read_table(
                    dam_file,
                    filters=[
                        ('ResourceName', '=', resource_name),
                        ('DeliveryDate', '>=', pd.Timestamp(year_start)),
                        ('DeliveryDate', '<=', pd.Timestamp(year_end))
                    ]
                ).to_pandas()
                
                if not dam_awards.empty:
                    # Load DAM prices
                    da_prices = self._load_price_data('DA', year)
                    
                    # Filter prices for settlement point and date range
                    if 'SettlementPoint' in da_prices.columns:
                        sp_column = 'SettlementPoint'
                    else:
                        sp_column = 'SettlementPointName'
                    
                    da_prices_filtered = da_prices[
                        (da_prices[sp_column] == settlement_point) &
                        (pd.to_datetime(da_prices['DeliveryDate']).dt.date >= year_start) &
                        (pd.to_datetime(da_prices['DeliveryDate']).dt.date <= year_end)
                    ]
                    
                    # Merge awards with prices
                    dam_awards['DeliveryDate'] = pd.to_datetime(dam_awards['DeliveryDate'])
                    da_prices_filtered['DeliveryDate'] = pd.to_datetime(da_prices_filtered['DeliveryDate'])
                    
                    # Handle hour format differences
                    if 'HourEnding' in dam_awards.columns:
                        dam_awards['HourMatch'] = dam_awards['HourEnding'].astype(str)
                    else:
                        dam_awards['HourMatch'] = dam_awards['hour'].astype(int).astype(str)
                    
                    if 'HourEnding' in da_prices_filtered.columns:
                        da_prices_filtered['HourMatch'] = da_prices_filtered['HourEnding'].astype(str)
                    else:
                        da_prices_filtered['HourMatch'] = da_prices_filtered['hour'].astype(int).astype(str)
                    
                    merged = pd.merge(
                        dam_awards,
                        da_prices_filtered[['DeliveryDate', 'HourMatch', 'SettlementPointPrice']],
                        on=['DeliveryDate', 'HourMatch'],
                        how='left'
                    )
                    
                    # Calculate DAM revenue
                    if 'AwardedQuantity' in merged.columns:
                        merged['dam_revenue'] = merged['AwardedQuantity'] * merged['SettlementPointPrice']
                        dam_revenue += merged['dam_revenue'].sum()
            
            # Load RT dispatch (SCED)
            sced_file = self.rollup_dir / 'SCED_Gen_Resources' / f"{year}.parquet"
            if sced_file.exists():
                # Note: SCED files might be very large, so we filter at read time
                try:
                    sced_data = pq.read_table(
                        sced_file,
                        filters=[
                            ('ResourceName', '=', resource_name)
                        ]
                    ).to_pandas()
                    
                    if not sced_data.empty:
                        # Load RT prices
                        rt_prices = self._load_price_data('RT', year)
                        
                        # Filter for settlement point
                        rt_prices_filtered = rt_prices[
                            rt_prices['SettlementPointName'] == settlement_point
                        ]
                        
                        # RT settlement is complex - need to aggregate 5-min SCED to 15-min settlement
                        # For now, simplified calculation
                        if 'TelemeteredOutput' in sced_data.columns and 'BasePoint' in sced_data.columns:
                            # Real-time deviation from DAM schedule
                            # This is a simplified calculation - actual ERCOT settlement is more complex
                            avg_price = rt_prices_filtered['SettlementPointPrice'].mean()
                            avg_output = sced_data['TelemeteredOutput'].mean()
                            
                            # Rough estimate: RT revenue from deviations
                            rt_revenue += avg_output * avg_price * len(sced_data) * (5/60)  # 5-minute intervals
                            
                except Exception as e:
                    logger.error(f"Error processing SCED data for {resource_name} in {year}: {e}")
        
        return dam_revenue, rt_revenue
    
    def calculate_as_revenue(self, resource_name: str,
                           start_date: date, end_date: date) -> Dict[str, float]:
        """Calculate ancillary service revenues for a BESS resource"""
        
        as_revenues = {
            'regup': 0.0,
            'regdown': 0.0,
            'rrs': 0.0,
            'ecrs': 0.0,
            'nspin': 0.0
        }
        
        # Process each year
        for year in range(start_date.year, end_date.year + 1):
            
            # Load DAM awards (contains AS awards)
            dam_file = self.rollup_dir / 'DAM_Gen_Resources' / f"{year}.parquet"
            if not dam_file.exists():
                continue
            
            try:
                # Read awards for this resource
                dam_awards = pq.read_table(
                    dam_file,
                    filters=[
                        ('ResourceName', '=', resource_name),
                        ('DeliveryDate', '>=', pd.Timestamp(date(year, 1, 1))),
                        ('DeliveryDate', '<=', pd.Timestamp(date(year, 12, 31)))
                    ]
                ).to_pandas()
                
                if dam_awards.empty:
                    continue
                
                # Load AS prices
                as_prices = self._load_price_data('AS', year)
                if as_prices.empty:
                    continue
                
                # Calculate revenue for each AS product
                as_columns = {
                    'RegUpAwarded': ('regup', 'REGUP'),
                    'RegDownAwarded': ('regdown', 'REGDN'),
                    'RRSAwarded': ('rrs', 'RRS'),
                    'ECRSAwarded': ('ecrs', 'ECRS'),
                    'NonSpinAwarded': ('nspin', 'NSPIN')
                }
                
                for award_col, (revenue_key, as_type) in as_columns.items():
                    if award_col in dam_awards.columns:
                        # Get AS prices for this type
                        as_type_prices = as_prices[as_prices['AncillaryType'] == as_type]
                        
                        if not as_type_prices.empty:
                            # Merge awards with prices
                            dam_awards['DeliveryDate'] = pd.to_datetime(dam_awards['DeliveryDate'])
                            as_type_prices['DeliveryDate'] = pd.to_datetime(as_type_prices['DeliveryDate'])
                            
                            # Handle hour matching
                            if 'HourEnding' in dam_awards.columns:
                                dam_awards['HourMatch'] = dam_awards['HourEnding'].astype(str)
                            else:
                                dam_awards['HourMatch'] = dam_awards.get('hour', 1).astype(int).astype(str)
                            
                            as_type_prices['HourMatch'] = as_type_prices['HourEnding'].astype(str)
                            
                            merged = pd.merge(
                                dam_awards[['DeliveryDate', 'HourMatch', award_col]],
                                as_type_prices[['DeliveryDate', 'HourMatch', 'MCPC']],
                                on=['DeliveryDate', 'HourMatch'],
                                how='left'
                            )
                            
                            # Calculate revenue
                            merged['revenue'] = merged[award_col] * merged['MCPC']
                            as_revenues[revenue_key] += merged['revenue'].sum()
                            
            except Exception as e:
                logger.error(f"Error calculating AS revenue for {resource_name} in {year}: {e}")
        
        return as_revenues
    
    def calculate_resource_revenue(self, resource_name: str,
                                  start_date: date, end_date: date) -> RevenueResult:
        """Calculate complete revenue for a BESS resource"""
        
        logger.info(f"Calculating revenue for {resource_name} from {start_date} to {end_date}")
        
        # Calculate energy revenue
        dam_revenue, rt_revenue = self.calculate_energy_revenue(resource_name, start_date, end_date)
        
        # Calculate AS revenue
        as_revenues = self.calculate_as_revenue(resource_name, start_date, end_date)
        
        # Create result
        result = RevenueResult(
            resource_name=resource_name,
            period_start=start_date,
            period_end=end_date,
            energy_revenue_dam=dam_revenue,
            energy_revenue_rt=rt_revenue,
            as_revenue_regup=as_revenues['regup'],
            as_revenue_regdown=as_revenues['regdown'],
            as_revenue_rrs=as_revenues['rrs'],
            as_revenue_ecrs=as_revenues['ecrs'],
            as_revenue_nspin=as_revenues['nspin'],
            total_energy_revenue=dam_revenue + rt_revenue,
            total_as_revenue=sum(as_revenues.values()),
            total_revenue=dam_revenue + rt_revenue + sum(as_revenues.values()),
            hours_operated=0  # TODO: Calculate from dispatch data
        )
        
        return result
    
    def calculate_all_resources(self, start_date: date, end_date: date,
                               resource_list: List[str] = None,
                               parallel: bool = True) -> pd.DataFrame:
        """Calculate revenue for multiple resources"""
        
        if resource_list is None:
            # Use all resources from registry
            resource_list = self.bess_registry['resource_name'].unique().tolist()
        
        results = []
        
        if parallel:
            # Process in parallel for speed
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = {
                    executor.submit(self.calculate_resource_revenue, resource, start_date, end_date): resource
                    for resource in resource_list
                }
                
                for future in as_completed(futures):
                    resource = futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Error calculating revenue for {resource}: {e}")
        else:
            # Sequential processing
            for resource in resource_list:
                try:
                    result = self.calculate_resource_revenue(resource, start_date, end_date)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error calculating revenue for {resource}: {e}")
        
        # Convert to DataFrame
        df = pd.DataFrame([vars(r) for r in results])
        
        # Add additional metrics
        if not df.empty:
            # Add capacity from registry
            df = df.merge(
                self.bess_registry[['resource_name', 'capacity_mw']],
                on='resource_name',
                how='left'
            )
            
            # Calculate revenue per MW
            df['revenue_per_mw'] = df['total_revenue'] / df['capacity_mw']
            
            # Classify strategy
            df['strategy'] = df.apply(self._classify_strategy, axis=1)
            
            # Sort by total revenue
            df = df.sort_values('total_revenue', ascending=False)
        
        return df
    
    def _classify_strategy(self, row) -> str:
        """Classify BESS operating strategy based on revenue mix"""
        if row['total_revenue'] == 0:
            return 'Inactive'
        
        energy_pct = row['total_energy_revenue'] / row['total_revenue']
        
        if energy_pct > 0.8:
            return 'Energy-Focused'
        elif energy_pct < 0.2:
            return 'AS-Focused'
        else:
            return 'Hybrid'

def generate_leaderboard(start_date: date, end_date: date,
                        top_n: int = 20) -> pd.DataFrame:
    """Generate BESS performance leaderboard"""
    
    logger.info("="*80)
    logger.info(f"Generating BESS Leaderboard: {start_date} to {end_date}")
    logger.info("="*80)
    
    # Initialize calculator
    calculator = BESSRevenueCalculator(ROLLUP_DIR)
    
    # Get top resources by capacity (for faster testing)
    top_resources = calculator.bess_registry.nlargest(top_n, 'capacity_mw')['resource_name'].tolist()
    
    # Calculate revenues
    results = calculator.calculate_all_resources(
        start_date=start_date,
        end_date=end_date,
        resource_list=top_resources,
        parallel=True
    )
    
    # Format for display
    if not results.empty:
        display_cols = [
            'resource_name', 'capacity_mw', 'total_revenue', 
            'total_energy_revenue', 'total_as_revenue',
            'revenue_per_mw', 'strategy'
        ]
        
        leaderboard = results[display_cols].head(top_n)
        
        # Format numbers
        for col in ['total_revenue', 'total_energy_revenue', 'total_as_revenue', 'revenue_per_mw']:
            leaderboard[col] = leaderboard[col].round(2)
        
        # Save to file
        output_path = OUTPUT_DIR / f"leaderboard_{start_date}_{end_date}.csv"
        leaderboard.to_csv(output_path, index=False)
        logger.info(f"Saved leaderboard to {output_path}")
        
        return leaderboard
    
    return pd.DataFrame()

def main():
    """Run Phase 2: Revenue calculation using Parquet files"""
    
    # Test with October 2024 data
    start_date = date(2024, 10, 1)
    end_date = date(2024, 10, 7)  # One week for testing
    
    # Generate leaderboard
    leaderboard = generate_leaderboard(start_date, end_date, top_n=10)
    
    if not leaderboard.empty:
        print("\n" + "="*100)
        print(f"BESS Revenue Leaderboard ({start_date} to {end_date})")
        print("="*100)
        print(leaderboard.to_string(index=False))
        print("="*100)
        
        # Summary statistics
        print(f"\nSummary Statistics:")
        print(f"  Total Revenue: ${leaderboard['total_revenue'].sum():,.2f}")
        print(f"  Average Revenue per BESS: ${leaderboard['total_revenue'].mean():,.2f}")
        print(f"  Energy vs AS Split: {leaderboard['total_energy_revenue'].sum():.1%} / {leaderboard['total_as_revenue'].sum():.1%}")
        
        # Strategy breakdown
        strategy_counts = leaderboard['strategy'].value_counts()
        print(f"\nStrategy Distribution:")
        for strategy, count in strategy_counts.items():
            print(f"  {strategy}: {count} resources")

if __name__ == "__main__":
    main()
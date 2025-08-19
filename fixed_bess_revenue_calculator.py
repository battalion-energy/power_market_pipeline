#!/usr/bin/env python3
"""
Fixed BESS Revenue Calculator with Proper Settlement Point Mapping
This version correctly maps BESS resources to their settlement points for accurate price matching
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FixedBessRevenueCalculator:
    def __init__(self, data_dir: Path = Path('/home/enrico/data/ERCOT_data')):
        self.data_dir = data_dir
        self.rollup_dir = data_dir / 'rollup_files'
        self.output_dir = data_dir / 'bess_analysis'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load settlement point mapping FIRST
        self.load_settlement_point_mapping()
        
        # Load BESS resources
        self.bess_resources = self.load_bess_resources()
        
    def load_settlement_point_mapping(self):
        """Load the official ERCOT settlement point mapping"""
        mapping_file = self.data_dir / 'Settlement_Points_List_and_Electrical_Buses_Mapping' / 'latest_mapping' / 'SP_List_EB_Mapping' / 'gen_node_map.csv'
        
        if mapping_file.exists():
            logger.info(f"Loading settlement point mapping from {mapping_file}")
            sp_map = pd.read_csv(mapping_file)
            
            # Create bidirectional mappings
            self.unit_to_sp = dict(zip(sp_map['UNIT_NAME'], sp_map['RESOURCE_NODE']))
            self.sp_to_units = {}
            for _, row in sp_map.iterrows():
                sp = row['RESOURCE_NODE']
                unit = row['UNIT_NAME']
                if sp not in self.sp_to_units:
                    self.sp_to_units[sp] = []
                self.sp_to_units[sp].append(unit)
            
            logger.info(f"Loaded {len(self.unit_to_sp)} unit-to-SP mappings")
            logger.info(f"Loaded {len(self.sp_to_units)} unique settlement points")
        else:
            logger.error(f"Settlement point mapping not found at {mapping_file}")
            self.unit_to_sp = {}
            self.sp_to_units = {}
    
    def load_bess_resources(self) -> pd.DataFrame:
        """Load and identify BESS resources with proper SP mapping"""
        
        # First, identify all BESS from DAM data
        bess_list = []
        
        # Check 2024 DAM Gen for BESS resources
        dam_gen_file = self.rollup_dir / 'DAM_Gen_Resources' / '2024.parquet'
        if dam_gen_file.exists():
            dam_gen = pd.read_parquet(dam_gen_file)
            
            # Filter for PWRSTR (Power Storage) resources
            bess_data = dam_gen[dam_gen['ResourceType'] == 'PWRSTR'].copy()
            
            if not bess_data.empty:
                # Get unique resources with their settlement points from file
                unique_bess = bess_data.groupby('ResourceName').agg({
                    'SettlementPointName': 'first',
                    'AwardedQuantity': 'count'
                }).reset_index()
                
                unique_bess.columns = ['resource_name', 'settlement_point_from_file', 'record_count']
                
                # Now map using the official mapping
                unique_bess['mapped_settlement_point'] = unique_bess['resource_name'].map(self.unit_to_sp)
                
                # Use mapped SP if available, otherwise use from file
                unique_bess['settlement_point'] = unique_bess['mapped_settlement_point'].fillna(
                    unique_bess['settlement_point_from_file']
                )
                
                # Log mapping results
                mapped_count = unique_bess['mapped_settlement_point'].notna().sum()
                logger.info(f"Successfully mapped {mapped_count}/{len(unique_bess)} BESS to settlement points")
                
                # Log unmapped resources
                unmapped = unique_bess[unique_bess['mapped_settlement_point'].isna()]
                if not unmapped.empty:
                    logger.warning(f"Unmapped BESS resources: {unmapped['resource_name'].tolist()[:10]}")
                
                bess_list = unique_bess[['resource_name', 'settlement_point']].to_dict('records')
        
        if not bess_list:
            logger.warning("No BESS resources found")
            return pd.DataFrame()
        
        # Create registry dataframe
        registry = pd.DataFrame(bess_list)
        
        # Add default capacity and duration
        registry['capacity_mw'] = 100.0  # Default
        registry['duration_hours'] = 2.0  # Default
        
        # Save registry
        registry_file = self.output_dir / 'bess_registry_fixed.parquet'
        registry.to_parquet(registry_file, index=False)
        logger.info(f"Saved fixed BESS registry with {len(registry)} resources to {registry_file}")
        
        return registry
    
    def calculate_revenues(self, year: int, limit: Optional[int] = None) -> pd.DataFrame:
        """Calculate revenues for BESS resources"""
        
        logger.info(f"Calculating revenues for {year}")
        
        if self.bess_resources.empty:
            logger.error("No BESS resources to process")
            return pd.DataFrame()
        
        # Load price data
        da_prices = self.load_da_prices(year)
        as_prices = self.load_as_prices(year)
        
        if da_prices.empty:
            logger.error(f"No DA prices for {year}")
            return pd.DataFrame()
        
        # Load award data
        dam_gen = self.load_dam_gen(year)
        
        if dam_gen.empty:
            logger.error(f"No DAM Gen data for {year}")
            return pd.DataFrame()
        
        # Process each BESS
        resources_to_process = self.bess_resources.head(limit) if limit else self.bess_resources
        
        results = []
        for _, bess in resources_to_process.iterrows():
            resource_name = bess['resource_name']
            settlement_point = bess['settlement_point']
            
            # Get awards for this resource
            resource_awards = dam_gen[dam_gen['ResourceName'] == resource_name].copy()
            
            if resource_awards.empty:
                continue
            
            # Ensure datetime column
            if 'DeliveryDate' in resource_awards.columns:
                resource_awards['datetime'] = pd.to_datetime(resource_awards['DeliveryDate'])
            
            # Calculate DA energy revenue
            da_revenue = self.calc_da_revenue(resource_awards, da_prices, settlement_point)
            
            # Calculate AS revenues
            as_revenue = self.calc_as_revenue(resource_awards, as_prices)
            
            results.append({
                'resource': resource_name,
                'settlement_point': settlement_point,
                'da_revenue': da_revenue,
                'as_revenue': as_revenue,
                'total_revenue': da_revenue + as_revenue
            })
            
            logger.info(f"{resource_name} ({settlement_point}): DA=${da_revenue:,.0f}, AS=${as_revenue:,.0f}")
        
        return pd.DataFrame(results)
    
    def load_da_prices(self, year: int) -> pd.DataFrame:
        """Load DA prices"""
        price_file = self.rollup_dir / 'flattened' / f'DA_prices_{year}.parquet'
        if not price_file.exists():
            return pd.DataFrame()
        
        prices = pd.read_parquet(price_file)
        
        # Ensure datetime column
        if 'datetime_ts' in prices.columns:
            prices['datetime'] = pd.to_datetime(prices['datetime_ts'])
        
        return prices
    
    def load_as_prices(self, year: int) -> pd.DataFrame:
        """Load AS prices"""
        price_file = self.rollup_dir / 'flattened' / f'AS_prices_{year}.parquet'
        if not price_file.exists():
            return pd.DataFrame()
        
        prices = pd.read_parquet(price_file)
        
        # Ensure datetime column
        if 'datetime_ts' in prices.columns:
            prices['datetime'] = pd.to_datetime(prices['datetime_ts'])
            
        return prices
    
    def load_dam_gen(self, year: int) -> pd.DataFrame:
        """Load DAM Gen awards"""
        dam_file = self.rollup_dir / 'DAM_Gen_Resources' / f'{year}.parquet'
        if not dam_file.exists():
            return pd.DataFrame()
        
        dam = pd.read_parquet(dam_file)
        
        # Filter for BESS
        return dam[dam['ResourceType'] == 'PWRSTR']
    
    def calc_da_revenue(self, awards: pd.DataFrame, prices: pd.DataFrame, settlement_point: str) -> float:
        """Calculate DA energy revenue using correct settlement point"""
        
        if awards.empty or prices.empty:
            return 0.0
        
        # Check if settlement point exists in price data
        if settlement_point in prices.columns:
            sp_prices = prices[['datetime', settlement_point]].copy()
            sp_prices.columns = ['datetime', 'price']
            logger.debug(f"Using prices from {settlement_point}")
        elif 'HB_BUSAVG' in prices.columns:
            # Fallback to hub average
            sp_prices = prices[['datetime', 'HB_BUSAVG']].copy()
            sp_prices.columns = ['datetime', 'price']
            logger.debug(f"Using HB_BUSAVG fallback for {settlement_point}")
        else:
            logger.warning(f"No price data available for {settlement_point}")
            return 0.0
        
        # Merge awards with prices
        merged = awards.merge(sp_prices, on='datetime', how='left')
        
        # Calculate revenue
        merged['revenue'] = merged['AwardedQuantity'] * merged['price']
        
        return merged['revenue'].sum()
    
    def calc_as_revenue(self, awards: pd.DataFrame, prices: pd.DataFrame) -> float:
        """Calculate AS revenues"""
        
        if awards.empty or prices.empty:
            return 0.0
        
        total_as = 0.0
        
        # Map AS award columns to price columns
        as_mapping = {
            'RegUpAwarded': 'REGUP',
            'RegDownAwarded': 'REGDN',
            'RRSAwarded': 'RRS',
            'NonSpinAwarded': 'NSPIN',
            'ECRSAwarded': 'ECRS'
        }
        
        for award_col, price_col in as_mapping.items():
            if award_col in awards.columns and price_col in prices.columns:
                # Merge and calculate
                merged = awards[['datetime', award_col]].merge(
                    prices[['datetime', price_col]], 
                    on='datetime', 
                    how='left'
                )
                merged['revenue'] = merged[award_col] * merged[price_col]
                total_as += merged['revenue'].sum()
        
        return total_as

def main():
    """Run fixed BESS revenue calculation"""
    
    print("=" * 80)
    print("FIXED BESS REVENUE CALCULATOR - WITH PROPER SETTLEMENT POINT MAPPING")
    print("=" * 80)
    
    calculator = FixedBessRevenueCalculator()
    
    # Calculate for 2024 sample
    results = calculator.calculate_revenues(2024, limit=20)
    
    if not results.empty:
        print("\n" + "=" * 80)
        print("RESULTS WITH CORRECT SETTLEMENT POINT MAPPING")
        print("=" * 80)
        
        print(results.to_string())
        
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Total Resources: {len(results)}")
        print(f"Total DA Revenue: ${results['da_revenue'].sum():,.0f}")
        print(f"Total AS Revenue: ${results['as_revenue'].sum():,.0f}")
        print(f"Total Revenue: ${results['total_revenue'].sum():,.0f}")
        
        # Save results
        output_file = Path('/tmp/fixed_bess_results.parquet')
        results.to_parquet(output_file, index=False)
        print(f"\nResults saved to {output_file}")
    else:
        print("No results generated")

if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
Complete BESS Revenue Calculator INCLUDING Real-Time Energy
Calculates DA, RT, and AS revenues for all BESS in ERCOT
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class CompleteBessCalculator:
    def __init__(self, data_dir: Path = Path('/home/enrico/data/ERCOT_data')):
        self.data_dir = data_dir
        self.rollup_dir = data_dir / 'rollup_files'
        self.output_dir = data_dir / 'bess_analysis'
        self.output_dir.mkdir(exist_ok=True)
        
        # Load settlement point mapping
        self.load_settlement_mapping()
        
    def load_settlement_mapping(self):
        """Load ERCOT settlement point mapping"""
        mapping_file = self.data_dir / 'Settlement_Points_List_and_Electrical_Buses_Mapping' / 'latest_mapping' / 'SP_List_EB_Mapping' / 'gen_node_map.csv'
        
        if mapping_file.exists():
            sp_map = pd.read_csv(mapping_file)
            self.unit_to_sp = dict(zip(sp_map['UNIT_NAME'], sp_map['RESOURCE_NODE']))
            logger.info(f"Loaded {len(self.unit_to_sp)} settlement point mappings")
        else:
            logger.warning("Settlement point mapping not found")
            self.unit_to_sp = {}
    
    def calculate_all_revenues(self, years: List[int] = [2023, 2024]) -> pd.DataFrame:
        """Calculate complete revenues including DA, RT, and AS"""
        
        all_results = []
        
        for year in years:
            logger.info(f"\n{'='*80}")
            logger.info(f"Processing {year}")
            logger.info(f"{'='*80}")
            
            # Load all required data
            data = self.load_year_data(year)
            if not data:
                continue
            
            # Get unique BESS resources
            bess_resources = data['dam_gen']['ResourceName'].unique()
            logger.info(f"Found {len(bess_resources)} BESS resources")
            
            # Process each BESS
            for idx, bess_name in enumerate(bess_resources, 1):
                if idx % 20 == 0:
                    logger.info(f"  Processed {idx}/{len(bess_resources)} resources...")
                
                # Calculate revenues for this BESS
                revenues = self.calculate_bess_revenues(bess_name, year, data)
                
                if revenues:
                    revenues['year'] = year
                    all_results.append(revenues)
        
        # Create complete results dataframe
        if all_results:
            results_df = pd.DataFrame(all_results)
            
            # Add rankings and metrics
            results_df = self.add_metrics(results_df)
            
            # Save results
            self.save_results(results_df)
            
            return results_df
        else:
            logger.error("No results generated")
            return pd.DataFrame()
    
    def load_year_data(self, year: int) -> Optional[Dict]:
        """Load all data files for a year"""
        
        data = {}
        
        # Load DAM Generation data
        dam_gen_file = self.rollup_dir / 'DAM_Gen_Resources' / f'{year}.parquet'
        if not dam_gen_file.exists():
            logger.warning(f"DAM Gen data not found for {year}")
            return None
        
        dam_gen = pd.read_parquet(dam_gen_file)
        data['dam_gen'] = dam_gen[dam_gen['ResourceType'] == 'PWRSTR'].copy()
        
        if 'DeliveryDate' in data['dam_gen'].columns:
            data['dam_gen']['datetime'] = pd.to_datetime(data['dam_gen']['DeliveryDate'])
        
        # Load DAM Load data (for charging)
        dam_load_file = self.rollup_dir / 'DAM_Load_Resources' / f'{year}.parquet'
        if dam_load_file.exists():
            dam_load = pd.read_parquet(dam_load_file)
            if 'DeliveryDate' in dam_load.columns:
                dam_load['datetime'] = pd.to_datetime(dam_load['DeliveryDate'])
            data['dam_load'] = dam_load
        else:
            data['dam_load'] = pd.DataFrame()
        
        # Load SCED Generation data (Real-Time dispatch)
        sced_gen_file = self.rollup_dir / 'SCED_Gen_Resources' / f'{year}.parquet'
        if sced_gen_file.exists():
            sced_gen = pd.read_parquet(sced_gen_file)
            # Filter for BESS
            data['sced_gen'] = sced_gen[sced_gen['ResourceType'] == 'PWRSTR'].copy() if 'ResourceType' in sced_gen.columns else sced_gen
            if 'SCEDTimestamp' in data['sced_gen'].columns:
                data['sced_gen']['datetime'] = pd.to_datetime(data['sced_gen']['SCEDTimestamp'])
            elif 'DeliveryDate' in data['sced_gen'].columns:
                data['sced_gen']['datetime'] = pd.to_datetime(data['sced_gen']['DeliveryDate'])
        else:
            data['sced_gen'] = pd.DataFrame()
            logger.warning(f"SCED Gen data not found for {year}")
        
        # Load SCED Load data
        sced_load_file = self.rollup_dir / 'SCED_Load_Resources' / f'{year}.parquet'
        if sced_load_file.exists():
            sced_load = pd.read_parquet(sced_load_file)
            if 'SCEDTimestamp' in sced_load.columns:
                sced_load['datetime'] = pd.to_datetime(sced_load['SCEDTimestamp'])
            data['sced_load'] = sced_load
        else:
            data['sced_load'] = pd.DataFrame()
        
        # Load price data
        da_price_file = self.rollup_dir / 'flattened' / f'DA_prices_{year}.parquet'
        if da_price_file.exists():
            da_prices = pd.read_parquet(da_price_file)
            if 'datetime_ts' in da_prices.columns:
                da_prices['datetime'] = pd.to_datetime(da_prices['datetime_ts'])
            data['da_prices'] = da_prices
        else:
            logger.warning(f"DA prices not found for {year}")
            data['da_prices'] = pd.DataFrame()
        
        # Load RT prices (15-minute intervals)
        rt_price_file = self.rollup_dir / 'flattened' / f'RT_prices_15min_{year}.parquet'
        if rt_price_file.exists():
            rt_prices = pd.read_parquet(rt_price_file)
            if 'datetime_ts' in rt_prices.columns:
                rt_prices['datetime'] = pd.to_datetime(rt_prices['datetime_ts'])
            data['rt_prices'] = rt_prices
            logger.info(f"Loaded RT prices with {len(rt_prices)} intervals")
        else:
            # Try hourly RT prices as fallback
            rt_hourly_file = self.rollup_dir / 'flattened' / f'RT_prices_hourly_{year}.parquet'
            if rt_hourly_file.exists():
                rt_prices = pd.read_parquet(rt_hourly_file)
                if 'datetime_ts' in rt_prices.columns:
                    rt_prices['datetime'] = pd.to_datetime(rt_prices['datetime_ts'])
                data['rt_prices'] = rt_prices
                logger.info(f"Using hourly RT prices")
            else:
                logger.warning(f"RT prices not found for {year}")
                data['rt_prices'] = pd.DataFrame()
        
        # Load AS prices
        as_price_file = self.rollup_dir / 'flattened' / f'AS_prices_{year}.parquet'
        if as_price_file.exists():
            as_prices = pd.read_parquet(as_price_file)
            if 'datetime_ts' in as_prices.columns:
                as_prices['datetime'] = pd.to_datetime(as_prices['datetime_ts'])
            data['as_prices'] = as_prices
        else:
            data['as_prices'] = pd.DataFrame()
        
        return data
    
    def calculate_bess_revenues(self, bess_name: str, year: int, data: Dict) -> Optional[Dict]:
        """Calculate all revenue streams for a single BESS"""
        
        # Get settlement point
        settlement_point = self.get_settlement_point(bess_name, data['dam_gen'])
        
        # Get awards for this BESS
        dam_gen_awards = data['dam_gen'][data['dam_gen']['ResourceName'] == bess_name].copy()
        
        if dam_gen_awards.empty:
            return None
        
        # Initialize results
        result = {
            'resource_name': bess_name,
            'settlement_point': settlement_point,
            'da_energy_revenue': 0,
            'da_energy_cost': 0,
            'rt_energy_revenue': 0,
            'rt_energy_cost': 0,
            'as_regup_revenue': 0,
            'as_regdn_revenue': 0,
            'as_rrs_revenue': 0,
            'as_nonspin_revenue': 0,
            'as_ecrs_revenue': 0
        }
        
        # 1. Calculate DA Energy Revenue/Cost
        if not data['da_prices'].empty:
            da_rev = self.calc_da_energy(dam_gen_awards, data['dam_load'], data['da_prices'], 
                                         bess_name, settlement_point)
            result.update(da_rev)
        
        # 2. Calculate RT Energy Revenue/Cost
        if not data['rt_prices'].empty and not data['sced_gen'].empty:
            rt_rev = self.calc_rt_energy(data['sced_gen'], data['sced_load'], data['rt_prices'],
                                         bess_name, settlement_point)
            result.update(rt_rev)
        
        # 3. Calculate AS Revenues
        if not data['as_prices'].empty:
            as_rev = self.calc_as_revenues(dam_gen_awards, data['as_prices'])
            result.update(as_rev)
        
        # Calculate totals
        result['da_net_energy'] = result['da_energy_revenue'] - result['da_energy_cost']
        result['rt_net_energy'] = result['rt_energy_revenue'] - result['rt_energy_cost']
        result['as_total_revenue'] = (result['as_regup_revenue'] + result['as_regdn_revenue'] + 
                                      result['as_rrs_revenue'] + result['as_nonspin_revenue'] + 
                                      result['as_ecrs_revenue'])
        result['total_energy_revenue'] = result['da_net_energy'] + result['rt_net_energy']
        result['total_revenue'] = result['total_energy_revenue'] + result['as_total_revenue']
        
        # Get capacity and MWh
        result['capacity_mw'] = abs(dam_gen_awards['AwardedQuantity']).max() if not dam_gen_awards.empty else 100
        result['total_mwh'] = abs(dam_gen_awards['AwardedQuantity']).sum()
        
        return result
    
    def get_settlement_point(self, bess_name: str, dam_gen: pd.DataFrame) -> str:
        """Get settlement point for BESS resource"""
        
        # First try the official mapping
        if bess_name in self.unit_to_sp:
            return self.unit_to_sp[bess_name]
        
        # Otherwise use from DAM data
        bess_data = dam_gen[dam_gen['ResourceName'] == bess_name]
        if not bess_data.empty and 'SettlementPointName' in bess_data.columns:
            return bess_data['SettlementPointName'].iloc[0]
        
        return 'UNKNOWN'
    
    def calc_da_energy(self, dam_gen: pd.DataFrame, dam_load: pd.DataFrame, 
                      da_prices: pd.DataFrame, bess_name: str, settlement_point: str) -> Dict:
        """Calculate DA energy revenues and costs"""
        
        result = {'da_energy_revenue': 0, 'da_energy_cost': 0}
        
        # Generation revenue (discharge)
        if not dam_gen.empty:
            if settlement_point in da_prices.columns:
                merged = dam_gen.merge(da_prices[['datetime', settlement_point]], 
                                      on='datetime', how='left')
                # Positive awards are generation
                gen_revenue = merged[merged['AwardedQuantity'] > 0]
                result['da_energy_revenue'] = (gen_revenue['AwardedQuantity'] * 
                                              gen_revenue[settlement_point]).sum()
            elif 'HB_BUSAVG' in da_prices.columns:
                merged = dam_gen.merge(da_prices[['datetime', 'HB_BUSAVG']], 
                                      on='datetime', how='left')
                gen_revenue = merged[merged['AwardedQuantity'] > 0]
                result['da_energy_revenue'] = (gen_revenue['AwardedQuantity'] * 
                                              gen_revenue['HB_BUSAVG']).sum()
        
        # Load cost (charging) - check load resources
        if not dam_load.empty and 'ResourceName' in dam_load.columns:
            # Look for matching load resource (e.g., BESS_NAME_LD1)
            load_names = [bess_name + '_LD1', bess_name + '_LD2', bess_name + '_LOAD']
            load_data = dam_load[dam_load['ResourceName'].isin(load_names)]
            
            if not load_data.empty:
                if 'datetime' not in load_data.columns and 'DeliveryDate' in load_data.columns:
                    load_data['datetime'] = pd.to_datetime(load_data['DeliveryDate'])
                
                if settlement_point in da_prices.columns:
                    merged = load_data.merge(da_prices[['datetime', settlement_point]], 
                                            on='datetime', how='left')
                    # Load awards are typically positive, cost = award * price
                    result['da_energy_cost'] = (merged['AwardedQuantity'] * 
                                               merged[settlement_point]).sum()
                elif 'HB_BUSAVG' in da_prices.columns:
                    merged = load_data.merge(da_prices[['datetime', 'HB_BUSAVG']], 
                                            on='datetime', how='left')
                    result['da_energy_cost'] = (merged['AwardedQuantity'] * 
                                               merged['HB_BUSAVG']).sum()
        
        # Also check for negative awards in gen (charging)
        if not dam_gen.empty:
            charge_data = dam_gen[dam_gen['AwardedQuantity'] < 0]
            if not charge_data.empty:
                if settlement_point in da_prices.columns:
                    merged = charge_data.merge(da_prices[['datetime', settlement_point]], 
                                              on='datetime', how='left')
                    # Negative awards mean charging, cost is abs(award) * price
                    result['da_energy_cost'] += abs((merged['AwardedQuantity'] * 
                                                     merged[settlement_point]).sum())
                elif 'HB_BUSAVG' in da_prices.columns:
                    merged = charge_data.merge(da_prices[['datetime', 'HB_BUSAVG']], 
                                              on='datetime', how='left')
                    result['da_energy_cost'] += abs((merged['AwardedQuantity'] * 
                                                     merged['HB_BUSAVG']).sum())
        
        return result
    
    def calc_rt_energy(self, sced_gen: pd.DataFrame, sced_load: pd.DataFrame,
                      rt_prices: pd.DataFrame, bess_name: str, settlement_point: str) -> Dict:
        """Calculate RT energy revenues and costs"""
        
        result = {'rt_energy_revenue': 0, 'rt_energy_cost': 0}
        
        if sced_gen.empty or rt_prices.empty:
            return result
        
        # Get SCED data for this BESS
        bess_sced = sced_gen[sced_gen['ResourceName'] == bess_name].copy()
        
        if bess_sced.empty:
            return result
        
        # Use BasePoint or TelemeteredOutput for actual dispatch
        output_col = 'BasePoint' if 'BasePoint' in bess_sced.columns else 'TelemeteredOutput'
        
        if output_col not in bess_sced.columns:
            return result
        
        # Match with RT prices
        if settlement_point in rt_prices.columns:
            merged = bess_sced.merge(rt_prices[['datetime', settlement_point]], 
                                    on='datetime', how='left')
            # Positive output is generation revenue
            gen_data = merged[merged[output_col] > 0]
            if not gen_data.empty:
                # Convert MW to MWh for 5-minute intervals (MW * 5/60)
                result['rt_energy_revenue'] = (gen_data[output_col] * gen_data[settlement_point] * (5/60)).sum()
            
            # Negative output is charging cost
            charge_data = merged[merged[output_col] < 0]
            if not charge_data.empty:
                result['rt_energy_cost'] = abs((charge_data[output_col] * charge_data[settlement_point] * (5/60)).sum())
        
        elif 'HB_BUSAVG' in rt_prices.columns:
            # Use hub average as fallback
            merged = bess_sced.merge(rt_prices[['datetime', 'HB_BUSAVG']], 
                                    on='datetime', how='left')
            gen_data = merged[merged[output_col] > 0]
            if not gen_data.empty:
                result['rt_energy_revenue'] = (gen_data[output_col] * gen_data['HB_BUSAVG'] * (5/60)).sum()
            
            charge_data = merged[merged[output_col] < 0]
            if not charge_data.empty:
                result['rt_energy_cost'] = abs((charge_data[output_col] * charge_data['HB_BUSAVG'] * (5/60)).sum())
        
        return result
    
    def calc_as_revenues(self, dam_gen: pd.DataFrame, as_prices: pd.DataFrame) -> Dict:
        """Calculate AS revenues"""
        
        result = {
            'as_regup_revenue': 0,
            'as_regdn_revenue': 0,
            'as_rrs_revenue': 0,
            'as_nonspin_revenue': 0,
            'as_ecrs_revenue': 0
        }
        
        if dam_gen.empty or as_prices.empty:
            return result
        
        # Map AS award columns to price columns
        as_mapping = {
            ('RegUpAwarded', 'REGUP', 'as_regup_revenue'),
            ('RegDownAwarded', 'REGDN', 'as_regdn_revenue'),
            ('RRSAwarded', 'RRS', 'as_rrs_revenue'),
            ('NonSpinAwarded', 'NSPIN', 'as_nonspin_revenue'),
            ('ECRSAwarded', 'ECRS', 'as_ecrs_revenue')
        }
        
        for award_col, price_col, revenue_key in as_mapping:
            if award_col in dam_gen.columns and price_col in as_prices.columns:
                merged = dam_gen[['datetime', award_col]].merge(
                    as_prices[['datetime', price_col]], 
                    on='datetime', 
                    how='left'
                )
                result[revenue_key] = (merged[award_col] * merged[price_col]).sum()
        
        return result
    
    def add_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add calculated metrics and rankings"""
        
        # Add revenue percentages
        df['da_pct'] = (df['da_net_energy'] / df['total_revenue'] * 100).fillna(0)
        df['rt_pct'] = (df['rt_net_energy'] / df['total_revenue'] * 100).fillna(0)
        df['as_pct'] = (df['as_total_revenue'] / df['total_revenue'] * 100).fillna(0)
        
        # Add per-MW metrics
        df['revenue_per_mw'] = df['total_revenue'] / df['capacity_mw']
        df['revenue_per_mwh'] = df['total_revenue'] / df['total_mwh'].replace(0, np.nan)
        
        # Add rankings
        df = df.sort_values('total_revenue', ascending=False)
        df['rank'] = range(1, len(df) + 1)
        
        return df
    
    def save_results(self, df: pd.DataFrame):
        """Save results to multiple formats"""
        
        # Save parquet
        parquet_file = self.output_dir / 'complete_bess_revenues_with_rt.parquet'
        df.to_parquet(parquet_file, index=False)
        logger.info(f"Saved to {parquet_file}")
        
        # Save CSV
        csv_file = self.output_dir / 'complete_bess_revenues_with_rt.csv'
        df.to_csv(csv_file, index=False)
        logger.info(f"Saved to {csv_file}")
        
        # Create summary
        self.print_summary(df)
    
    def print_summary(self, df: pd.DataFrame):
        """Print comprehensive summary"""
        
        print("\n" + "="*100)
        print("COMPLETE BESS REVENUE ANALYSIS - INCLUDING DA, RT, AND AS")
        print("="*100)
        
        # Top 20 by total revenue
        print("\nTOP 20 BESS PROJECTS BY TOTAL REVENUE")
        print("-"*100)
        print(f"{'Rank':<5} {'Resource':<25} {'Total Revenue':>15} {'DA':>12} {'RT':>12} {'AS':>12} {'Settlement Point':<20}")
        print("-"*100)
        
        for _, row in df.head(20).iterrows():
            print(f"{row['rank']:<5d} {row['resource_name']:<25} ${row['total_revenue']:>14,.0f} "
                  f"${row['da_net_energy']:>11,.0f} ${row['rt_net_energy']:>11,.0f} "
                  f"${row['as_total_revenue']:>11,.0f} {row['settlement_point']:<20}")
        
        # Overall statistics
        print("\n" + "="*100)
        print("OVERALL STATISTICS")
        print("="*100)
        
        total_revenue = df['total_revenue'].sum()
        total_da = df['da_net_energy'].sum()
        total_rt = df['rt_net_energy'].sum()
        total_as = df['as_total_revenue'].sum()
        
        print(f"Total BESS Projects: {df['resource_name'].nunique()}")
        print(f"Total Revenue: ${total_revenue:,.0f}")
        print(f"  - DA Energy (net): ${total_da:,.0f} ({total_da/total_revenue*100:.1f}%)")
        print(f"  - RT Energy (net): ${total_rt:,.0f} ({total_rt/total_revenue*100:.1f}%)")
        print(f"  - AS Services: ${total_as:,.0f} ({total_as/total_revenue*100:.1f}%)")
        
        # Revenue distribution
        print("\n" + "="*100)
        print("REVENUE DISTRIBUTION BY SERVICE")
        print("="*100)
        
        # Count projects by primary revenue source
        df['primary_source'] = df[['da_net_energy', 'rt_net_energy', 'as_total_revenue']].idxmax(axis=1)
        source_counts = df['primary_source'].value_counts()
        
        print("\nProjects by Primary Revenue Source:")
        for source, count in source_counts.items():
            source_name = source.replace('_net_energy', '').replace('_total_revenue', '').upper()
            print(f"  {source_name}: {count} projects")
        
        # Projects with RT revenue
        rt_projects = df[df['rt_net_energy'] > 0]
        print(f"\nProjects with RT Revenue: {len(rt_projects)} ({len(rt_projects)/len(df)*100:.1f}%)")
        print(f"Average RT Revenue (when > 0): ${rt_projects['rt_net_energy'].mean():,.0f}")

def main():
    """Run complete BESS revenue analysis"""
    
    calculator = CompleteBessCalculator()
    results = calculator.calculate_all_revenues(years=[2024])  # Can add more years
    
    if not results.empty:
        print("\n✅ Complete BESS revenue analysis finished!")
        print(f"   Results saved to: /home/enrico/data/ERCOT_data/bess_analysis/")
    else:
        print("❌ No results generated")

if __name__ == '__main__':
    main()
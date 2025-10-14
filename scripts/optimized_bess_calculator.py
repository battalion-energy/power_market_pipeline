#!/usr/bin/env python3
"""
Optimized BESS Revenue Calculator with RT Integration
Fast, parallel processing with proper datetime alignment
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizedBessCalculator:
    def __init__(self):
        self.data_dir = Path('/home/enrico/data/ERCOT_data')
        self.rollup_dir = self.data_dir / 'rollup_files'
        self.output_dir = self.data_dir / 'bess_analysis'
        self.output_dir.mkdir(exist_ok=True)
        
        # Load mappings once
        self.load_mappings()
    
    def load_mappings(self):
        """Load settlement point mappings once"""
        mapping_file = self.data_dir / 'Settlement_Points_List_and_Electrical_Buses_Mapping/latest_mapping/SP_List_EB_Mapping/gen_node_map.csv'
        sp_map = pd.read_csv(mapping_file)
        self.unit_to_sp = dict(zip(sp_map['UNIT_NAME'], sp_map['RESOURCE_NODE']))
        logger.info(f"Loaded {len(self.unit_to_sp)} settlement mappings")
    
    def calculate_all_fast(self):
        """Fast calculation using optimized data loading and processing"""
        
        print("="*100)
        print("OPTIMIZED BESS REVENUE CALCULATOR - WITH RT INTEGRATION")
        print("="*100)
        
        all_results = []
        years = [2023, 2024]  # 2025 needs date fix first
        
        for year in years:
            print(f"\n{'='*60}")
            print(f"Processing {year}")
            print(f"{'='*60}")
            
            # Pre-load all data for the year
            data = self.preload_year_data(year)
            
            if not data:
                continue
            
            # Get unique BESS
            bess_list = data['bess_list']
            print(f"Processing {len(bess_list)} BESS resources...")
            
            # Process in batches for memory efficiency
            batch_size = 20
            for i in range(0, len(bess_list), batch_size):
                batch = bess_list[i:i+batch_size]
                batch_results = self.process_batch(batch, data, year)
                all_results.extend(batch_results)
                
                if (i + batch_size) % 100 == 0:
                    print(f"  Processed {i + batch_size}/{len(bess_list)} resources...")
        
        # Create final dataframe
        if all_results:
            df = pd.DataFrame(all_results)
            self.save_and_display_results(df)
            return df
        
        return pd.DataFrame()
    
    def preload_year_data(self, year):
        """Pre-load and prepare all data for a year"""
        
        data = {}
        
        # Load DAM Gen
        dam_gen_file = self.rollup_dir / 'DAM_Gen_Resources' / f'{year}.parquet'
        if not dam_gen_file.exists():
            return None
        
        dam_gen = pd.read_parquet(dam_gen_file)
        bess_dam = dam_gen[dam_gen['ResourceType'] == 'PWRSTR'].copy()
        
        # Fix datetime
        if 'DeliveryDate' in bess_dam.columns:
            bess_dam['datetime'] = pd.to_datetime(bess_dam['DeliveryDate'])
        
        data['dam_gen'] = bess_dam
        data['bess_list'] = bess_dam['ResourceName'].unique().tolist()
        
        # Load prices - convert datetime once
        print("  Loading DA prices...")
        da_prices = pd.read_parquet(self.rollup_dir / 'flattened' / f'DA_prices_{year}.parquet')
        if 'datetime_ts' in da_prices.columns:
            da_prices['datetime'] = pd.to_datetime(da_prices['datetime_ts'])
        data['da_prices'] = da_prices.set_index('datetime')
        
        # Load AS prices
        print("  Loading AS prices...")
        as_file = self.rollup_dir / 'flattened' / f'AS_prices_{year}.parquet'
        if as_file.exists():
            as_prices = pd.read_parquet(as_file)
            if 'datetime_ts' in as_prices.columns:
                as_prices['datetime'] = pd.to_datetime(as_prices['datetime_ts'])
            data['as_prices'] = as_prices.set_index('datetime')
        else:
            data['as_prices'] = pd.DataFrame()
        
        # Load RT prices - try hourly first (faster)
        print("  Loading RT prices...")
        rt_file = self.rollup_dir / 'flattened' / f'RT_prices_hourly_{year}.parquet'
        if rt_file.exists():
            rt_prices = pd.read_parquet(rt_file)
            if 'datetime_ts' in rt_prices.columns:
                rt_prices['datetime'] = pd.to_datetime(rt_prices['datetime_ts'])
            data['rt_prices'] = rt_prices.set_index('datetime')
        else:
            data['rt_prices'] = pd.DataFrame()
        
        # Load SCED Gen
        print("  Loading SCED data...")
        sced_file = self.rollup_dir / 'SCED_Gen_Resources' / f'{year}.parquet'
        if sced_file.exists():
            # Only load BESS SCED data to save memory
            sced = pd.read_parquet(sced_file)
            
            # Fix SCED datetime
            if 'datetime' in sced.columns and sced['datetime'].dtype == 'object':
                print("    Converting SCED datetime...")
                sced['datetime'] = pd.to_datetime(sced['datetime'], format='%m/%d/%Y %H:%M:%S')
            
            # Filter for BESS only
            if 'ResourceType' in sced.columns:
                sced = sced[sced['ResourceType'] == 'PWRSTR']
            else:
                # Filter by matching resource names
                sced = sced[sced['ResourceName'].isin(data['bess_list'])]
            
            data['sced_gen'] = sced
            print(f"    Loaded {len(sced):,} SCED records for BESS")
        else:
            data['sced_gen'] = pd.DataFrame()
        
        return data
    
    def process_batch(self, batch, data, year):
        """Process a batch of BESS resources"""
        
        results = []
        
        for bess_name in batch:
            # Get settlement point
            sp = self.unit_to_sp.get(bess_name)
            if not sp:
                # Use from DAM data
                bess_dam = data['dam_gen'][data['dam_gen']['ResourceName'] == bess_name]
                if not bess_dam.empty:
                    sp = bess_dam['SettlementPointName'].iloc[0]
                else:
                    sp = 'UNKNOWN'
            
            # Calculate revenues
            result = {
                'year': year,
                'resource_name': bess_name,
                'settlement_point': sp
            }
            
            # DA revenues
            da_revs = self.calc_da_fast(bess_name, data['dam_gen'], data['da_prices'], sp)
            result.update(da_revs)
            
            # AS revenues
            as_revs = self.calc_as_fast(bess_name, data['dam_gen'], data['as_prices'])
            result.update(as_revs)
            
            # RT revenues
            rt_revs = self.calc_rt_fast(bess_name, data['sced_gen'], data['rt_prices'], sp)
            result.update(rt_revs)
            
            # Calculate totals
            result['da_net'] = result['da_revenue'] - result['da_cost']
            result['total_revenue'] = result['da_net'] + result['rt_net'] + result['as_total']
            
            results.append(result)
        
        return results
    
    def calc_da_fast(self, bess_name, dam_gen, da_prices, sp):
        """Fast DA calculation using vectorized operations"""
        
        result = {'da_revenue': 0, 'da_cost': 0}
        
        # Get awards for this BESS
        awards = dam_gen[dam_gen['ResourceName'] == bess_name]
        
        if awards.empty or da_prices.empty:
            return result
        
        # Use settlement point or hub average
        if sp in da_prices.columns:
            price_col = sp
        elif 'HB_BUSAVG' in da_prices.columns:
            price_col = 'HB_BUSAVG'
        else:
            return result
        
        # Vectorized merge and calculation
        awards = awards.set_index('datetime')
        
        # Join with prices (much faster than merge)
        awards['price'] = da_prices[price_col]
        
        # Calculate revenues
        gen_mask = awards['AwardedQuantity'] > 0
        result['da_revenue'] = (awards.loc[gen_mask, 'AwardedQuantity'] * 
                                awards.loc[gen_mask, 'price']).sum()
        
        charge_mask = awards['AwardedQuantity'] < 0
        result['da_cost'] = abs((awards.loc[charge_mask, 'AwardedQuantity'] * 
                                 awards.loc[charge_mask, 'price']).sum())
        
        return result
    
    def calc_as_fast(self, bess_name, dam_gen, as_prices):
        """Fast AS calculation"""
        
        result = {'as_total': 0}
        
        awards = dam_gen[dam_gen['ResourceName'] == bess_name]
        
        if awards.empty or as_prices.empty:
            return result
        
        awards = awards.set_index('datetime')
        
        # AS service mappings
        as_map = {
            'RegUpAwarded': 'REGUP',
            'RegDownAwarded': 'REGDN',
            'RRSAwarded': 'RRS',
            'NonSpinAwarded': 'NSPIN',
            'ECRSAwarded': 'ECRS'
        }
        
        total_as = 0
        for award_col, price_col in as_map.items():
            if award_col in awards.columns and price_col in as_prices.columns:
                # Vectorized calculation
                awards[f'{award_col}_price'] = as_prices[price_col]
                mask = awards[award_col] > 0
                revenue = (awards.loc[mask, award_col] * 
                          awards.loc[mask, f'{award_col}_price']).sum()
                total_as += revenue
        
        result['as_total'] = total_as
        
        return result
    
    def calc_rt_fast(self, bess_name, sced_gen, rt_prices, sp):
        """Fast RT calculation with proper datetime alignment"""
        
        result = {'rt_revenue': 0, 'rt_cost': 0, 'rt_net': 0}
        
        if sced_gen.empty or rt_prices.empty:
            return result
        
        # Get SCED data for this BESS
        sced = sced_gen[sced_gen['ResourceName'] == bess_name]
        
        if sced.empty:
            return result
        
        # Use BasePoint for dispatch
        if 'BasePoint' not in sced.columns:
            return result
        
        # Aggregate SCED to hourly (5-min to hourly)
        sced = sced.set_index('datetime')
        
        # Resample to hourly averages
        hourly_sced = sced[['BasePoint']].resample('H').mean()
        
        # Get RT prices
        if sp in rt_prices.columns:
            price_col = sp
        elif 'HB_BUSAVG' in rt_prices.columns:
            price_col = 'HB_BUSAVG'
        else:
            return result
        
        # Join with prices
        hourly_sced['price'] = rt_prices[price_col]
        
        # Calculate RT revenues (dispatch * price * hours)
        gen_mask = hourly_sced['BasePoint'] > 0
        result['rt_revenue'] = (hourly_sced.loc[gen_mask, 'BasePoint'] * 
                               hourly_sced.loc[gen_mask, 'price']).sum()
        
        charge_mask = hourly_sced['BasePoint'] < 0
        result['rt_cost'] = abs((hourly_sced.loc[charge_mask, 'BasePoint'] * 
                                 hourly_sced.loc[charge_mask, 'price']).sum())
        
        result['rt_net'] = result['rt_revenue'] - result['rt_cost']
        
        return result
    
    def save_and_display_results(self, df):
        """Save and display results"""
        
        # Save files
        parquet_file = self.output_dir / 'optimized_bess_revenues.parquet'
        csv_file = self.output_dir / 'optimized_bess_revenues.csv'
        
        df.to_parquet(parquet_file, index=False)
        df.to_csv(csv_file, index=False)
        
        # Display summary
        print("\n" + "="*100)
        print("OPTIMIZED RESULTS - INCLUDING DA, RT, AND AS")
        print("="*100)
        
        # Aggregate by resource
        agg = df.groupby('resource_name').agg({
            'total_revenue': 'sum',
            'da_net': 'sum',
            'rt_net': 'sum',
            'as_total': 'sum',
            'settlement_point': 'first'
        }).sort_values('total_revenue', ascending=False)
        
        print("\nTOP 25 BESS PROJECTS")
        print("-"*100)
        print(f"{'Rank':<5} {'Resource':<25} {'Total':>15} {'DA':>12} {'RT':>12} {'AS':>12} {'Settlement':<20}")
        print("-"*100)
        
        for idx, (name, row) in enumerate(agg.head(25).iterrows(), 1):
            print(f"{idx:<5d} {name:<25} ${row['total_revenue']:>14,.0f} "
                  f"${row['da_net']:>11,.0f} ${row['rt_net']:>11,.0f} "
                  f"${row['as_total']:>11,.0f} {row['settlement_point']:<20}")
        
        # Summary
        print("\n" + "="*100)
        print("SUMMARY")
        print("="*100)
        
        total = df['total_revenue'].sum()
        da_total = df['da_net'].sum()
        rt_total = df['rt_net'].sum()
        as_total = df['as_total'].sum()
        
        print(f"Total Revenue: ${total:,.0f}")
        print(f"  DA: ${da_total:,.0f} ({da_total/total*100:.1f}%)")
        print(f"  RT: ${rt_total:,.0f} ({rt_total/total*100:.1f}%)")
        print(f"  AS: ${as_total:,.0f} ({as_total/total*100:.1f}%)")
        
        print(f"\n✅ Results saved to:")
        print(f"   {parquet_file}")
        print(f"   {csv_file}")

def main():
    calculator = OptimizedBessCalculator()
    calculator.calculate_all_fast()

if __name__ == '__main__':
    import time
    start = time.time()
    main()
    elapsed = time.time() - start
    print(f"\n⏱️  Total processing time: {elapsed:.1f} seconds")
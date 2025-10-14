#!/usr/bin/env python3
"""
FINAL Complete BESS Revenue Calculator
Properly handles DAM Load Resources for charging costs/revenues
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class CompleteBessCalculatorFinal:
    def __init__(self):
        self.data_dir = Path('/home/enrico/data/ERCOT_data')
        self.rollup_dir = self.data_dir / 'rollup_files'
        self.output_dir = self.data_dir / 'bess_analysis'
        self.output_dir.mkdir(exist_ok=True)
        
        # Load mappings
        self.load_mappings()
        
    def load_mappings(self):
        """Load settlement point and resource mappings"""
        # Settlement point mapping
        mapping_file = self.data_dir / 'Settlement_Points_List_and_Electrical_Buses_Mapping/latest_mapping/SP_List_EB_Mapping/gen_node_map.csv'
        sp_map = pd.read_csv(mapping_file)
        self.unit_to_sp = dict(zip(sp_map['UNIT_NAME'], sp_map['RESOURCE_NODE']))
        
        # Build BESS to Load Resource mapping
        self.build_load_resource_mapping()
        
    def build_load_resource_mapping(self):
        """Build mapping of BESS Gen to Load resources"""
        self.gen_to_load = {}
        
        # Common patterns
        # FLOWERII_BESS1 -> FLOWERII_LD1
        # BATCAVE_BES1 -> BATCAVE_LD1
        # RRANCHES_UNIT1 -> RRANCHES_LD1, RRANCHES_LD2
        
        # Load actual DAM Load data to find mappings
        dam_load_file = self.rollup_dir / 'DAM_Load_Resources/2024.parquet'
        if dam_load_file.exists():
            dam_load = pd.read_parquet(dam_load_file)
            load_resources = dam_load['Load Resource Name'].unique()
            
            # Build mapping based on name patterns
            for load_res in load_resources:
                if '_LD' in load_res:
                    # Extract base name
                    base = load_res.split('_LD')[0]
                    
                    # Find matching gen resources
                    gen_patterns = [
                        f'{base}_BESS1', f'{base}_BESS2',
                        f'{base}_BES1', f'{base}_BES2',
                        f'{base}_UNIT1', f'{base}_UNIT2',
                        f'{base}_ESS1', f'{base}_ESS2'
                    ]
                    
                    for gen_name in gen_patterns:
                        if gen_name not in self.gen_to_load:
                            self.gen_to_load[gen_name] = []
                        self.gen_to_load[gen_name].append(load_res)
            
            logger.info(f"Built Gen->Load mapping for {len(self.gen_to_load)} BESS resources")
    
    def calculate_complete_revenues(self, years=[2023, 2024]):
        """Calculate complete revenues including proper DAM Load"""
        
        print("="*100)
        print("FINAL COMPLETE BESS REVENUE CALCULATOR")
        print("Including DAM Load Resources for Accurate DA Revenue")
        print("="*100)
        
        all_results = []
        
        for year in years:
            print(f"\nProcessing {year}...")
            
            # Load all data
            data = self.load_year_data(year)
            if not data:
                continue
            
            # Process each BESS
            bess_list = data['bess_list']
            print(f"  Processing {len(bess_list)} BESS resources...")
            
            for idx, bess_name in enumerate(bess_list):
                if (idx + 1) % 50 == 0:
                    print(f"    Processed {idx + 1}/{len(bess_list)}...")
                
                result = self.calculate_bess_revenue(bess_name, data, year)
                if result:
                    all_results.append(result)
        
        # Create results dataframe
        if all_results:
            df = pd.DataFrame(all_results)
            self.display_results(df)
            
            # Save results
            output_file = self.output_dir / 'final_complete_bess_revenues.parquet'
            df.to_parquet(output_file, index=False)
            
            csv_file = self.output_dir / 'final_complete_bess_revenues.csv'
            df.to_csv(csv_file, index=False)
            
            print(f"\n✅ Results saved to:")
            print(f"   {output_file}")
            print(f"   {csv_file}")
            
            return df
        
        return pd.DataFrame()
    
    def load_year_data(self, year):
        """Load all data for a year"""
        data = {}
        
        # Load DAM Gen
        dam_gen_file = self.rollup_dir / 'DAM_Gen_Resources' / f'{year}.parquet'
        if not dam_gen_file.exists():
            return None
        
        dam_gen = pd.read_parquet(dam_gen_file)
        bess_gen = dam_gen[dam_gen['ResourceType'] == 'PWRSTR'].copy()
        
        if 'DeliveryDate' in bess_gen.columns:
            bess_gen['datetime'] = pd.to_datetime(bess_gen['DeliveryDate'])
        
        data['dam_gen'] = bess_gen
        data['bess_list'] = bess_gen['ResourceName'].unique().tolist()
        
        # Load DAM Load
        print(f"  Loading DAM Load Resources...")
        dam_load_file = self.rollup_dir / 'DAM_Load_Resources' / f'{year}.parquet'
        if dam_load_file.exists():
            dam_load = pd.read_parquet(dam_load_file)
            
            # Fix datetime
            if 'DeliveryDate' in dam_load.columns:
                dam_load['datetime'] = pd.to_datetime(dam_load['DeliveryDate'])
            elif 'Delivery Date' in dam_load.columns:
                dam_load['datetime'] = pd.to_datetime(dam_load['Delivery Date'])
            
            data['dam_load'] = dam_load
            print(f"    Loaded {len(dam_load):,} DAM Load records")
        else:
            data['dam_load'] = pd.DataFrame()
        
        # Load prices
        da_prices = pd.read_parquet(self.rollup_dir / 'flattened' / f'DA_prices_{year}.parquet')
        if 'datetime_ts' in da_prices.columns:
            da_prices['datetime'] = pd.to_datetime(da_prices['datetime_ts'])
        data['da_prices'] = da_prices
        
        # Load AS prices
        as_file = self.rollup_dir / 'flattened' / f'AS_prices_{year}.parquet'
        if as_file.exists():
            as_prices = pd.read_parquet(as_file)
            if 'datetime_ts' in as_prices.columns:
                as_prices['datetime'] = pd.to_datetime(as_prices['datetime_ts'])
            data['as_prices'] = as_prices
        else:
            data['as_prices'] = pd.DataFrame()
        
        # Load RT prices (simplified - hourly)
        rt_file = self.rollup_dir / 'flattened' / f'RT_prices_hourly_{year}.parquet'
        if rt_file.exists():
            rt_prices = pd.read_parquet(rt_file)
            if 'datetime_ts' in rt_prices.columns:
                rt_prices['datetime'] = pd.to_datetime(rt_prices['datetime_ts'])
            data['rt_prices'] = rt_prices
        else:
            data['rt_prices'] = pd.DataFrame()
        
        # Load SCED
        sced_file = self.rollup_dir / 'SCED_Gen_Resources' / f'{year}.parquet'
        if sced_file.exists():
            sced = pd.read_parquet(sced_file)
            
            # Fix datetime
            if 'datetime' in sced.columns and sced['datetime'].dtype == 'object':
                sced['datetime'] = pd.to_datetime(sced['datetime'], format='%m/%d/%Y %H:%M:%S')
            
            # Filter for BESS
            if 'ResourceType' in sced.columns:
                sced = sced[sced['ResourceType'] == 'PWRSTR']
            
            data['sced_gen'] = sced
        else:
            data['sced_gen'] = pd.DataFrame()
        
        return data
    
    def calculate_bess_revenue(self, bess_name, data, year):
        """Calculate complete revenue for a BESS including Load resources"""
        
        # Get settlement point
        sp = self.unit_to_sp.get(bess_name)
        if not sp:
            bess_data = data['dam_gen'][data['dam_gen']['ResourceName'] == bess_name]
            if not bess_data.empty:
                sp = bess_data['SettlementPointName'].iloc[0]
            else:
                sp = 'UNKNOWN'
        
        result = {
            'year': year,
            'resource_name': bess_name,
            'settlement_point': sp,
            'da_gen_revenue': 0,  # Generation (discharge) revenue
            'da_load_cost': 0,    # Load (charging) cost
            'da_net': 0,
            'rt_revenue': 0,
            'rt_cost': 0,
            'rt_net': 0,
            'as_total': 0,
            'total_revenue': 0
        }
        
        # Calculate DA Generation revenue
        gen_awards = data['dam_gen'][data['dam_gen']['ResourceName'] == bess_name]
        
        if not gen_awards.empty:
            # Get price column
            if sp in data['da_prices'].columns:
                price_col = sp
            elif 'HB_BUSAVG' in data['da_prices'].columns:
                price_col = 'HB_BUSAVG'
            else:
                price_col = None
            
            if price_col:
                merged = gen_awards.merge(data['da_prices'][['datetime', price_col]], 
                                         on='datetime', how='left')
                
                # Generation revenue (positive awards)
                gen_mask = merged['AwardedQuantity'] > 0
                result['da_gen_revenue'] = (merged.loc[gen_mask, 'AwardedQuantity'] * 
                                           merged.loc[gen_mask, price_col]).sum()
                
                # Some BESS might have negative awards in Gen (charging)
                charge_mask = merged['AwardedQuantity'] < 0
                if charge_mask.any():
                    result['da_load_cost'] += abs((merged.loc[charge_mask, 'AwardedQuantity'] * 
                                                   merged.loc[charge_mask, price_col]).sum())
        
        # Calculate DA Load cost (charging) - THIS IS THE KEY FIX
        if not data['dam_load'].empty and bess_name in self.gen_to_load:
            load_resources = self.gen_to_load[bess_name]
            
            for load_res in load_resources:
                load_awards = data['dam_load'][data['dam_load']['Load Resource Name'] == load_res]
                
                if not load_awards.empty:
                    # Get the award quantity column
                    award_col = None
                    for col in ['Awarded Quantity', 'Max Power Consumption for Load Resource', 
                               'Low Power Consumption for Load Resource']:
                        if col in load_awards.columns:
                            award_col = col
                            break
                    
                    if award_col and price_col:
                        merged = load_awards.merge(data['da_prices'][['datetime', price_col]], 
                                                  on='datetime', how='left')
                        
                        # Load cost (always positive in Load file, represents charging)
                        load_cost = (merged[award_col] * merged[price_col]).sum()
                        result['da_load_cost'] += abs(load_cost)
        
        # Calculate DA net
        result['da_net'] = result['da_gen_revenue'] - result['da_load_cost']
        
        # Calculate AS revenues
        if not data['as_prices'].empty and not gen_awards.empty:
            as_map = {
                'RegUpAwarded': 'REGUP',
                'RegDownAwarded': 'REGDN',
                'RRSAwarded': 'RRS',
                'NonSpinAwarded': 'NSPIN',
                'ECRSAwarded': 'ECRS'
            }
            
            for award_col, price_col in as_map.items():
                if award_col in gen_awards.columns and price_col in data['as_prices'].columns:
                    merged = gen_awards[['datetime', award_col]].merge(
                        data['as_prices'][['datetime', price_col]], 
                        on='datetime', how='left'
                    )
                    result['as_total'] += (merged[award_col] * merged[price_col]).sum()
        
        # Calculate RT revenues (simplified)
        if not data['sced_gen'].empty:
            sced = data['sced_gen'][data['sced_gen']['ResourceName'] == bess_name]
            
            if not sced.empty and 'BasePoint' in sced.columns and not data['rt_prices'].empty:
                # Aggregate to hourly
                sced = sced.set_index('datetime')
                hourly = sced[['BasePoint']].resample('H').mean()
                
                if sp in data['rt_prices'].columns:
                    rt_price_col = sp
                elif 'HB_BUSAVG' in data['rt_prices'].columns:
                    rt_price_col = 'HB_BUSAVG'
                else:
                    rt_price_col = None
                
                if rt_price_col:
                    rt_prices_indexed = data['rt_prices'].set_index('datetime')
                    hourly['price'] = rt_prices_indexed[rt_price_col]
                    
                    # RT revenue/cost
                    gen_mask = hourly['BasePoint'] > 0
                    result['rt_revenue'] = (hourly.loc[gen_mask, 'BasePoint'] * 
                                          hourly.loc[gen_mask, 'price']).sum()
                    
                    charge_mask = hourly['BasePoint'] < 0
                    result['rt_cost'] = abs((hourly.loc[charge_mask, 'BasePoint'] * 
                                            hourly.loc[charge_mask, 'price']).sum())
        
        result['rt_net'] = result['rt_revenue'] - result['rt_cost']
        
        # Calculate total
        result['total_revenue'] = result['da_net'] + result['rt_net'] + result['as_total']
        
        return result
    
    def display_results(self, df):
        """Display comprehensive results"""
        
        print("\n" + "="*100)
        print("COMPLETE BESS REVENUE ANALYSIS - WITH PROPER DAM LOAD ACCOUNTING")
        print("="*100)
        
        # Aggregate by resource
        agg = df.groupby('resource_name').agg({
            'da_gen_revenue': 'sum',
            'da_load_cost': 'sum',
            'da_net': 'sum',
            'rt_net': 'sum',
            'as_total': 'sum',
            'total_revenue': 'sum',
            'settlement_point': 'first'
        }).sort_values('total_revenue', ascending=False)
        
        print("\nTOP 20 BESS PROJECTS")
        print("-"*100)
        print(f"{'Rank':<5} {'Resource':<20} {'Total':>12} {'DA Gen':>10} {'DA Load':>10} {'DA Net':>10} {'RT Net':>10} {'AS':>10}")
        print("-"*100)
        
        for idx, (name, row) in enumerate(agg.head(20).iterrows(), 1):
            print(f"{idx:<5d} {name:<20} ${row['total_revenue']:>11,.0f} "
                  f"${row['da_gen_revenue']:>9,.0f} ${row['da_load_cost']:>9,.0f} "
                  f"${row['da_net']:>9,.0f} ${row['rt_net']:>9,.0f} ${row['as_total']:>9,.0f}")
        
        # Summary statistics
        print("\n" + "="*100)
        print("SUMMARY")
        print("="*100)
        
        total = df['total_revenue'].sum()
        da_gen = df['da_gen_revenue'].sum()
        da_load = df['da_load_cost'].sum()
        da_net = df['da_net'].sum()
        rt_net = df['rt_net'].sum()
        as_total = df['as_total'].sum()
        
        print(f"Total Revenue: ${total:,.0f}")
        print(f"  DA Generation Revenue: ${da_gen:,.0f}")
        print(f"  DA Load Cost: ${da_load:,.0f}")
        print(f"  DA Net: ${da_net:,.0f} ({da_net/total*100:.1f}%)")
        print(f"  RT Net: ${rt_net:,.0f} ({rt_net/total*100:.1f}%)")
        print(f"  AS: ${as_total:,.0f} ({as_total/total*100:.1f}%)")
        
        # Check for missing load data
        missing_load = agg[agg['da_load_cost'] == 0]
        if len(missing_load) > 0:
            print(f"\n⚠️  Warning: {len(missing_load)} BESS have $0 load cost - may be missing Load Resource mapping")
            print(f"   Examples: {missing_load.head(5).index.tolist()}")

def main():
    calculator = CompleteBessCalculatorFinal()
    calculator.calculate_complete_revenues(years=[2024])  # Test with 2024 first

if __name__ == '__main__':
    import time
    start = time.time()
    main()
    elapsed = time.time() - start
    print(f"\n⏱️  Processing time: {elapsed:.1f} seconds")
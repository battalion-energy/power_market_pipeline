#!/usr/bin/env python3
"""
CORRECTED BESS Revenue Calculator
Includes DAM charging from Energy Bid Awards at settlement points
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

class CorrectedBessCalculator:
    def __init__(self):
        self.data_dir = Path('/home/enrico/data/ERCOT_data')
        self.rollup_dir = self.data_dir / 'rollup_files'
        self.output_dir = self.data_dir / 'bess_analysis'
        self.output_dir.mkdir(exist_ok=True)
        
        # Load mappings
        self.load_mappings()
        
    def load_mappings(self):
        """Load settlement point mappings"""
        mapping_file = self.data_dir / 'Settlement_Points_List_and_Electrical_Buses_Mapping/latest_mapping/SP_List_EB_Mapping/gen_node_map.csv'
        sp_map = pd.read_csv(mapping_file)
        self.unit_to_sp = dict(zip(sp_map['UNIT_NAME'], sp_map['RESOURCE_NODE']))
        logger.info(f"Loaded {len(self.unit_to_sp)} unit to settlement point mappings")
    
    def get_dam_charging_from_energy_bids(self, settlement_point, year):
        """
        Get DAM charging from Energy Bid Awards (negative values)
        This is the KEY INNOVATION - Load Resources charge at the same settlement point!
        """
        energy_bid_file = self.rollup_dir / 'DAM_Energy_Bid_Awards' / f'{year}.parquet'
        
        if not energy_bid_file.exists():
            logger.warning(f"Energy Bid Awards file not found for {year}, trying CSV approach")
            return self.get_dam_charging_from_csv(settlement_point, year)
        
        # Load Energy Bid Awards
        energy_bids = pd.read_parquet(energy_bid_file)
        
        # Filter for this settlement point
        sp_bids = energy_bids[energy_bids['SettlementPoint'] == settlement_point]
        
        # Negative awards = Load Resource charging!
        charging = sp_bids[sp_bids['EnergyBidAwardMW'] < 0].copy()
        charging['charging_mw'] = abs(charging['EnergyBidAwardMW'])
        
        logger.info(f"  Found {len(charging)} charging periods from Energy Bid Awards")
        
        return charging
    
    def get_dam_charging_from_csv(self, settlement_point, year):
        """Fallback: Read directly from CSV files"""
        csv_dir = self.data_dir / '60-Day_DAM_Disclosure_Reports' / 'csv'
        
        all_charging = []
        
        # Pattern for Energy Bid Award files
        for csv_file in csv_dir.glob(f'*EnergyBidAward*{year%100:02d}.csv'):
            try:
                df = pd.read_csv(csv_file)
                
                # Filter for settlement point and negative awards
                sp_data = df[df['Settlement Point'] == settlement_point]
                charging = sp_data[sp_data['Energy Only Bid Award in MW'] < 0]
                
                if not charging.empty:
                    charging_clean = charging[['Delivery Date', 'Hour Ending', 
                                              'Energy Only Bid Award in MW', 
                                              'Settlement Point Price']].copy()
                    charging_clean['charging_mw'] = abs(charging_clean['Energy Only Bid Award in MW'])
                    all_charging.append(charging_clean)
                    
            except Exception as e:
                logger.debug(f"Error reading {csv_file}: {e}")
        
        if all_charging:
            result = pd.concat(all_charging, ignore_index=True)
            logger.info(f"  Found {len(result)} charging records from CSV files")
            return result
        
        return pd.DataFrame()
    
    def calculate_complete_bess_revenue(self, bess_name, year):
        """Calculate complete BESS revenue including DAM charging costs"""
        
        result = {
            'year': year,
            'resource_name': bess_name,
            'settlement_point': None,
            'dam_discharge_revenue': 0,
            'dam_charge_cost': 0,
            'dam_net': 0,
            'rt_discharge_revenue': 0,
            'rt_charge_cost': 0,
            'rt_net': 0,
            'as_revenue': 0,
            'total_net_revenue': 0
        }
        
        # 1. Get DAM Gen data and settlement point
        dam_gen_file = self.rollup_dir / 'DAM_Gen_Resources' / f'{year}.parquet'
        if not dam_gen_file.exists():
            logger.warning(f"DAM Gen file not found for {year}")
            return result
            
        dam_gen = pd.read_parquet(dam_gen_file)
        bess_gen = dam_gen[dam_gen['ResourceName'] == bess_name]
        
        if bess_gen.empty:
            logger.warning(f"No DAM Gen data for {bess_name}")
            return result
        
        # Get settlement point from Gen Resource data
        if 'SettlementPointName' in bess_gen.columns:
            settlement_point = bess_gen['SettlementPointName'].iloc[0]
        else:
            # Fallback to mapping
            settlement_point = self.unit_to_sp.get(bess_name)
        
        if not settlement_point:
            logger.warning(f"No settlement point found for {bess_name}")
            return result
            
        result['settlement_point'] = settlement_point
        logger.info(f"\nProcessing {bess_name} -> {settlement_point}")
        
        # 2. Load prices
        da_prices = pd.read_parquet(self.rollup_dir / 'flattened' / f'DA_prices_{year}.parquet')
        
        # Get price column
        if settlement_point in da_prices.columns:
            price_col = settlement_point
        elif 'HB_BUSAVG' in da_prices.columns:
            price_col = 'HB_BUSAVG'
            logger.info(f"  Using HB_BUSAVG for pricing (settlement point not in price file)")
        else:
            logger.warning(f"  No price data available")
            return result
        
        # 3. Calculate DAM discharge revenue (from Gen Awards)
        if 'datetime' in bess_gen.columns:
            bess_gen = bess_gen.set_index('datetime')
        elif 'DeliveryDate' in bess_gen.columns:
            bess_gen['datetime'] = pd.to_datetime(bess_gen['DeliveryDate'])
            bess_gen = bess_gen.set_index('datetime')
            
        if 'datetime_ts' in da_prices.columns:
            da_prices['datetime'] = pd.to_datetime(da_prices['datetime_ts'])
            da_prices = da_prices.set_index('datetime')
            
        # Join with prices
        discharge_data = bess_gen[bess_gen['AwardedQuantity'] > 0][['AwardedQuantity']]
        if not discharge_data.empty:
            discharge_with_price = discharge_data.join(da_prices[[price_col]], how='left')
            result['dam_discharge_revenue'] = (discharge_with_price['AwardedQuantity'] * 
                                              discharge_with_price[price_col]).sum()
            logger.info(f"  DAM discharge: {len(discharge_data)} hours, ${result['dam_discharge_revenue']:,.0f}")
        
        # 4. Calculate DAM charging cost (from Energy Bid Awards) - THE KEY FIX!
        dam_charging = self.get_dam_charging_from_energy_bids(settlement_point, year)
        
        if not dam_charging.empty:
            # Convert dates if needed
            if 'Delivery Date' in dam_charging.columns:
                dam_charging['datetime'] = pd.to_datetime(dam_charging['Delivery Date'])
                if 'Hour Ending' in dam_charging.columns:
                    dam_charging['datetime'] += pd.to_timedelta(dam_charging['Hour Ending'] - 1, unit='h')
            
            # Calculate charging cost
            if 'Settlement Point Price' in dam_charging.columns:
                # Price already in the award data
                result['dam_charge_cost'] = (dam_charging['charging_mw'] * 
                                            dam_charging['Settlement Point Price']).sum()
            else:
                # Need to join with price data
                dam_charging = dam_charging.set_index('datetime')
                charge_with_price = dam_charging.join(da_prices[[price_col]], how='left')
                result['dam_charge_cost'] = (charge_with_price['charging_mw'] * 
                                            charge_with_price[price_col]).sum()
            
            logger.info(f"  DAM charging: {len(dam_charging)} hours, ${result['dam_charge_cost']:,.0f}")
        else:
            # Infer charging from hours with zero discharge
            zero_discharge_hours = bess_gen[bess_gen['AwardedQuantity'] == 0]
            if not zero_discharge_hours.empty:
                # Assume charging during low-price hours
                zero_with_price = zero_discharge_hours.join(da_prices[[price_col]], how='left')
                low_price_threshold = da_prices[price_col].quantile(0.25)
                
                charging_hours = zero_with_price[zero_with_price[price_col] < low_price_threshold]
                if not charging_hours.empty:
                    # Estimate charging at 80% of max capacity
                    estimated_charge_mw = 80  # Default assumption
                    result['dam_charge_cost'] = len(charging_hours) * estimated_charge_mw * charging_hours[price_col].mean()
                    logger.info(f"  DAM charging (inferred): {len(charging_hours)} hours, ${result['dam_charge_cost']:,.0f}")
        
        # 5. Calculate DAM net
        result['dam_net'] = result['dam_discharge_revenue'] - result['dam_charge_cost']
        
        # 6. Calculate RT revenues (from SCED)
        sced_file = self.rollup_dir / 'SCED_Gen_Resources' / f'{year}.parquet'
        if sced_file.exists():
            sced = pd.read_parquet(sced_file)
            
            # Fix datetime if needed
            if 'datetime' in sced.columns and sced['datetime'].dtype == 'object':
                sced['datetime'] = pd.to_datetime(sced['datetime'], format='%m/%d/%Y %H:%M:%S')
            
            bess_sced = sced[sced['ResourceName'] == bess_name]
            
            if not bess_sced.empty and 'BasePoint' in bess_sced.columns:
                # Aggregate to hourly
                bess_sced = bess_sced.set_index('datetime')
                hourly = bess_sced[['BasePoint']].resample('h').mean()
                
                # Load RT prices
                rt_file = self.rollup_dir / 'flattened' / f'RT_prices_hourly_{year}.parquet'
                if rt_file.exists():
                    rt_prices = pd.read_parquet(rt_file)
                    if 'datetime_ts' in rt_prices.columns:
                        rt_prices['datetime'] = pd.to_datetime(rt_prices['datetime_ts'])
                        rt_prices = rt_prices.set_index('datetime')
                    
                    # Get appropriate price column
                    if settlement_point in rt_prices.columns:
                        rt_price_col = settlement_point
                    elif 'HB_BUSAVG' in rt_prices.columns:
                        rt_price_col = 'HB_BUSAVG'
                    else:
                        rt_price_col = None
                    
                    if rt_price_col:
                        hourly = hourly.join(rt_prices[[rt_price_col]], how='left')
                        
                        # Discharge revenue
                        discharge = hourly[hourly['BasePoint'] > 0]
                        if not discharge.empty:
                            result['rt_discharge_revenue'] = (discharge['BasePoint'] * 
                                                             discharge[rt_price_col]).sum()
                        
                        # Charging cost (negative BasePoint - rare but exists)
                        charge = hourly[hourly['BasePoint'] < 0]
                        if not charge.empty:
                            result['rt_charge_cost'] = abs((charge['BasePoint'] * 
                                                           charge[rt_price_col]).sum())
                
                logger.info(f"  RT discharge: ${result['rt_discharge_revenue']:,.0f}")
                logger.info(f"  RT charge: ${result['rt_charge_cost']:,.0f}")
        
        # 7. Calculate RT net
        result['rt_net'] = result['rt_discharge_revenue'] - result['rt_charge_cost']
        
        # 8. Calculate AS revenues
        as_file = self.rollup_dir / 'flattened' / f'AS_prices_{year}.parquet'
        if as_file.exists() and not bess_gen.empty:
            as_prices = pd.read_parquet(as_file)
            if 'datetime_ts' in as_prices.columns:
                as_prices['datetime'] = pd.to_datetime(as_prices['datetime_ts'])
                as_prices = as_prices.set_index('datetime')
            
            # AS revenue mapping
            as_map = {
                'RegUpAwarded': 'REGUP',
                'RegDownAwarded': 'REGDN',
                'RRSAwarded': 'RRS',
                'NonSpinAwarded': 'NSPIN',
                'ECRSAwarded': 'ECRS'
            }
            
            for award_col, price_col in as_map.items():
                if award_col in bess_gen.columns and price_col in as_prices.columns:
                    as_data = bess_gen[bess_gen[award_col] > 0][[award_col]]
                    if not as_data.empty:
                        as_with_price = as_data.join(as_prices[[price_col]], how='left')
                        as_rev = (as_with_price[award_col] * as_with_price[price_col]).sum()
                        result['as_revenue'] += as_rev
            
            logger.info(f"  AS revenue: ${result['as_revenue']:,.0f}")
        
        # 9. Calculate total
        result['total_net_revenue'] = result['dam_net'] + result['rt_net'] + result['as_revenue']
        
        logger.info(f"  TOTAL NET: ${result['total_net_revenue']:,.0f}")
        
        return result
    
    def run_analysis(self, years=None, limit=None):
        """Run complete BESS revenue analysis"""
        
        if years is None:
            years = [2024]
        
        print("="*100)
        print("CORRECTED BESS REVENUE CALCULATOR")
        print("Including DAM Charging from Energy Bid Awards")
        print("="*100)
        
        all_results = []
        
        for year in years:
            print(f"\nProcessing year {year}...")
            
            # Get list of BESS from DAM Gen Resources
            dam_gen_file = self.rollup_dir / 'DAM_Gen_Resources' / f'{year}.parquet'
            if not dam_gen_file.exists():
                print(f"  DAM Gen file not found for {year}")
                continue
            
            dam_gen = pd.read_parquet(dam_gen_file)
            bess_list = dam_gen[dam_gen['ResourceType'] == 'PWRSTR']['ResourceName'].unique()
            
            print(f"  Found {len(bess_list)} BESS resources")
            
            if limit:
                bess_list = bess_list[:limit]
                print(f"  Limited to {limit} for testing")
            
            for bess_name in bess_list:
                result = self.calculate_complete_bess_revenue(bess_name, year)
                all_results.append(result)
        
        # Create results dataframe
        df = pd.DataFrame(all_results)
        
        # Display leaderboard
        self.display_leaderboard(df)
        
        # Save results
        output_file = self.output_dir / 'corrected_bess_revenues.parquet'
        df.to_parquet(output_file, index=False)
        
        csv_file = self.output_dir / 'corrected_bess_revenues.csv'
        df.to_csv(csv_file, index=False)
        
        print(f"\n✅ Results saved to:")
        print(f"   {output_file}")
        print(f"   {csv_file}")
        
        return df
    
    def display_leaderboard(self, df):
        """Display revenue leaderboard"""
        
        print("\n" + "="*100)
        print("BESS REVENUE LEADERBOARD (CORRECTED)")
        print("="*100)
        
        # Aggregate by resource
        agg = df.groupby('resource_name').agg({
            'dam_discharge_revenue': 'sum',
            'dam_charge_cost': 'sum',
            'dam_net': 'sum',
            'rt_discharge_revenue': 'sum',
            'rt_charge_cost': 'sum',
            'rt_net': 'sum',
            'as_revenue': 'sum',
            'total_net_revenue': 'sum',
            'settlement_point': 'first'
        }).sort_values('total_net_revenue', ascending=False)
        
        print(f"\n{'Rank':<5} {'Resource':<20} {'Total Net':>12} {'DAM Net':>12} {'RT Net':>12} {'AS':>12}")
        print("-"*80)
        
        for idx, (name, row) in enumerate(agg.head(20).iterrows(), 1):
            print(f"{idx:<5d} {name:<20} ${row['total_net_revenue']:>11,.0f} "
                  f"${row['dam_net']:>11,.0f} ${row['rt_net']:>11,.0f} ${row['as_revenue']:>11,.0f}")
        
        # Summary statistics
        print("\n" + "="*100)
        print("SUMMARY STATISTICS")
        print("="*100)
        
        print(f"Total BESS analyzed: {len(agg)}")
        print(f"Total Net Revenue: ${agg['total_net_revenue'].sum():,.0f}")
        print(f"  DAM Net: ${agg['dam_net'].sum():,.0f}")
        print(f"    Discharge Revenue: ${agg['dam_discharge_revenue'].sum():,.0f}")
        print(f"    Charging Cost: ${agg['dam_charge_cost'].sum():,.0f}")
        print(f"  RT Net: ${agg['rt_net'].sum():,.0f}")
        print(f"  AS Revenue: ${agg['as_revenue'].sum():,.0f}")
        
        # Check energy balance
        total_discharge = agg['dam_discharge_revenue'].sum() + agg['rt_discharge_revenue'].sum()
        total_charge_cost = agg['dam_charge_cost'].sum() + agg['rt_charge_cost'].sum()
        
        if total_charge_cost > 0:
            efficiency_implied = 1 - (total_charge_cost / total_discharge)
            print(f"\nImplied round-trip efficiency: {efficiency_implied:.1%}")
        
        # Flag potential issues
        zero_charge = agg[agg['dam_charge_cost'] == 0]
        if len(zero_charge) > 0:
            print(f"\n⚠️  {len(zero_charge)} BESS show $0 DAM charging cost - may need Energy Bid Award data")
            print(f"   Examples: {zero_charge.head(5).index.tolist()}")

def main():
    calculator = CorrectedBessCalculator()
    
    # Run for 2024 with limit for testing
    df = calculator.run_analysis(years=[2024], limit=50)
    
    print("\n🎉 Analysis complete! This should show more realistic net revenues with charging costs included.")

if __name__ == '__main__':
    import time
    start = time.time()
    main()
    elapsed = time.time() - start
    print(f"\n⏱️  Processing time: {elapsed:.1f} seconds")
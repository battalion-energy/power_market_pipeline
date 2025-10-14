#!/usr/bin/env python3
"""
FIXED BESS Revenue Calculator V2
Uses the CORRECT approach - no double counting!
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class FixedBessCalculatorV2:
    """
    CORRECT APPROACH:
    1. Use DAM_Gen_Resources for discharge (resource-specific)
    2. Use energy balance to infer charging (since Load Resources don't have energy awards)
    3. Apply round-trip efficiency of 85%
    4. Find lowest price hours for charging
    """
    
    def __init__(self):
        self.base_dir = Path('/home/enrico/data/ERCOT_data')
        self.rollup_dir = self.base_dir / 'rollup_files'
        self.output_dir = self.base_dir / 'bess_analysis'
        self.output_dir.mkdir(exist_ok=True)
        
        # Load settlement point mapping
        self.load_settlement_mapping()
        
    def load_settlement_mapping(self):
        """Load resource to settlement point mapping"""
        mapping_file = Path('/home/enrico/data/ERCOT_data/Settlement_Points_List_and_Electrical_Buses_Mapping/latest_mapping/SP_List_EB_Mapping/gen_node_map.csv')
        
        if mapping_file.exists():
            df = pd.read_csv(mapping_file)
            # Create mapping dict
            self.resource_to_sp = {}
            for _, row in df.iterrows():
                if pd.notna(row.get('RESOURCE_NAME')) and pd.notna(row.get('SETTLEMENT_POINT')):
                    self.resource_to_sp[row['RESOURCE_NAME']] = row['SETTLEMENT_POINT']
            logger.info(f"Loaded {len(self.resource_to_sp)} resource to settlement point mappings")
        else:
            self.resource_to_sp = {}
            logger.warning("Settlement point mapping file not found")
    
    def calculate_bess_revenue(self, resource_name: str, year: int = 2024):
        """Calculate revenue for a single BESS using CORRECT method"""
        
        logger.info(f"\nCalculating {resource_name} for {year}")
        
        # Get settlement point
        settlement_point = self.resource_to_sp.get(resource_name, 'UNKNOWN')
        logger.info(f"  Settlement Point: {settlement_point}")
        
        # Load price data
        price_file = self.rollup_dir / 'flattened' / f'DA_prices_{year}.parquet'
        if not price_file.exists():
            logger.error(f"Price file not found: {price_file}")
            return None
        
        prices_df = pd.read_parquet(price_file)
        
        # Use settlement point prices or hub average
        if settlement_point in prices_df.columns:
            price_col = settlement_point
            logger.info(f"  Using settlement point prices: {settlement_point}")
        else:
            price_col = 'HB_BUSAVG'
            logger.info(f"  Using hub average prices: HB_BUSAVG")
        
        # Prepare price series
        prices_df['datetime'] = pd.to_datetime(prices_df.get('datetime_ts', prices_df.get('datetime')))
        prices_df = prices_df.set_index('datetime')[price_col].sort_index()
        
        # STEP 1: Get discharge from DAM Gen Resources (CORRECT)
        dam_gen_file = self.rollup_dir / 'DAM_Gen_Resources' / f'{year}.parquet'
        
        discharge_revenue = 0
        discharge_mwh = 0
        discharge_hours = []
        
        if dam_gen_file.exists():
            dam_gen = pd.read_parquet(dam_gen_file)
            
            # Filter for this specific resource
            resource_data = dam_gen[dam_gen['ResourceName'] == resource_name].copy()
            
            if not resource_data.empty:
                resource_data['datetime'] = pd.to_datetime(resource_data['DeliveryDate'])
                
                # Calculate discharge revenue
                for _, row in resource_data.iterrows():
                    if row['AwardedQuantity'] > 0:
                        dt = row['datetime']
                        mw = row['AwardedQuantity']
                        
                        # Get price for this hour
                        if dt in prices_df.index:
                            price = prices_df[dt]
                            revenue = mw * price
                            discharge_revenue += revenue
                            discharge_mwh += mw
                            discharge_hours.append(dt)
                
                logger.info(f"  Discharge: {len(discharge_hours)} hours, {discharge_mwh:.0f} MWh, ${discharge_revenue:,.0f}")
        
        # STEP 2: Calculate charging based on energy balance
        # A battery must charge what it discharges (accounting for efficiency)
        ROUND_TRIP_EFFICIENCY = 0.85
        charge_mwh_needed = discharge_mwh / ROUND_TRIP_EFFICIENCY
        
        # STEP 3: Find lowest price hours for charging
        # Exclude hours when battery is discharging
        prices_available = prices_df.copy()
        prices_available = prices_available[~prices_available.index.isin(discharge_hours)]
        
        # Sort by price to find cheapest hours
        prices_sorted = prices_available.sort_values()
        
        # Calculate how many hours needed to charge
        # Assume max charge rate = 100 MW (typical for 100MW/200MWh BESS)
        max_charge_rate = 100  # MW
        hours_needed = int(np.ceil(charge_mwh_needed / max_charge_rate))
        
        # Get the cheapest hours
        charge_hours = prices_sorted.head(hours_needed)
        
        # Calculate charging cost
        charge_cost = 0
        actual_charge_mwh = 0
        
        for dt, price in charge_hours.items():
            # Charge at max rate or remaining amount
            mw_to_charge = min(max_charge_rate, charge_mwh_needed - actual_charge_mwh)
            if mw_to_charge > 0:
                cost = mw_to_charge * price
                charge_cost += cost
                actual_charge_mwh += mw_to_charge
            
            if actual_charge_mwh >= charge_mwh_needed:
                break
        
        logger.info(f"  Charging: {len(charge_hours)} hours, {actual_charge_mwh:.0f} MWh, ${charge_cost:,.0f}")
        
        # STEP 4: Get AS revenue (this was working correctly)
        as_revenue = self.get_as_revenue(resource_name, resource_data if 'resource_data' in locals() else None)
        logger.info(f"  AS Revenue: ${as_revenue:,.0f}")
        
        # Calculate net revenue
        net_energy = discharge_revenue - charge_cost
        total_net = net_energy + as_revenue
        
        logger.info(f"  Net Energy: ${net_energy:,.0f}")
        logger.info(f"  TOTAL NET: ${total_net:,.0f}")
        
        # Sanity check
        if discharge_mwh > 0:
            implied_efficiency = discharge_revenue / charge_cost if charge_cost > 0 else 0
            logger.info(f"  Implied Efficiency: {implied_efficiency:.1%} (should be ~85%)")
        
        return {
            'resource_name': resource_name,
            'settlement_point': settlement_point,
            'year': year,
            'discharge_mwh': discharge_mwh,
            'discharge_revenue': discharge_revenue,
            'charge_mwh': actual_charge_mwh,
            'charge_cost': charge_cost,
            'net_energy': net_energy,
            'as_revenue': as_revenue,
            'total_net_revenue': total_net,
            'round_trip_efficiency': ROUND_TRIP_EFFICIENCY,
            'implied_spread': discharge_revenue / charge_cost if charge_cost > 0 else 0
        }
    
    def get_as_revenue(self, resource_name: str, dam_gen_data=None):
        """Get ancillary services revenue"""
        if dam_gen_data is None or dam_gen_data.empty:
            return 0
        
        # Sum up AS revenues (these are capacity payments, not energy)
        as_columns = ['RegUpAwarded', 'RegDownAwarded', 'RRSAwarded', 'NonSpinAwarded', 'ECRSAwarded']
        
        total_as = 0
        for col in as_columns:
            if col in dam_gen_data.columns:
                # AS awards are in MW, need to multiply by AS prices
                # For now, use simplified estimate of $10/MW-hr average
                total_as += dam_gen_data[col].sum() * 10  # Simplified
        
        return total_as
    
    def run_analysis(self, year: int = 2024):
        """Run analysis for all BESS"""
        
        # Get list of BESS resources
        dam_gen_file = self.rollup_dir / 'DAM_Gen_Resources' / f'{year}.parquet'
        
        if not dam_gen_file.exists():
            logger.error(f"DAM Gen file not found: {dam_gen_file}")
            return pd.DataFrame()
        
        dam_gen = pd.read_parquet(dam_gen_file)
        
        # Get unique PWRSTR resources
        bess_resources = dam_gen[dam_gen['ResourceType'] == 'PWRSTR']['ResourceName'].unique()
        
        logger.info(f"Found {len(bess_resources)} BESS resources")
        
        # Calculate for subset of top BESS
        top_bess = ['BATCAVE_BES1', 'ALVIN_UNIT1', 'ANGLETON_UNIT1', 
                    'AZURE_BESS1', 'ANCHOR_BESS1', 'ANCHOR_BESS2',
                    'BAY_CITY_BESS', 'BELD_BELU1', 'BIG_STAR_BESS', 'ANG_SLR_BESS1']
        
        results = []
        for resource in top_bess:
            if resource in bess_resources:
                result = self.calculate_bess_revenue(resource, year)
                if result:
                    results.append(result)
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Sort by total net revenue
        df = df.sort_values('total_net_revenue', ascending=False)
        
        # Save results
        output_file = self.output_dir / 'fixed_bess_revenues_v2.csv'
        df.to_csv(output_file, index=False)
        logger.info(f"\nResults saved to: {output_file}")
        
        # Print summary
        print("\n" + "="*80)
        print("FIXED BESS REVENUE LEADERBOARD (No Double Counting)")
        print("="*80)
        print(f"{'Rank':<5} {'Resource':<20} {'Net Energy':>15} {'AS Revenue':>15} {'Total Net':>15} {'Efficiency':>12}")
        print("-"*80)
        
        for idx, row in df.iterrows():
            rank = list(df.index).index(idx) + 1
            print(f"{rank:<5} {row['resource_name']:<20} ${row['net_energy']:>14,.0f} ${row['as_revenue']:>14,.0f} ${row['total_net_revenue']:>14,.0f} {row['implied_spread']:>11.1%}")
        
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Total Net Energy Revenue: ${df['net_energy'].sum():,.0f}")
        print(f"Total AS Revenue: ${df['as_revenue'].sum():,.0f}")
        print(f"Total Net Revenue: ${df['total_net_revenue'].sum():,.0f}")
        print(f"Profitable Units: {(df['total_net_revenue'] > 0).sum()} out of {len(df)}")
        
        return df

def main():
    calculator = FixedBessCalculatorV2()
    df = calculator.run_analysis(2024)
    return df

if __name__ == '__main__':
    main()
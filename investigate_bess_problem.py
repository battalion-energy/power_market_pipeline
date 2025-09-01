#!/usr/bin/env python3
"""
Investigate BESS calculation problems
Focus on BATCAVE which shows impossible 391% efficiency
"""

import pandas as pd
import numpy as np
from pathlib import Path

def investigate_batcave():
    """Deep dive into BATCAVE calculations"""
    
    print("\n" + "="*80)
    print("INVESTIGATING BATCAVE_BES1 - SOMETHING IS VERY WRONG")
    print("="*80)
    
    # 1. Check DAM Gen Awards (Discharge)
    print("\n1. DAM GEN AWARDS (Discharge):")
    print("-"*50)
    
    dam_gen_file = Path('/home/enrico/data/ERCOT_data/rollup_files/DAM_Gen_Resources/2024.parquet')
    dam_gen = pd.read_parquet(dam_gen_file)
    
    batcave_gen = dam_gen[dam_gen['ResourceName'] == 'BATCAVE_BES1']
    print(f"Total BATCAVE discharge records: {len(batcave_gen)}")
    print(f"Total MWh from Gen Awards: {batcave_gen['AwardedQuantity'].sum():,.0f}")
    
    # Get actual hours with discharge
    discharge_hours = batcave_gen[batcave_gen['AwardedQuantity'] > 0]
    print(f"Hours with discharge > 0: {len(discharge_hours)}")
    print(f"Average MW when discharging: {discharge_hours['AwardedQuantity'].mean():.1f}")
    
    # 2. Check Energy Bid Awards (should have charging)
    print("\n2. ENERGY BID AWARDS (Charging):")
    print("-"*50)
    
    eba_dir = Path('/home/enrico/data/ERCOT_data/60_Day_DAM_Disclosure/2024/DAM_EnergyBidAwards')
    
    total_charging_mw = 0
    total_discharge_from_eba = 0
    charge_hours = 0
    discharge_hours_eba = 0
    
    if eba_dir.exists():
        csv_files = sorted(eba_dir.glob('*.csv'))[:5]  # Check first 5 files
        
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            
            # Look for BATCAVE_RN
            batcave_eba = df[df['SettlementPoint'] == 'BATCAVE_RN']
            
            if not batcave_eba.empty:
                # Negative = charging
                charging = batcave_eba[batcave_eba['EnergyBidAwardMW'] < 0]
                total_charging_mw += abs(charging['EnergyBidAwardMW'].sum())
                charge_hours += len(charging)
                
                # Positive = discharge (PROBLEM if exists!)
                discharging = batcave_eba[batcave_eba['EnergyBidAwardMW'] > 0]
                if not discharging.empty:
                    print(f"⚠️ FOUND POSITIVE AWARDS in {csv_file.name}: {len(discharging)} hours")
                    total_discharge_from_eba += discharging['EnergyBidAwardMW'].sum()
                    discharge_hours_eba += len(discharging)
    
    print(f"\nFrom Energy Bid Awards files (sample):")
    print(f"  Charging hours: {charge_hours}")
    print(f"  Total charging MW: {total_charging_mw:.0f}")
    print(f"  Discharge hours (should be 0!): {discharge_hours_eba}")
    print(f"  Total discharge MW from EBA: {total_discharge_from_eba:.0f}")
    
    if discharge_hours_eba > 0:
        print("\n🚨 CRITICAL ISSUE: Energy Bid Awards has POSITIVE values!")
        print("   These might be discharge that we're ALREADY counting from DAM Gen!")
    
    # 3. Check price data
    print("\n3. PRICE ANALYSIS:")
    print("-"*50)
    
    price_file = Path('/home/enrico/data/ERCOT_data/rollup_files/flattened/DA_prices_2024.parquet')
    prices = pd.read_parquet(price_file)
    
    # Check if BATCAVE_RN exists in prices
    if 'BATCAVE_RN' in prices.columns:
        batcave_prices = prices['BATCAVE_RN'].dropna()
        print(f"BATCAVE_RN prices found:")
        print(f"  Mean: ${batcave_prices.mean():.2f}")
        print(f"  Min: ${batcave_prices.min():.2f}")
        print(f"  Max: ${batcave_prices.max():.2f}")
    else:
        print("BATCAVE_RN NOT in price file - using HB_BUSAVG")
        if 'HB_BUSAVG' in prices.columns:
            hub_prices = prices['HB_BUSAVG'].dropna()
            print(f"HB_BUSAVG prices:")
            print(f"  Mean: ${hub_prices.mean():.2f}")
    
    # 4. Check what corrected_bess_calculator is actually doing
    print("\n4. CHECKING OUR CALCULATION LOGIC:")
    print("-"*50)
    
    # Simulate the calculation
    print("\nOur calculation appears to be:")
    print("1. Get discharge MW from DAM Gen Awards")
    print("2. Get charging MW from Energy Bid Awards (negative values)")
    print("3. Multiply by prices")
    
    print("\nPOTENTIAL ISSUES IDENTIFIED:")
    print("- If Energy Bid Awards has POSITIVE values, we might be double-counting discharge")
    print("- The 391% efficiency suggests discharge revenue is way too high")
    print("- Some units only charge and never discharge - that's impossible for a battery!")

def check_other_problem_units():
    """Check units that only charge"""
    
    print("\n" + "="*80)
    print("CHECKING UNITS THAT ONLY CHARGE (NO DISCHARGE)")
    print("="*80)
    
    problem_units = ['ANCHOR_BESS1', 'ANG_SLR_BESS1', 'BELD_BELU1', 'BIG_STAR_BESS']
    
    dam_gen_file = Path('/home/enrico/data/ERCOT_data/rollup_files/DAM_Gen_Resources/2024.parquet')
    dam_gen = pd.read_parquet(dam_gen_file)
    
    for unit in problem_units:
        print(f"\n{unit}:")
        
        # Check if unit exists in DAM Gen
        unit_gen = dam_gen[dam_gen['ResourceName'] == unit]
        
        if unit_gen.empty:
            print(f"  ❌ NOT FOUND in DAM Gen Awards!")
            # Try variations
            variations = [unit.replace('_', ''), unit.replace('BESS', 'BES'), unit + '1']
            for var in variations:
                var_gen = dam_gen[dam_gen['ResourceName'].str.contains(var, na=False)]
                if not var_gen.empty:
                    print(f"  Found variation: {var_gen['ResourceName'].unique()}")
        else:
            awards = unit_gen['AwardedQuantity'].sum()
            print(f"  ✅ Found in DAM Gen: {len(unit_gen)} records, {awards:.0f} MWh total")
            
            if awards == 0:
                print(f"  ⚠️ But all awards are ZERO!")

def check_actual_calculation():
    """Check what the corrected calculator is actually doing"""
    
    print("\n" + "="*80)
    print("REVERSE ENGINEERING THE CALCULATION")
    print("="*80)
    
    # Load the corrected revenues
    df = pd.read_csv('/home/enrico/data/ERCOT_data/bess_analysis/corrected_bess_revenues.csv')
    
    # For BATCAVE
    batcave = df[df['resource_name'] == 'BATCAVE_BES1'].iloc[0]
    
    print("\nBATCAVE_BES1 Results:")
    print(f"  Discharge Revenue: ${batcave['dam_discharge_revenue']:,.0f}")
    print(f"  Charge Cost: ${batcave['dam_charge_cost']:,.0f}")
    
    # Try to reverse engineer
    # If discharge revenue is $880,435 and we have 16,992 MWh from DAM Gen
    dam_gen_file = Path('/home/enrico/data/ERCOT_data/rollup_files/DAM_Gen_Resources/2024.parquet')
    dam_gen = pd.read_parquet(dam_gen_file)
    batcave_gen = dam_gen[dam_gen['ResourceName'] == 'BATCAVE_BES1']
    
    total_mwh = batcave_gen['AwardedQuantity'].sum()
    if total_mwh > 0:
        implied_price = batcave['dam_discharge_revenue'] / total_mwh
        print(f"\nImplied average discharge price: ${implied_price:.2f}/MWh")
        
    # Check charge cost
    if batcave['dam_charge_cost'] > 0:
        # Assuming similar MWh for charging
        print(f"\nIf charging MWh = {total_mwh:.0f} (same as discharge):")
        implied_charge_price = batcave['dam_charge_cost'] / total_mwh
        print(f"  Implied average charge price: ${implied_charge_price:.2f}/MWh")
        
        print(f"\n⚠️ PROBLEM: Discharge price / Charge price = {implied_price/implied_charge_price:.1f}x")
        print("   This ratio is way too high! Should be ~1.2-1.5x for arbitrage")
        
    print("\n🔍 HYPOTHESIS: We're calculating charging MWh incorrectly!")
    print("   The charging MWh might be much lower than actual")

if __name__ == '__main__':
    investigate_batcave()
    check_other_problem_units()
    check_actual_calculation()
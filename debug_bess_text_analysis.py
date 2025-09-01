#!/usr/bin/env python3
"""
Debug BESS Revenue Calculations - Text Analysis
Find out why batteries appear unprofitable
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def main():
    print("\n" + "="*80)
    print("BESS REVENUE DEBUGGING - FINDING THE PROBLEM")
    print("="*80)
    
    # Load corrected revenues
    df = pd.read_csv('/home/enrico/data/ERCOT_data/bess_analysis/corrected_bess_revenues.csv')
    
    print("\n1. OVERALL METRICS:")
    print("-"*50)
    print(f"Total DAM Discharge Revenue: ${df['dam_discharge_revenue'].sum():,.0f}")
    print(f"Total DAM Charge Cost: ${df['dam_charge_cost'].sum():,.0f}")
    print(f"Charge/Discharge Ratio: {df['dam_charge_cost'].sum()/df['dam_discharge_revenue'].sum():.2f}")
    print(f"Total AS Revenue: ${df['as_revenue'].sum():,.0f}")
    
    print("\n2. PROBLEM IDENTIFICATION:")
    print("-"*50)
    
    # Units with no discharge but have charging
    zero_discharge = df[(df['dam_discharge_revenue'] == 0) & (df['dam_charge_cost'] > 0)]
    if not zero_discharge.empty:
        print(f"\n⚠️ PROBLEM: {len(zero_discharge)} units charging but NOT discharging:")
        for _, row in zero_discharge.iterrows():
            print(f"  {row['resource_name']:20} - Charged ${row['dam_charge_cost']:,.0f}, Discharged $0")
    
    # Units where charging > discharge (losing money on arbitrage)
    losing_money = df[df['dam_charge_cost'] > df['dam_discharge_revenue']]
    print(f"\n⚠️ PROBLEM: {len(losing_money)}/10 units losing money on energy arbitrage:")
    for _, row in losing_money.iterrows():
        loss = row['dam_charge_cost'] - row['dam_discharge_revenue']
        print(f"  {row['resource_name']:20} - Loss: ${loss:,.0f}")
    
    print("\n3. DEEP DIVE - BATCAVE (Should be most profitable):")
    print("-"*50)
    
    batcave = df[df['resource_name'] == 'BATCAVE_BES1'].iloc[0]
    print(f"BATCAVE_BES1:")
    print(f"  DAM Discharge Revenue: ${batcave['dam_discharge_revenue']:,.0f}")
    print(f"  DAM Charge Cost: ${batcave['dam_charge_cost']:,.0f}")
    print(f"  Net DAM: ${batcave['dam_net']:,.0f}")
    print(f"  AS Revenue: ${batcave['as_revenue']:,.0f}")
    print(f"  Total Net: ${batcave['total_net_revenue']:,.0f}")
    
    # Calculate implied efficiency
    if batcave['dam_charge_cost'] > 0:
        implied_eff = batcave['dam_discharge_revenue'] / batcave['dam_charge_cost']
        print(f"  Implied Efficiency: {implied_eff:.1%}")
        if implied_eff > 1.2:
            print(f"  ✅ This looks reasonable for arbitrage")
        else:
            print(f"  ⚠️ This efficiency is too low for profitable arbitrage")
    
    print("\n4. CHECKING ACTUAL MW DATA:")
    print("-"*50)
    
    # Load DAM Gen data to check MW quantities
    dam_gen_file = Path('/home/enrico/data/ERCOT_data/rollup_files/DAM_Gen_Resources/2024.parquet')
    if dam_gen_file.exists():
        dam_gen = pd.read_parquet(dam_gen_file)
        
        # Get BATCAVE data
        batcave_gen = dam_gen[dam_gen['ResourceName'] == 'BATCAVE_BES1']
        if not batcave_gen.empty:
            print(f"\nBATCAVE_BES1 DAM Gen Awards:")
            print(f"  Total hours with awards: {len(batcave_gen)}")
            print(f"  Total MWh discharged: {batcave_gen['AwardedQuantity'].sum():,.0f}")
            print(f"  Average MW when dispatched: {batcave_gen['AwardedQuantity'].mean():.1f}")
            print(f"  Max MW: {batcave_gen['AwardedQuantity'].max():.1f}")
            
            # Check if this matches revenue
            # Rough check: Revenue / MWh should be reasonable price
            if batcave_gen['AwardedQuantity'].sum() > 0:
                implied_price = batcave['dam_discharge_revenue'] / batcave_gen['AwardedQuantity'].sum()
                print(f"  Implied discharge price: ${implied_price:.2f}/MWh")
                if 20 < implied_price < 100:
                    print(f"  ✅ Price looks reasonable")
                else:
                    print(f"  ⚠️ Price seems off - check calculation")
    
    print("\n5. CHECKING ENERGY BID AWARDS (Charging):")
    print("-"*50)
    
    # Check a sample Energy Bid Awards file
    eba_dir = Path('/home/enrico/data/ERCOT_data/60_Day_DAM_Disclosure/2024/DAM_EnergyBidAwards')
    if eba_dir.exists():
        csv_files = sorted(eba_dir.glob('*.csv'))
        if csv_files:
            # Check first file for BATCAVE
            sample_file = csv_files[0]
            print(f"\nChecking {sample_file.name}...")
            
            try:
                sample_df = pd.read_csv(sample_file)
                
                # Look for BATCAVE settlement point
                batcave_sp = sample_df[sample_df['SettlementPoint'] == 'BATCAVE_RN']
                if not batcave_sp.empty:
                    print(f"\nBATCAVE_RN in Energy Bid Awards:")
                    print(f"  Total records: {len(batcave_sp)}")
                    
                    negative = batcave_sp[batcave_sp['EnergyBidAwardMW'] < 0]
                    positive = batcave_sp[batcave_sp['EnergyBidAwardMW'] > 0]
                    
                    print(f"  Negative awards (charging): {len(negative)} hours")
                    print(f"  Total charging MW: {negative['EnergyBidAwardMW'].sum():.0f}")
                    print(f"  Positive awards: {len(positive)} hours")
                    print(f"  Total positive MW: {positive['EnergyBidAwardMW'].sum():.0f}")
                    
                    # Check if we're counting both positive and negative
                    if len(positive) > 0:
                        print(f"\n⚠️ POTENTIAL ISSUE: Found POSITIVE awards in Energy Bid Awards")
                        print(f"   We might be double-counting discharge!")
            except Exception as e:
                print(f"Error reading sample file: {e}")
    
    print("\n6. HYPOTHESIS TESTING:")
    print("-"*50)
    
    print("\n❓ Hypothesis 1: We're double-counting discharge")
    print("   Check: Are we adding discharge from both DAM Gen AND Energy Bid Awards?")
    
    print("\n❓ Hypothesis 2: We're using wrong prices")
    print("   Check: Are we using settlement point prices or hub average?")
    
    print("\n❓ Hypothesis 3: We're mismatching charge/discharge hours")
    print("   Check: Are charge and discharge in same hours (impossible)?")
    
    print("\n❓ Hypothesis 4: The charging MW quantities are wrong")
    print("   Check: Are negative Energy Bid Awards actually in MWh not MW?")
    
    print("\n7. PRICE CHECK:")
    print("-"*50)
    
    # Load price data
    price_file = Path('/home/enrico/data/ERCOT_data/rollup_files/flattened/DA_prices_2024.parquet')
    if price_file.exists():
        prices = pd.read_parquet(price_file)
        
        if 'HB_BUSAVG' in prices.columns:
            hub_prices = prices['HB_BUSAVG'].dropna()
            print(f"\nHub Average (HB_BUSAVG) Prices:")
            print(f"  Mean: ${hub_prices.mean():.2f}/MWh")
            print(f"  Min: ${hub_prices.min():.2f}/MWh")
            print(f"  Max: ${hub_prices.max():.2f}/MWh")
            print(f"  Std Dev: ${hub_prices.std():.2f}")
            
            # Check for arbitrage opportunity
            low_price_hours = hub_prices[hub_prices < hub_prices.quantile(0.25)]
            high_price_hours = hub_prices[hub_prices > hub_prices.quantile(0.75)]
            
            print(f"\nArbitrage Opportunity:")
            print(f"  Bottom 25% price: ${low_price_hours.mean():.2f}/MWh")
            print(f"  Top 25% price: ${high_price_hours.mean():.2f}/MWh")
            print(f"  Spread: ${high_price_hours.mean() - low_price_hours.mean():.2f}/MWh")
            
            spread_ratio = high_price_hours.mean() / low_price_hours.mean()
            print(f"  Ratio: {spread_ratio:.2f}x")
            
            if spread_ratio > 1.5:
                print(f"  ✅ Good arbitrage opportunity exists")
            else:
                print(f"  ⚠️ Limited arbitrage opportunity")
    
    print("\n8. SETTLEMENT POINT ANALYSIS:")
    print("-"*50)
    
    print("\nSettlement Points Used:")
    for _, row in df.iterrows():
        print(f"  {row['resource_name']:20} -> {row['settlement_point']}")
    
    # Check if any are using wrong settlement points
    if 'NOT_FOUND' in df['settlement_point'].str.upper().values or \
       'UNKNOWN' in df['settlement_point'].str.upper().values:
        print("\n⚠️ PROBLEM: Some units have invalid settlement points!")
    
    print("\n" + "="*80)
    print("RECOMMENDED NEXT STEPS:")
    print("="*80)
    print("\n1. Check if Energy Bid Awards positive values are being added to discharge")
    print("2. Verify MW vs MWh in charging data")
    print("3. Check if we're using correct settlement point prices")
    print("4. Validate that charge/discharge don't occur in same hour")
    print("5. Compare our calculations with ERCOT settlement statements")

if __name__ == '__main__':
    main()
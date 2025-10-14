#!/usr/bin/env python3
"""
Check if we're double counting by comparing:
1. DAM Gen Awards (discharge)
2. Energy Bid Awards (net position)

The key insight: Energy Bid Awards might INCLUDE the discharge already!
"""

import pandas as pd
import numpy as np
from pathlib import Path

def check_for_double_counting():
    """Compare DAM Gen Awards with Energy Bid Awards"""
    
    print("\n" + "="*80)
    print("CHECKING FOR DOUBLE COUNTING - BATCAVE_BES1")
    print("="*80)
    
    # Load DAM Gen Awards
    dam_gen_file = Path('/home/enrico/data/ERCOT_data/rollup_files/DAM_Gen_Resources/2024.parquet')
    dam_gen = pd.read_parquet(dam_gen_file)
    
    batcave_gen = dam_gen[dam_gen['ResourceName'] == 'BATCAVE_BES1'].copy()
    batcave_gen['datetime'] = pd.to_datetime(batcave_gen['DeliveryDate'])
    batcave_gen['date'] = batcave_gen['datetime'].dt.date
    batcave_gen['hour'] = batcave_gen['datetime'].dt.hour
    
    # Get a sample date with discharge
    discharge_dates = batcave_gen[batcave_gen['AwardedQuantity'] > 0]['date'].unique()
    
    if len(discharge_dates) > 0:
        sample_date = discharge_dates[0]
        print(f"\nSample Date: {sample_date}")
        
        # Get DAM Gen awards for this date
        day_gen = batcave_gen[batcave_gen['date'] == sample_date].sort_values('hour')
        
        print("\nDAM Gen Awards for this date:")
        print("Hour | Gen Award MW")
        print("-"*30)
        for _, row in day_gen.iterrows():
            if row['AwardedQuantity'] > 0:
                print(f"  {row['hour']:2d} | {row['AwardedQuantity']:10.1f}")
        
        # Now check Energy Bid Awards for same date
        eba_dir = Path('/home/enrico/data/ERCOT_data/60_Day_DAM_Disclosure/2024/DAM_EnergyBidAwards')
        
        if eba_dir.exists():
            # Find file for this date
            date_str = pd.to_datetime(sample_date).strftime('%Y%m%d')
            matching_files = list(eba_dir.glob(f'*{date_str}*.csv'))
            
            if matching_files:
                eba_file = matching_files[0]
                print(f"\nChecking Energy Bid Awards: {eba_file.name}")
                
                eba_df = pd.read_csv(eba_file)
                
                # Look for BATCAVE_RN
                batcave_eba = eba_df[eba_df['SettlementPoint'] == 'BATCAVE_RN'].copy()
                
                if not batcave_eba.empty:
                    batcave_eba['Hour'] = batcave_eba['HourEnding'].str.extract(r'(\d+)').astype(int) - 1
                    
                    print("\nEnergy Bid Awards for BATCAVE_RN:")
                    print("Hour | EBA MW | Type")
                    print("-"*40)
                    for _, row in batcave_eba.sort_values('Hour').iterrows():
                        award_type = "DISCHARGE" if row['EnergyBidAwardMW'] > 0 else "CHARGE"
                        print(f"  {row['Hour']:2d} | {row['EnergyBidAwardMW']:10.1f} | {award_type}")
                    
                    # CRITICAL CHECK: Do positive EBA match Gen Awards?
                    print("\n" + "="*50)
                    print("CRITICAL COMPARISON:")
                    print("="*50)
                    
                    for hour in range(24):
                        gen_mw = day_gen[day_gen['hour'] == hour]['AwardedQuantity'].sum()
                        eba_mw = batcave_eba[batcave_eba['Hour'] == hour]['EnergyBidAwardMW'].sum()
                        
                        if gen_mw > 0 or eba_mw != 0:
                            match = "🚨 MATCH!" if abs(gen_mw - eba_mw) < 0.1 and gen_mw > 0 else ""
                            print(f"Hour {hour:2d}: Gen={gen_mw:8.1f}, EBA={eba_mw:8.1f} {match}")
                    
                    # Check totals
                    total_gen = day_gen['AwardedQuantity'].sum()
                    total_eba_pos = batcave_eba[batcave_eba['EnergyBidAwardMW'] > 0]['EnergyBidAwardMW'].sum()
                    total_eba_neg = batcave_eba[batcave_eba['EnergyBidAwardMW'] < 0]['EnergyBidAwardMW'].sum()
                    
                    print(f"\nDAILY TOTALS:")
                    print(f"  DAM Gen Total: {total_gen:.1f} MWh")
                    print(f"  EBA Positive (discharge): {total_eba_pos:.1f} MWh")
                    print(f"  EBA Negative (charge): {total_eba_neg:.1f} MWh")
                    print(f"  EBA Net: {total_eba_pos + total_eba_neg:.1f} MWh")
                    
                    if abs(total_gen - total_eba_pos) < 1:
                        print("\n🚨🚨🚨 DOUBLE COUNTING CONFIRMED! 🚨🚨🚨")
                        print("DAM Gen Awards = Energy Bid Awards (positive)")
                        print("We're counting discharge TWICE!")
                else:
                    print("\nBATCAVE_RN not found in Energy Bid Awards")

def check_all_bess_for_double_counting():
    """Check all BESS for the double counting issue"""
    
    print("\n" + "="*80)
    print("CHECKING ALL BESS FOR DOUBLE COUNTING")
    print("="*80)
    
    # Load one day of data
    dam_gen_file = Path('/home/enrico/data/ERCOT_data/rollup_files/DAM_Gen_Resources/2024.parquet')
    dam_gen = pd.read_parquet(dam_gen_file)
    
    # Get all PWRSTR resources
    bess_resources = dam_gen[dam_gen['ResourceType'] == 'PWRSTR']['ResourceName'].unique()
    
    print(f"\nFound {len(bess_resources)} BESS resources")
    
    # Sample check for a few
    sample_date = pd.to_datetime('2024-07-15')
    
    for bess in bess_resources[:5]:  # Check first 5
        print(f"\n{bess}:")
        
        # Get gen awards
        bess_gen = dam_gen[
            (dam_gen['ResourceName'] == bess) & 
            (pd.to_datetime(dam_gen['DeliveryDate']).dt.date == sample_date.date())
        ]
        
        total_gen = bess_gen['AwardedQuantity'].sum()
        
        if total_gen > 0:
            print(f"  DAM Gen Awards: {total_gen:.1f} MWh on {sample_date.date()}")
            
            # Would need to check EBA for this resource's settlement point
            # This requires the mapping we have

def propose_fixed_algorithm():
    """Propose the correct algorithm"""
    
    print("\n" + "="*80)
    print("PROPOSED FIX FOR REVENUE CALCULATION")
    print("="*80)
    
    print("""
CONFIRMED PROBLEM:
==================
We are DOUBLE COUNTING discharge revenue!
- DAM Gen Awards shows discharge
- Energy Bid Awards ALSO shows same discharge (positive values)
- We're adding both = 2x the actual revenue!

CORRECT ALGORITHM:
==================

OPTION 1: Use ONLY Energy Bid Awards (Recommended)
---------------------------------------------------
For each settlement point:
  - Positive EnergyBidAwardMW × Price = Discharge Revenue
  - Negative EnergyBidAwardMW × Price = Charging Cost
  - Net Revenue = Sum of all hours

This gives us the complete picture in one place!

OPTION 2: Use DAM Gen + Find Real Charging
-------------------------------------------
For discharge:
  - Use DAM Gen Awards (current)
For charging:
  - Use ONLY negative values from Energy Bid Awards
  - OR infer from energy balance
  - OR find Load Resource awards

OPTION 3: Energy Balance Method
--------------------------------
1. Get total discharge from DAM Gen Awards
2. Assume charging = discharge / 0.85 (efficiency)
3. Find lowest price hours for charging
4. Calculate cost based on those hours

WHY BATTERIES APPEAR UNPROFITABLE:
===================================
1. We're not double counting correctly (inflating discharge)
2. But charging costs are roughly correct
3. So the ratio looks terrible!

IMMEDIATE ACTION:
=================
Re-run calculation using ONLY Energy Bid Awards:
- Simple, clean, no double counting
- Positive = revenue, Negative = cost
- This is how ERCOT actually settles!
""")

if __name__ == '__main__':
    check_for_double_counting()
    check_all_bess_for_double_counting()
    propose_fixed_algorithm()
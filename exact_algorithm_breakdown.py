#!/usr/bin/env python3
"""
EXACT ALGORITHM BREAKDOWN - Finding where we're messing up
The issue is likely with offer curves and how we're calculating revenues
"""

import pandas as pd
import numpy as np
from pathlib import Path

def show_exact_algorithm():
    """Show EXACTLY what our algorithm is doing"""
    
    print("\n" + "="*80)
    print("EXACT REVENUE CALCULATION ALGORITHM - WHAT WE'RE DOING")
    print("="*80)
    
    print("""
CURRENT ALGORITHM (from corrected_bess_calculator.py):

1. DAM DISCHARGE REVENUE:
   - Source: DAM_Gen_Resources/{year}.parquet
   - Filter: ResourceType == 'PWRSTR' AND ResourceName matches BESS
   - Calculation: 
     For each hour:
       Revenue = AwardedQuantity (MW) × DAM_Price ($/MWh)
     Total = Sum all hours
     
2. DAM CHARGING COST:
   - Source: 60d_DAM_EnergyBidAwards-*.csv
   - Filter: SettlementPoint matches BESS settlement point
   - Calculation:
     For each hour where EnergyBidAwardMW < 0:
       Cost = abs(EnergyBidAwardMW) × DAM_Price ($/MWh)
     Total = Sum all hours
     
3. PRICES:
   - Source: DA_prices_{year}.parquet
   - If settlement_point not in file: Use HB_BUSAVG
   - Match by datetime/hour

THE PROBLEM:
============
""")
    
    print("\n🚨 CRITICAL ISSUES WITH THIS ALGORITHM:")
    print("-"*50)
    
    print("""
1. ENERGY BID AWARDS MISUNDERSTANDING:
   - EnergyBidAwardMW is NOT just charging!
   - It's the ENTIRE bid curve award at that settlement point
   - Could include BOTH Gen and Load awards
   - Positive values = selling energy (gen)
   - Negative values = buying energy (load)
   
2. WE'RE MISSING THE OFFER CURVES:
   - DAM has OFFER curves (gen willing to sell)
   - DAM has BID curves (load willing to buy)
   - BESS submits BOTH:
     * Offer curve: "I'll discharge X MW at $Y"
     * Bid curve: "I'll charge X MW at $Y"
   
3. SETTLEMENT POINT CONFUSION:
   - Gen Resource has offers at settlement point
   - Load Resource has bids at SAME settlement point
   - Energy Bid Awards shows NET position at settlement point
   
4. DOUBLE COUNTING RISK:
   - DAM Gen Awards shows gen dispatch
   - Energy Bid Awards might ALSO show gen dispatch
   - We could be counting discharge TWICE!
""")

def show_correct_algorithm():
    """Show what the algorithm SHOULD be"""
    
    print("\n" + "="*80)
    print("CORRECT ALGORITHM - WHAT WE SHOULD BE DOING")
    print("="*80)
    
    print("""
CORRECT DAM REVENUE CALCULATION:

METHOD 1: Using Resource-Specific Awards
=========================================
1. DISCHARGE (Gen Resource):
   - Source: DAM_Gen_Resources
   - Filter: ResourceName == '{BESS_NAME}'
   - Revenue = Σ(AwardedQuantity × SettlementPointPrice)

2. CHARGING (Load Resource):
   - Source: DAM_Load_Resources (if it has energy awards)
   - OR: Look for NEGATIVE awards in consolidated data
   - Cost = Σ(ChargingMW × SettlementPointPrice)

METHOD 2: Using Energy Bid Awards (Net Position)
================================================
1. For each settlement point:
   - Positive EnergyBidAwardMW = Net selling (discharge > charge)
   - Negative EnergyBidAwardMW = Net buying (charge > discharge)
   - But this is NET, not gross!

THE KEY INSIGHT:
================
Energy Bid Awards shows NET position, not individual charge/discharge!
If BESS offers 100MW discharge and bids 80MW charge in same hour:
  - Energy Bid Award = +20MW (net discharge)
  - But actual flows: 100MW out, 80MW in (impossible physically!)

WHAT'S REALLY HAPPENING:
========================
BESS can't charge and discharge simultaneously!
The awards should show:
  - Hours with positive awards = discharge only
  - Hours with negative awards = charge only
  - Never both in same hour
""")

def analyze_actual_data():
    """Look at actual data to understand the pattern"""
    
    print("\n" + "="*80)
    print("ANALYZING ACTUAL DATA PATTERNS")
    print("="*80)
    
    # Check BATCAVE in DAM Gen
    dam_gen_file = Path('/home/enrico/data/ERCOT_data/rollup_files/DAM_Gen_Resources/2024.parquet')
    dam_gen = pd.read_parquet(dam_gen_file)
    
    batcave_gen = dam_gen[dam_gen['ResourceName'] == 'BATCAVE_BES1']
    batcave_gen['datetime'] = pd.to_datetime(batcave_gen['DeliveryDate'])
    
    print("\n1. BATCAVE DAM Gen Awards Pattern:")
    print("-"*50)
    
    # Check distribution of awards
    print(f"Total hours in year: 8784")
    print(f"Hours with record in DAM Gen: {len(batcave_gen)}")
    print(f"Hours with AwardedQuantity > 0: {(batcave_gen['AwardedQuantity'] > 0).sum()}")
    print(f"Hours with AwardedQuantity = 0: {(batcave_gen['AwardedQuantity'] == 0).sum()}")
    
    # Check a sample day
    sample_date = '2024-07-15'
    sample_day = batcave_gen[batcave_gen['datetime'].dt.date.astype(str) == sample_date]
    
    if not sample_day.empty:
        print(f"\nSample Day ({sample_date}):")
        print("Hour | Award MW | AS Awards")
        print("-"*40)
        for _, row in sample_day.head(24).iterrows():
            hour = row['datetime'].hour
            award = row['AwardedQuantity']
            reg_up = row.get('RegUpAwarded', 0)
            print(f"  {hour:2d} | {award:8.1f} | RegUp: {reg_up:.1f}")
    
    print("\n2. UNDERSTANDING THE PATTERN:")
    print("-"*50)
    print("""
What we're seeing:
- BESS has record for every hour (even with 0 MW)
- Only ~800 hours with actual discharge awards
- Most hours are 0 MW (providing AS only)

This means:
- BESS is mostly providing Ancillary Services
- Energy arbitrage is limited to peak spread hours
- We need to find WHERE the charging awards are!
""")

def find_charging_data():
    """Try to find where charging data really is"""
    
    print("\n" + "="*80)
    print("HUNTING FOR CHARGING DATA")
    print("="*80)
    
    print("""
Possible locations for charging data:

1. DAM_Load_Resources (but has no energy awards)
2. Energy Bid Awards (but might be net position)
3. Energy Only Offers (might show bid curves)
4. Three Part Offers (might have startup/min energy)

Let's check what we actually have...
""")
    
    # Check what files exist
    base_dir = Path('/home/enrico/data/ERCOT_data/60_Day_DAM_Disclosure/2024')
    
    subdirs = ['DAM_Load_Resources', 'DAM_EnergyBidAwards', 'DAM_EnergyOnlyOffers', 
               'DAM_ThreePartOffers', 'DAM_SelfSchedules']
    
    for subdir in subdirs:
        dir_path = base_dir / subdir
        if dir_path.exists():
            files = list(dir_path.glob('*.csv'))
            if files:
                print(f"\n{subdir}: {len(files)} files")
                
                # Sample first file
                sample = pd.read_csv(files[0], nrows=5)
                print(f"  Columns: {', '.join(sample.columns[:5])}...")
                
                # Check for BATCAVE
                if 'SettlementPoint' in sample.columns or 'ResourceName' in sample.columns:
                    full_df = pd.read_csv(files[0])
                    
                    # Look for BATCAVE
                    if 'SettlementPoint' in full_df.columns:
                        batcave_data = full_df[full_df['SettlementPoint'] == 'BATCAVE_RN']
                        if not batcave_data.empty:
                            print(f"  ✅ Found BATCAVE_RN: {len(batcave_data)} records")
                            
                            # Check for charge indicators
                            if 'EnergyBidAwardMW' in batcave_data.columns:
                                neg = (batcave_data['EnergyBidAwardMW'] < 0).sum()
                                pos = (batcave_data['EnergyBidAwardMW'] > 0).sum()
                                print(f"     Negative (charge): {neg}, Positive (discharge): {pos}")

def propose_solution():
    """Propose the correct solution"""
    
    print("\n" + "="*80)
    print("PROPOSED SOLUTION")
    print("="*80)
    
    print("""
THE REAL PROBLEM:
=================
We're treating Energy Bid Awards as pure charging data, but it's actually
the NET cleared position at the settlement point!

CORRECT APPROACH:
=================

1. For DISCHARGE Revenue:
   - Use DAM_Gen_Resources -> AwardedQuantity
   - This is correct and working
   
2. For CHARGING Cost:
   OPTION A: Infer from Energy Bid Awards
   - When EnergyBidAwardMW < 0: Battery is net buyer (charging)
   - When EnergyBidAwardMW > 0: Battery is net seller (discharging)
   - BUT: Don't double count with DAM Gen Awards!
   
   OPTION B: Infer from dispatch pattern
   - Use state-of-charge logic
   - If battery discharged X MWh, it must have charged ~X/0.85 MWh
   - Find hours with lowest prices for charging
   
   OPTION C: Look for Load Resource awards
   - Check if there's a paired Load Resource
   - Use its awards for charging

3. AVOIDING DOUBLE COUNTING:
   - If using Energy Bid Awards, DON'T also use DAM Gen Awards
   - They might represent the same dispatch!

4. PRICE APPLICATION:
   - Use settlement point specific prices when available
   - Otherwise use hub average (HB_BUSAVG)
   - Make sure to match hour-by-hour

IMMEDIATE FIX:
==============
1. Check if Energy Bid Awards positive values overlap with DAM Gen Awards
2. If yes, we're double counting - use only one source
3. Verify charging MWh is reasonable (should be ~1.2x discharge MWh)
4. Recalculate with proper algorithm
""")

if __name__ == '__main__':
    show_exact_algorithm()
    show_correct_algorithm()
    analyze_actual_data()
    find_charging_data()
    propose_solution()
#!/usr/bin/env python3
"""
Debug BESS Revenue Calculations with Visualizations
Something is wrong - batteries should be profitable!
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Set up plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_and_analyze():
    """Load data and create debug visualizations"""
    
    # Load corrected revenues
    df = pd.read_csv('/home/enrico/data/ERCOT_data/bess_analysis/corrected_bess_revenues.csv')
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Revenue Breakdown by Component
    ax1 = plt.subplot(2, 3, 1)
    df_plot = df.set_index('resource_name')
    df_plot[['dam_discharge_revenue', 'dam_charge_cost', 'as_revenue']].plot(
        kind='bar', ax=ax1, stacked=False
    )
    ax1.set_title('Revenue vs Cost Breakdown\n(Something is WRONG here!)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Revenue ($)')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax1.legend(['DAM Discharge', 'DAM Charge Cost', 'AS Revenue'])
    plt.xticks(rotation=45, ha='right')
    
    # 2. Net Revenue Waterfall
    ax2 = plt.subplot(2, 3, 2)
    df_sorted = df.sort_values('total_net_revenue', ascending=False)
    colors = ['green' if x > 0 else 'red' for x in df_sorted['total_net_revenue']]
    bars = ax2.bar(range(len(df_sorted)), df_sorted['total_net_revenue'], color=colors)
    ax2.set_title('Net Revenue by BESS\n(Why so many losses?)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Net Revenue ($)')
    ax2.set_xticks(range(len(df_sorted)))
    ax2.set_xticklabels(df_sorted['resource_name'], rotation=45, ha='right')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # 3. Charge vs Discharge Revenue Ratio
    ax3 = plt.subplot(2, 3, 3)
    df['charge_discharge_ratio'] = df['dam_charge_cost'] / (df['dam_discharge_revenue'] + 0.1)  # avoid div by 0
    ax3.bar(df['resource_name'], df['charge_discharge_ratio'])
    ax3.set_title('Charge Cost / Discharge Revenue\n(Should be ~0.7-0.9 for efficiency)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Ratio')
    ax3.axhline(y=1.0, color='red', linestyle='--', label='Break-even')
    ax3.axhline(y=0.85, color='green', linestyle='--', label='85% RTE')
    ax3.legend()
    plt.xticks(rotation=45, ha='right')
    
    # 4. AS Revenue Dependency
    ax4 = plt.subplot(2, 3, 4)
    df['as_dependency'] = df['as_revenue'] / (df['total_net_revenue'].abs() + 1)  # avoid div by 0
    df['as_dependency'] = df['as_dependency'].clip(0, 10)  # clip extreme values
    ax4.bar(df['resource_name'], df['as_dependency'] * 100)
    ax4.set_title('AS Revenue Dependency %\n(Why so high?)', fontsize=14, fontweight='bold')
    ax4.set_ylabel('AS % of Total Revenue')
    ax4.axhline(y=100, color='red', linestyle='--', alpha=0.5)
    plt.xticks(rotation=45, ha='right')
    
    # 5. Energy Balance Check
    ax5 = plt.subplot(2, 3, 5)
    # Load actual MW data to check energy balance
    dam_gen_file = Path('/home/enrico/data/ERCOT_data/rollup_files/DAM_Gen_Resources/2024.parquet')
    if dam_gen_file.exists():
        dam_gen = pd.read_parquet(dam_gen_file)
        # Get BATCAVE data for deep dive
        batcave = dam_gen[dam_gen['ResourceName'].str.contains('BATCAVE', na=False)]
        if not batcave.empty:
            batcave['date'] = pd.to_datetime(batcave['DeliveryDate']).dt.date
            daily_awards = batcave.groupby('date')['AwardedQuantity'].sum()
            ax5.plot(daily_awards.index, daily_awards.values, label='Daily Awards (MW)', alpha=0.7)
            ax5.set_title('BATCAVE Daily Awards Pattern\n(Check for issues)', fontsize=14, fontweight='bold')
            ax5.set_ylabel('MW')
            ax5.legend()
            plt.xticks(rotation=45, ha='right')
    
    # 6. Price Analysis
    ax6 = plt.subplot(2, 3, 6)
    # Try to load price data and check if we're using right prices
    price_file = Path('/home/enrico/data/ERCOT_data/rollup_files/flattened/DA_prices_2024.parquet')
    if price_file.exists():
        prices = pd.read_parquet(price_file)
        if 'HB_BUSAVG' in prices.columns:
            # Check price distribution
            price_data = prices['HB_BUSAVG'].dropna()
            ax6.hist(price_data, bins=50, edgecolor='black', alpha=0.7)
            ax6.axvline(price_data.mean(), color='red', linestyle='--', label=f'Mean: ${price_data.mean():.2f}')
            ax6.set_title(f'DAM Price Distribution (HB_BUSAVG)\nMean: ${price_data.mean():.2f}', fontsize=14, fontweight='bold')
            ax6.set_xlabel('Price ($/MWh)')
            ax6.set_ylabel('Frequency')
            ax6.legend()
    
    plt.suptitle('BESS Revenue Debug Analysis - Finding the Problem', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('/home/enrico/data/ERCOT_data/bess_analysis/debug_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print detailed analysis
    print("\n" + "="*80)
    print("DEBUGGING BESS REVENUE CALCULATIONS")
    print("="*80)
    
    print("\n1. REVENUE COMPONENT ANALYSIS:")
    print("-"*50)
    print(f"Total DAM Discharge Revenue: ${df['dam_discharge_revenue'].sum():,.0f}")
    print(f"Total DAM Charge Cost: ${df['dam_charge_cost'].sum():,.0f}")
    print(f"Total AS Revenue: ${df['as_revenue'].sum():,.0f}")
    print(f"Total Net Revenue: ${df['total_net_revenue'].sum():,.0f}")
    
    print("\n2. SUSPICIOUS PATTERNS:")
    print("-"*50)
    
    # Check for units with zero discharge but high charge
    zero_discharge = df[df['dam_discharge_revenue'] == 0]
    if not zero_discharge.empty:
        print(f"\n⚠️ Units with ZERO discharge revenue but charging costs:")
        for _, row in zero_discharge.iterrows():
            if row['dam_charge_cost'] > 0:
                print(f"  - {row['resource_name']}: Charged ${row['dam_charge_cost']:,.0f} but discharged $0")
    
    # Check charge/discharge ratio
    df['efficiency'] = df['dam_discharge_revenue'] / (df['dam_charge_cost'] + 1)
    bad_efficiency = df[df['efficiency'] < 1.0]
    print(f"\n⚠️ Units losing money on arbitrage (efficiency < 1.0):")
    for _, row in bad_efficiency.iterrows():
        print(f"  - {row['resource_name']}: Efficiency = {row['efficiency']:.2f}")
    
    # Check if we're double counting charging
    print("\n3. SANITY CHECKS:")
    print("-"*50)
    
    # For BATCAVE (should be profitable)
    batcave_row = df[df['resource_name'] == 'BATCAVE_BES1'].iloc[0]
    print(f"\nBATCAVE_BES1 Analysis:")
    print(f"  Discharge Revenue: ${batcave_row['dam_discharge_revenue']:,.0f}")
    print(f"  Charge Cost: ${batcave_row['dam_charge_cost']:,.0f}")
    print(f"  Implied Efficiency: {(batcave_row['dam_discharge_revenue']/batcave_row['dam_charge_cost']):.2f}")
    print(f"  AS Revenue: ${batcave_row['as_revenue']:,.0f}")
    print(f"  Net Revenue: ${batcave_row['total_net_revenue']:,.0f}")
    
    # Check if charging costs are too high
    print("\n4. CHARGING COST ANALYSIS:")
    print("-"*50)
    avg_charge_cost = df['dam_charge_cost'].mean()
    avg_discharge_rev = df['dam_discharge_revenue'].mean()
    print(f"Average Charge Cost: ${avg_charge_cost:,.0f}")
    print(f"Average Discharge Revenue: ${avg_discharge_rev:,.0f}")
    print(f"Ratio: {(avg_charge_cost/avg_discharge_rev):.2f} (should be < 1.0)")
    
    return df

def check_energy_bid_awards():
    """Check if we're reading Energy Bid Awards correctly"""
    print("\n" + "="*80)
    print("CHECKING ENERGY BID AWARDS DATA")
    print("="*80)
    
    # Check if Energy Bid Awards files exist
    eba_pattern = Path('/home/enrico/data/ERCOT_data/60_Day_DAM_Disclosure/2024/DAM_EnergyBidAwards')
    if eba_pattern.exists():
        csv_files = list(eba_pattern.glob('*.csv'))
        print(f"\nFound {len(csv_files)} Energy Bid Award CSV files")
        
        if csv_files:
            # Sample one file
            sample_file = csv_files[0]
            print(f"\nSampling: {sample_file.name}")
            sample_df = pd.read_csv(sample_file, nrows=1000)
            
            # Check for BESS resources
            if 'SettlementPoint' in sample_df.columns:
                # Look for negative values (charging)
                negative_awards = sample_df[sample_df['EnergyBidAwardMW'] < 0]
                print(f"Negative awards (charging): {len(negative_awards)} records")
                
                # Check settlement points
                bess_sps = ['BATCAVE_RN', 'ALVIN_RN', 'ANCHOR_ALL', 'AZURE_RN']
                for sp in bess_sps:
                    sp_data = sample_df[sample_df['SettlementPoint'] == sp]
                    if not sp_data.empty:
                        print(f"\n{sp}:")
                        print(f"  Total records: {len(sp_data)}")
                        print(f"  Negative (charging): {len(sp_data[sp_data['EnergyBidAwardMW'] < 0])}")
                        print(f"  Positive (discharge): {len(sp_data[sp_data['EnergyBidAwardMW'] > 0])}")
                        print(f"  Sum MW: {sp_data['EnergyBidAwardMW'].sum():.2f}")

def check_dam_gen_awards():
    """Check DAM Gen Awards for discharge data"""
    print("\n" + "="*80)
    print("CHECKING DAM GEN AWARDS DATA")
    print("="*80)
    
    dam_gen_file = Path('/home/enrico/data/ERCOT_data/rollup_files/DAM_Gen_Resources/2024.parquet')
    if dam_gen_file.exists():
        dam_gen = pd.read_parquet(dam_gen_file)
        
        # Filter for PWRSTR (BESS)
        bess_data = dam_gen[dam_gen['ResourceType'] == 'PWRSTR']
        print(f"\nTotal PWRSTR records: {len(bess_data)}")
        
        # Check specific BESS
        for bess_name in ['BATCAVE_BES1', 'ALVIN_UNIT1', 'ANCHOR_BESS1']:
            unit_data = bess_data[bess_data['ResourceName'] == bess_name]
            if not unit_data.empty:
                print(f"\n{bess_name}:")
                print(f"  Total hours: {len(unit_data)}")
                print(f"  Total MW awarded: {unit_data['AwardedQuantity'].sum():,.0f}")
                print(f"  Avg MW/hour: {unit_data['AwardedQuantity'].mean():.2f}")
                print(f"  Max MW: {unit_data['AwardedQuantity'].max():.2f}")
                
                # Check if awards match capacity
                if 'BATCAVE' in bess_name:
                    # BATCAVE is 100MW
                    over_capacity = unit_data[unit_data['AwardedQuantity'] > 100]
                    if not over_capacity.empty:
                        print(f"  ⚠️ Awards over 100MW capacity: {len(over_capacity)} hours")

if __name__ == '__main__':
    # Create visualizations
    df = load_and_analyze()
    
    # Deep dive into data sources
    check_energy_bid_awards()
    check_dam_gen_awards()
    
    print("\n" + "="*80)
    print("VISUALIZATION SAVED TO: /home/enrico/data/ERCOT_data/bess_analysis/debug_visualization.png")
    print("="*80)
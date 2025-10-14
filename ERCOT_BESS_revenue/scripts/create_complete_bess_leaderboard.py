#!/usr/bin/env python3
"""
Create Complete BESS Leaderboard for ALL ERCOT Battery Projects
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def create_complete_leaderboard():
    """Create comprehensive leaderboard for all BESS in ERCOT"""
    
    data_dir = Path('/home/enrico/data/ERCOT_data')
    rollup_dir = data_dir / 'rollup_files'
    output_dir = data_dir / 'bess_analysis'
    output_dir.mkdir(exist_ok=True)
    
    # Load settlement point mapping
    mapping_file = data_dir / 'Settlement_Points_List_and_Electrical_Buses_Mapping' / 'latest_mapping' / 'SP_List_EB_Mapping' / 'gen_node_map.csv'
    sp_map = pd.read_csv(mapping_file)
    unit_to_sp = dict(zip(sp_map['UNIT_NAME'], sp_map['RESOURCE_NODE']))
    
    logger.info("=" * 80)
    logger.info("CREATING COMPLETE BESS LEADERBOARD FOR ALL ERCOT")
    logger.info("=" * 80)
    
    # Process multiple years
    all_results = []
    years = [2023, 2024]  # Can expand to more years
    
    for year in years:
        logger.info(f"\nProcessing {year}...")
        
        # Load data
        dam_gen_file = rollup_dir / 'DAM_Gen_Resources' / f'{year}.parquet'
        da_price_file = rollup_dir / 'flattened' / f'DA_prices_{year}.parquet'
        as_price_file = rollup_dir / 'flattened' / f'AS_prices_{year}.parquet'
        
        if not all([dam_gen_file.exists(), da_price_file.exists()]):
            logger.warning(f"Missing data for {year}")
            continue
        
        # Load files
        dam_gen = pd.read_parquet(dam_gen_file)
        da_prices = pd.read_parquet(da_price_file)
        
        # Get BESS resources
        bess_data = dam_gen[dam_gen['ResourceType'] == 'PWRSTR'].copy()
        unique_bess = bess_data['ResourceName'].unique()
        
        logger.info(f"Found {len(unique_bess)} BESS resources in {year}")
        
        # Ensure datetime columns
        if 'datetime_ts' in da_prices.columns:
            da_prices['datetime'] = pd.to_datetime(da_prices['datetime_ts'])
        if 'DeliveryDate' in bess_data.columns:
            bess_data['datetime'] = pd.to_datetime(bess_data['DeliveryDate'])
        
        # Load AS prices if available
        as_prices = pd.read_parquet(as_price_file) if as_price_file.exists() else pd.DataFrame()
        if not as_prices.empty and 'datetime_ts' in as_prices.columns:
            as_prices['datetime'] = pd.to_datetime(as_prices['datetime_ts'])
        
        # Calculate revenue for each BESS
        for bess_name in unique_bess:
            # Get data for this BESS
            bess_awards = bess_data[bess_data['ResourceName'] == bess_name].copy()
            
            if bess_awards.empty:
                continue
            
            # Get settlement point
            settlement_point = bess_awards['SettlementPointName'].iloc[0]
            
            # Map to correct settlement point if possible
            mapped_sp = unit_to_sp.get(bess_name)
            if mapped_sp:
                settlement_point = mapped_sp
            
            # Calculate DA revenue
            da_revenue = 0
            if settlement_point in da_prices.columns:
                merged = bess_awards.merge(
                    da_prices[['datetime', settlement_point]], 
                    on='datetime', 
                    how='left'
                )
                da_revenue = (merged['AwardedQuantity'] * merged[settlement_point]).sum()
            elif 'HB_BUSAVG' in da_prices.columns:
                merged = bess_awards.merge(
                    da_prices[['datetime', 'HB_BUSAVG']], 
                    on='datetime', 
                    how='left'
                )
                da_revenue = (merged['AwardedQuantity'] * merged['HB_BUSAVG']).sum()
            
            # Calculate AS revenue
            as_revenue = 0
            if not as_prices.empty:
                as_cols = {
                    'RegUpAwarded': 'REGUP',
                    'RegDownAwarded': 'REGDN',
                    'RRSAwarded': 'RRS',
                    'NonSpinAwarded': 'NSPIN',
                    'ECRSAwarded': 'ECRS'
                }
                
                for award_col, price_col in as_cols.items():
                    if award_col in bess_awards.columns and price_col in as_prices.columns:
                        merged = bess_awards[['datetime', award_col]].merge(
                            as_prices[['datetime', price_col]], 
                            on='datetime', 
                            how='left'
                        )
                        as_revenue += (merged[award_col] * merged[price_col]).sum()
            
            # Calculate metrics
            total_revenue = da_revenue + as_revenue
            total_mwh = abs(bess_awards['AwardedQuantity']).sum()
            operating_hours = len(bess_awards)
            
            # Get capacity estimate (max award)
            capacity_mw = abs(bess_awards['AwardedQuantity']).max() if not bess_awards.empty else 100
            
            all_results.append({
                'year': year,
                'resource_name': bess_name,
                'settlement_point': settlement_point,
                'da_revenue': da_revenue,
                'as_revenue': as_revenue,
                'total_revenue': total_revenue,
                'total_mwh': total_mwh,
                'operating_hours': operating_hours,
                'capacity_mw': capacity_mw,
                'revenue_per_mw': total_revenue / capacity_mw if capacity_mw > 0 else 0,
                'revenue_per_mwh': total_revenue / total_mwh if total_mwh > 0 else 0
            })
    
    # Create leaderboard
    leaderboard = pd.DataFrame(all_results)
    
    if not leaderboard.empty:
        # Add rankings
        leaderboard = leaderboard.sort_values('total_revenue', ascending=False)
        leaderboard['rank'] = range(1, len(leaderboard) + 1)
        
        # Create annual summary
        annual_summary = leaderboard.groupby(['year', 'resource_name']).agg({
            'total_revenue': 'sum',
            'da_revenue': 'sum',
            'as_revenue': 'sum',
            'total_mwh': 'sum',
            'capacity_mw': 'first',
            'settlement_point': 'first'
        }).reset_index()
        
        annual_summary = annual_summary.sort_values(['year', 'total_revenue'], ascending=[True, False])
        
        # Save outputs
        leaderboard_file = output_dir / 'complete_bess_leaderboard.parquet'
        leaderboard.to_parquet(leaderboard_file, index=False)
        logger.info(f"Saved complete leaderboard to {leaderboard_file}")
        
        # Save CSV for easy viewing
        csv_file = output_dir / 'complete_bess_leaderboard.csv'
        leaderboard.to_csv(csv_file, index=False)
        logger.info(f"Saved CSV to {csv_file}")
        
        # Print top performers
        print("\n" + "=" * 80)
        print("TOP 20 BESS PROJECTS BY TOTAL REVENUE (ALL YEARS)")
        print("=" * 80)
        
        top_overall = leaderboard.groupby('resource_name').agg({
            'total_revenue': 'sum',
            'da_revenue': 'sum',
            'as_revenue': 'sum',
            'settlement_point': 'first',
            'capacity_mw': 'first'
        }).sort_values('total_revenue', ascending=False).head(20)
        
        for idx, (name, row) in enumerate(top_overall.iterrows(), 1):
            print(f"{idx:2d}. {name:25s} ${row['total_revenue']:12,.0f} "
                  f"(DA: ${row['da_revenue']:10,.0f}, AS: ${row['as_revenue']:10,.0f}) "
                  f"[{row['settlement_point']}]")
        
        # Summary statistics
        print("\n" + "=" * 80)
        print("OVERALL STATISTICS")
        print("=" * 80)
        
        total_projects = leaderboard['resource_name'].nunique()
        total_revenue = leaderboard['total_revenue'].sum()
        total_da = leaderboard['da_revenue'].sum()
        total_as = leaderboard['as_revenue'].sum()
        
        print(f"Total BESS Projects: {total_projects}")
        print(f"Total Revenue: ${total_revenue:,.0f}")
        print(f"  - DA Energy: ${total_da:,.0f} ({total_da/total_revenue*100:.1f}%)")
        print(f"  - AS Services: ${total_as:,.0f} ({total_as/total_revenue*100:.1f}%)")
        print(f"Average per Project: ${total_revenue/total_projects:,.0f}")
        
        # Revenue distribution
        print("\n" + "=" * 80)
        print("REVENUE DISTRIBUTION")
        print("=" * 80)
        
        top10_revenue = leaderboard.nlargest(10, 'total_revenue')['total_revenue'].sum()
        print(f"Top 10 projects: ${top10_revenue:,.0f} ({top10_revenue/total_revenue*100:.1f}% of total)")
        
        # Projects by revenue tier
        tiers = [
            (10_000_000, float('inf'), '>$10M'),
            (5_000_000, 10_000_000, '$5M-$10M'),
            (1_000_000, 5_000_000, '$1M-$5M'),
            (500_000, 1_000_000, '$500K-$1M'),
            (100_000, 500_000, '$100K-$500K'),
            (0, 100_000, '<$100K')
        ]
        
        print("\nProjects by Revenue Tier:")
        for low, high, label in tiers:
            count = len(leaderboard[(leaderboard['total_revenue'] >= low) & 
                                   (leaderboard['total_revenue'] < high)])
            if count > 0:
                print(f"  {label:12s}: {count:3d} projects")
        
        return leaderboard
    else:
        logger.error("No data to create leaderboard")
        return pd.DataFrame()

if __name__ == '__main__':
    leaderboard = create_complete_leaderboard()
    
    if not leaderboard.empty:
        print("\n✅ Complete BESS leaderboard created successfully!")
        print(f"   Location: /home/enrico/data/ERCOT_data/bess_analysis/complete_bess_leaderboard.csv")
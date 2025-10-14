#!/usr/bin/env python3
"""
TBX Summary Report - Compare arbitrage values across all years
"""

import pandas as pd
from pathlib import Path

def generate_tbx_summary():
    tbx_dir = Path("/home/enrico/data/ERCOT_data/tbx_results")
    
    # Load all annual data
    years = [2021, 2022, 2023, 2024, 2025]
    all_data = []
    
    for year in years:
        file_path = tbx_dir / f"tbx_annual_{year}.parquet"
        if file_path.exists():
            df = pd.read_parquet(file_path)
            df['year'] = year
            all_data.append(df)
    
    # Combine all years
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Adjust 2025 values to annual equivalent (231 days -> 365 days)
    mask_2025 = combined_df['year'] == 2025
    adjustment_factor = 365 / 231
    combined_df.loc[mask_2025, 'tb2_da_revenue_annualized'] = combined_df.loc[mask_2025, 'tb2_da_revenue'] * adjustment_factor
    combined_df.loc[mask_2025, 'tb4_da_revenue_annualized'] = combined_df.loc[mask_2025, 'tb4_da_revenue'] * adjustment_factor
    combined_df.loc[~mask_2025, 'tb2_da_revenue_annualized'] = combined_df.loc[~mask_2025, 'tb2_da_revenue']
    combined_df.loc[~mask_2025, 'tb4_da_revenue_annualized'] = combined_df.loc[~mask_2025, 'tb4_da_revenue']
    
    print("=" * 80)
    print("TBX BATTERY ARBITRAGE ANALYSIS - COMPREHENSIVE SUMMARY")
    print("=" * 80)
    print()
    
    # Top node by year
    print("📊 TOP PERFORMING NODES BY YEAR (TB2 Day-Ahead)")
    print("-" * 60)
    for year in years:
        year_data = combined_df[combined_df['year'] == year]
        top_node = year_data.nlargest(1, 'tb2_da_revenue')
        if not top_node.empty:
            row = top_node.iloc[0]
            days = f"({row['days_count']} days)" if year == 2025 else ""
            print(f"{year}: {row['node']:<15} ${row['tb2_da_revenue']:>10,.2f} {days}")
    
    print()
    print("📈 YEAR-OVER-YEAR COMPARISON (Average across all nodes)")
    print("-" * 60)
    
    yearly_avg = combined_df.groupby('year').agg({
        'tb2_da_revenue': 'mean',
        'tb4_da_revenue': 'mean',
        'tb2_da_revenue_annualized': 'mean',
        'tb4_da_revenue_annualized': 'mean',
        'days_count': 'first'
    })
    
    print(f"{'Year':<8} {'TB2 Avg ($)':<15} {'TB4 Avg ($)':<15} {'TB2 Ann. ($)':<15} {'Days'}")
    for year in years:
        if year in yearly_avg.index:
            row = yearly_avg.loc[year]
            tb2_ann = row['tb2_da_revenue_annualized'] if year == 2025 else row['tb2_da_revenue']
            print(f"{year:<8} ${row['tb2_da_revenue']:>12,.2f} ${row['tb4_da_revenue']:>12,.2f} ${tb2_ann:>12,.2f} {row['days_count']:>4.0f}")
    
    print()
    print("🏆 ALL-TIME TOP 10 NODES (TB2 Day-Ahead, Annualized)")
    print("-" * 60)
    
    # Best performing node-year combinations
    combined_df['node_year'] = combined_df['node'] + ' (' + combined_df['year'].astype(str) + ')'
    top_10 = combined_df.nlargest(10, 'tb2_da_revenue_annualized')
    
    print(f"{'Rank':<5} {'Node (Year)':<25} {'TB2 Revenue ($)':<20} {'TB4 Revenue ($)'}")
    for i, (_, row) in enumerate(top_10.iterrows(), 1):
        print(f"{i:<5} {row['node_year']:<25} ${row['tb2_da_revenue_annualized']:>17,.2f} ${row['tb4_da_revenue_annualized']:>17,.2f}")
    
    print()
    print("📉 2025 YTD PERFORMANCE (Through July 31)")
    print("-" * 60)
    
    data_2025 = combined_df[combined_df['year'] == 2025]
    top_5_2025 = data_2025.nlargest(5, 'tb2_da_revenue')
    
    print(f"{'Node':<15} {'TB2 YTD ($)':<15} {'TB2 Proj. ($)':<15} {'TB4 YTD ($)':<15} {'TB4 Proj. ($)'}")
    for _, row in top_5_2025.iterrows():
        print(f"{row['node']:<15} ${row['tb2_da_revenue']:>12,.2f} ${row['tb2_da_revenue_annualized']:>12,.2f} ${row['tb4_da_revenue']:>12,.2f} ${row['tb4_da_revenue_annualized']:>12,.2f}")
    
    print()
    print("💡 KEY INSIGHTS")
    print("-" * 60)
    
    # Calculate some metrics
    best_year = yearly_avg['tb2_da_revenue'].idxmax()
    best_avg = yearly_avg.loc[best_year, 'tb2_da_revenue']
    
    worst_year = yearly_avg['tb2_da_revenue'].idxmin()
    worst_avg = yearly_avg.loc[worst_year, 'tb2_da_revenue']
    
    # 2025 projected vs 2024
    avg_2024 = yearly_avg.loc[2024, 'tb2_da_revenue']
    avg_2025_proj = yearly_avg.loc[2025, 'tb2_da_revenue_annualized']
    pct_change = ((avg_2025_proj - avg_2024) / avg_2024) * 100
    
    print(f"• Best year for arbitrage: {best_year} (avg ${best_avg:,.2f} per node)")
    print(f"• Worst year for arbitrage: {worst_year} (avg ${worst_avg:,.2f} per node)")
    print(f"• 2025 projection vs 2024: {pct_change:+.1f}% change")
    print(f"• TB4 premium over TB2: ~{(yearly_avg['tb4_da_revenue'].mean() / yearly_avg['tb2_da_revenue'].mean() - 1) * 100:.0f}%")
    print(f"• LZ_WEST dominance: Appears in top position {len([1 for y in years if combined_df[(combined_df['year']==y)].nlargest(1, 'tb2_da_revenue')['node'].values[0] == 'LZ_WEST'])}/{len(years)} years")
    
    print()
    print("=" * 80)

if __name__ == "__main__":
    generate_tbx_summary()
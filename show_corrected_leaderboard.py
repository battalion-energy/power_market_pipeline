#!/usr/bin/env python3
"""
Show the corrected BESS leaderboard with charging costs included
Compare with old method to show the impact
"""

from corrected_bess_calculator import CorrectedBessCalculator
import pandas as pd

def main():
    print('='*100)
    print('CORRECTED BESS LEADERBOARD - WITH CHARGING COSTS INCLUDED')
    print('='*100)
    
    # Run the corrected calculator
    calc = CorrectedBessCalculator()
    df = calc.run_analysis(years=[2024], limit=30)  # Top 30 BESS
    
    # Show comparison
    print('\n' + '='*100)
    print('IMPACT ANALYSIS: Before vs After Including Charging Costs')
    print('='*100)
    
    # Calculate what revenues would be WITHOUT charging costs (old method)
    df['old_total_revenue'] = df['dam_discharge_revenue'] + df['rt_discharge_revenue'] + df['as_revenue']
    df['revenue_difference'] = df['total_net_revenue'] - df['old_total_revenue']
    df['pct_change'] = (df['revenue_difference'] / df['old_total_revenue'] * 100).fillna(0).round(1)
    
    # Show the impact for top performers
    print('\n📊 TOP 15 BESS - Revenue Impact of Including Charging Costs:')
    print('-'*100)
    
    comparison = df.head(15).copy()
    
    print(f"{'Rank':<5} {'Resource':<20} {'Old Method':>15} {'Charging Cost':>15} {'Corrected Net':>15} {'Change':>10}")
    print('-'*100)
    
    for idx, row in comparison.iterrows():
        rank = idx + 1
        print(f"{rank:<5} {row['resource_name']:<20} ${row['old_total_revenue']:>14,.0f} ${row['dam_charge_cost']:>14,.0f} ${row['total_net_revenue']:>14,.0f} {row['pct_change']:>9.1f}%")
    
    # Summary statistics
    print('\n' + '='*100)
    print('CHARGING COST IMPACT SUMMARY')
    print('='*100)
    
    total_old = df['old_total_revenue'].sum()
    total_charging = df['dam_charge_cost'].sum() + df['rt_charge_cost'].sum()
    total_new = df['total_net_revenue'].sum()
    
    print(f'Revenue WITHOUT charging costs:  ${total_old:,.0f}')
    print(f'Total charging costs discovered:  ${total_charging:,.0f}')
    print(f'Revenue WITH charging costs:     ${total_new:,.0f}')
    print(f'Overall revenue reduction:        {((total_new - total_old) / total_old * 100):.1f}%')
    
    print(f'\nCharging cost statistics:')
    print(f'  Average per BESS:     ${df["dam_charge_cost"].mean():,.0f}')
    print(f'  Median per BESS:      ${df["dam_charge_cost"].median():,.0f}')
    print(f'  Maximum:              ${df["dam_charge_cost"].max():,.0f} ({df.loc[df["dam_charge_cost"].idxmax(), "resource_name"]})')
    print(f'  Minimum:              ${df["dam_charge_cost"].min():,.0f}')
    
    # Profitability analysis
    print(f'\n📈 PROFITABILITY ANALYSIS:')
    print(f'  Units with positive total revenue: {(df["total_net_revenue"] > 0).sum()} out of {len(df)}')
    print(f'  Units with positive DAM net:       {(df["dam_net"] > 0).sum()} out of {len(df)}')
    print(f'  Units profitable without AS:       {((df["dam_net"] + df["rt_net"]) > 0).sum()} out of {len(df)}')
    
    # AS dependency
    df['as_dependency'] = (df['as_revenue'] / df['total_net_revenue'] * 100).clip(upper=500)
    high_as_dependency = df[df['as_dependency'] > 80]
    
    print(f'\n⚠️  RISK METRICS:')
    print(f'  Units with >80% AS dependency: {len(high_as_dependency)}')
    if len(high_as_dependency) > 0:
        print(f'  Examples: {high_as_dependency.head(3)["resource_name"].tolist()}')
    
    # Winners and losers
    print(f'\n🏆 BIGGEST CHANGES (After Including Charging):')
    df_sorted = df.sort_values('revenue_difference')
    
    print(f'\nMost Impacted (Negative):')
    for idx, row in df_sorted.head(3).iterrows():
        print(f'  {row["resource_name"]}: ${row["revenue_difference"]:,.0f} ({row["pct_change"]:.1f}% reduction)')
    
    print(f'\nLeast Impacted:')
    for idx, row in df_sorted.tail(3).iterrows():
        if row['revenue_difference'] < 0:
            print(f'  {row["resource_name"]}: ${row["revenue_difference"]:,.0f} ({abs(row["pct_change"]):.1f}% reduction)')
    
    # Save updated leaderboard
    leaderboard_file = '/home/enrico/data/ERCOT_data/bess_analysis/corrected_leaderboard_2024.csv'
    df.to_csv(leaderboard_file, index=False)
    print(f'\n✅ Corrected leaderboard saved to: {leaderboard_file}')
    
    return df

if __name__ == '__main__':
    main()
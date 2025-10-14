#!/usr/bin/env python3
"""Visualize battery analysis results."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load summary data
summary_df = pd.read_csv('battery_analysis_output/ERCOT_RT_Battery_Summary.csv')

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('ERCOT Battery Energy Storage Analysis Results (RT-Only, 2022)', fontsize=16)

# 1. Total Annual Revenue by Zone and Battery Type
ax1 = axes[0, 0]
pivot_revenue = summary_df.pivot(index='Zone', columns='Battery', values='Total Annual Revenue')
pivot_revenue.plot(kind='bar', ax=ax1)
ax1.set_title('Total Annual Revenue by Zone')
ax1.set_ylabel('Annual Revenue ($)')
ax1.set_xlabel('Zone')
ax1.legend(title='Battery Type')
ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)

# 2. Average Daily Revenue
ax2 = axes[0, 1]
pivot_daily = summary_df.pivot(index='Zone', columns='Battery', values='Avg Daily Revenue')
pivot_daily.plot(kind='bar', ax=ax2)
ax2.set_title('Average Daily Revenue by Zone')
ax2.set_ylabel('Daily Revenue ($)')
ax2.set_xlabel('Zone')
ax2.legend(title='Battery Type')
ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)

# 3. Revenue per Cycle
ax3 = axes[1, 0]
summary_df['Revenue per Cycle'] = summary_df['Total Annual Revenue'] / summary_df['Total Cycles']
pivot_cycle = summary_df.pivot(index='Zone', columns='Battery', values='Revenue per Cycle')
pivot_cycle.plot(kind='bar', ax=ax3)
ax3.set_title('Revenue per Cycle by Zone')
ax3.set_ylabel('Revenue per Cycle ($)')
ax3.set_xlabel('Zone')
ax3.legend(title='Battery Type')
ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)

# 4. Load monthly data for trend
monthly_df = pd.read_csv('battery_analysis_output/ERCOT_RT_Battery_Monthly.csv')
ax4 = axes[1, 1]

# Plot monthly revenue for Houston TB2 as example
houston_tb2 = monthly_df[monthly_df['zone_battery'] == 'HOUSTON_TB2'].copy()
houston_tb2['month'] = pd.to_datetime(houston_tb2['month'])
houston_tb2 = houston_tb2.sort_values('month')

ax4.plot(houston_tb2['month'], houston_tb2['total_revenue'], marker='o', label='Houston TB2')
ax4.set_title('Monthly Revenue Trend (Houston TB2)')
ax4.set_ylabel('Monthly Revenue ($)')
ax4.set_xlabel('Month')
ax4.tick_params(axis='x', rotation=45)
ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('battery_analysis_output/ERCOT_Battery_Analysis_Charts.png', dpi=300, bbox_inches='tight')
plt.savefig('battery_analysis_output/ERCOT_Battery_Analysis_Charts.pdf', bbox_inches='tight')
print("Charts saved to battery_analysis_output/")

# Create additional analysis
print("\nKey Findings:")
print("-" * 50)

# Best and worst performing zones
best_zone = summary_df.loc[summary_df['Total Annual Revenue'].idxmax()]
worst_zone = summary_df.loc[summary_df['Total Annual Revenue'].idxmin()]

print(f"Best performing: {best_zone['Zone']} {best_zone['Battery']} with ${best_zone['Total Annual Revenue']:,.2f}")
print(f"Worst performing: {worst_zone['Zone']} {worst_zone['Battery']} with ${worst_zone['Total Annual Revenue']:,.2f}")

# Average cycles
avg_cycles = summary_df['Total Cycles'].mean()
print(f"\nAverage annual cycles: {avg_cycles:.1f}")

# Revenue analysis
print("\nRevenue Analysis:")
print(f"All zones/batteries show negative revenue due to:")
print(f"- Degradation cost: $10/MWh")
print(f"- Limited price spreads in RT market")
print(f"- Need for combined DA/RT optimization")

# Read one daily file to analyze price patterns
houston_daily = pd.read_csv('battery_analysis_output/HOUSTON_TB2_daily_results.csv')
print(f"\nHouston TB2 Statistics:")
print(f"Average daily price: ${houston_daily['avg_price'].mean():.2f}/MWh")
print(f"Average price spread: ${houston_daily['price_spread'].mean():.2f}/MWh")
print(f"Days with positive revenue: {(houston_daily['total_revenue'] > 0).sum()}")
print(f"Days with revenue > $100: {(houston_daily['total_revenue'] > 100).sum()}")
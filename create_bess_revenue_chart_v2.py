#!/usr/bin/env python3
"""
Create BESS revenue chart using ACTUAL EFFECTIVE CAPACITY from DAM AS awards.

The issue: Some batteries have nameplate capacity X but only offer Y MW into AS markets.
Revenue/kW should be calculated based on the capacity actually participating in markets.
"""

import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load all years
dfs = []
for year in range(2019, 2025):
    file = f"bess_revenue_{year}.csv"
    if Path(file).exists():
        df = pd.read_csv(file)
        dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)

print(f"Loaded {len(df_all)} battery-year records")

# Calculate EFFECTIVE CAPACITY for each battery from DAM AS awards
dam_dir = Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/rollup_files/DAM_Gen_Resources")

effective_capacities = {}

for year in range(2019, 2025):
    dam_file = dam_dir / f"{year}.parquet"
    if not dam_file.exists():
        continue

    df_dam = pl.read_parquet(dam_file)

    # Get unique batteries
    batteries = df_all[df_all['year'] == year]['gen_resource'].unique()

    for battery in batteries:
        if pd.isna(battery):
            continue

        df_bat = df_dam.filter(pl.col("ResourceName") == battery)
        if len(df_bat) == 0:
            continue

        try:
            # Max AS award = effective capacity
            regup_max = df_bat.select(pl.col("RegUpAwarded").max()).item() or 0
            regdown_max = df_bat.select(pl.col("RegDownAwarded").max()).item() or 0
            rrs_max = df_bat.select(pl.col("RRSAwarded").max()).item() or 0
            ecrs_max = df_bat.select(pl.col("ECRSAwarded").max()).item() or 0
            nonspin_max = df_bat.select(pl.col("NonSpinAwarded").max()).item() or 0

            effective_cap = max(regup_max, regdown_max, rrs_max, ecrs_max, nonspin_max)

            if effective_cap > 0:
                # Use max effective capacity across all years
                if battery not in effective_capacities:
                    effective_capacities[battery] = effective_cap
                else:
                    effective_capacities[battery] = max(effective_capacities[battery], effective_cap)
        except:
            continue

print(f"Calculated effective capacities for {len(effective_capacities)} batteries")

# Add effective capacity to dataframe
df_all['effective_capacity_mw'] = df_all['gen_resource'].map(effective_capacities)

# Filter to batteries with effective capacity data
df_all = df_all[df_all['effective_capacity_mw'].notna()].copy()

print(f"Batteries with effective capacity: {len(df_all)}")

# Aggregate by battery across all years
revenue_components = df_all.groupby('bess_name').agg({
    'effective_capacity_mw': 'max',  # Use max across years
    'capacity_mw': 'first',  # Original stated capacity
    'dam_discharge_revenue': 'sum',
    'rt_net_revenue': 'sum',
    'dam_as_gen_regup': 'sum',
    'dam_as_gen_regdown': 'sum',
    'dam_as_load_regup': 'sum',
    'dam_as_load_regdown': 'sum',
    'dam_as_gen_rrs': 'sum',
    'dam_as_load_rrs': 'sum',
    'dam_as_gen_ecrs': 'sum',
    'dam_as_load_ecrs': 'sum',
    'dam_as_gen_nonspin': 'sum',
    'dam_as_load_nonspin': 'sum',
    'total_revenue': 'sum',
}).reset_index()

years_operating = df_all.groupby('bess_name')['year'].nunique()
revenue_components['years_operating'] = revenue_components['bess_name'].map(years_operating)

# Convert to $/kW using EFFECTIVE capacity
revenue_components['da_per_kw'] = revenue_components['dam_discharge_revenue'] / (revenue_components['effective_capacity_mw'] * 1000 * revenue_components['years_operating'])
revenue_components['rt_per_kw'] = revenue_components['rt_net_revenue'] / (revenue_components['effective_capacity_mw'] * 1000 * revenue_components['years_operating'])
revenue_components['regup_per_kw'] = (revenue_components['dam_as_gen_regup'] + revenue_components['dam_as_load_regup']) / (revenue_components['effective_capacity_mw'] * 1000 * revenue_components['years_operating'])
revenue_components['regdown_per_kw'] = (revenue_components['dam_as_gen_regdown'] + revenue_components['dam_as_load_regdown']) / (revenue_components['effective_capacity_mw'] * 1000 * revenue_components['years_operating'])
revenue_components['reserves_per_kw'] = (revenue_components['dam_as_gen_rrs'] + revenue_components['dam_as_load_rrs']) / (revenue_components['effective_capacity_mw'] * 1000 * revenue_components['years_operating'])
revenue_components['ecrs_per_kw'] = (revenue_components['dam_as_gen_ecrs'] + revenue_components['dam_as_load_ecrs']) / (revenue_components['effective_capacity_mw'] * 1000 * revenue_components['years_operating'])
revenue_components['nonspin_per_kw'] = (revenue_components['dam_as_gen_nonspin'] + revenue_components['dam_as_load_nonspin']) / (revenue_components['effective_capacity_mw'] * 1000 * revenue_components['years_operating'])
revenue_components['total_per_kw'] = revenue_components['total_revenue'] / (revenue_components['effective_capacity_mw'] * 1000 * revenue_components['years_operating'])

# Sort by total revenue per kW
revenue_components = revenue_components.sort_values('total_per_kw', ascending=False)

print("\nTop 10 batteries by revenue/kW (using effective capacity):")
print(revenue_components[['bess_name', 'total_per_kw', 'effective_capacity_mw', 'capacity_mw', 'years_operating']].head(10))

# Create stacked bar chart
fig, ax = plt.subplots(figsize=(20, 10))

x = np.arange(len(revenue_components))
width = 0.8

colors = {
    'da': '#8B7BC8',
    'rt': '#5FD4AF',
    'regup': '#F4E76E',
    'regdown': '#F4A6A6',
    'reserves': '#5DADE2',
    'ecrs': '#D7BDE2',
    'nonspin': '#34495E'
}

bottom = np.zeros(len(revenue_components))

for component, color in colors.items():
    col = f'{component}_per_kw'
    ax.bar(x, revenue_components[col], width, bottom=bottom,
           color=color, label=component.upper().replace('_', ' '))
    bottom += revenue_components[col].values

ax.set_ylabel('Revenue ($/kW)', fontsize=14, fontweight='bold')
ax.set_title('ERCOT BESS Fleet Revenue by Source (2019-2024)\nUsing Effective Capacity from DAM AS Awards',
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(revenue_components['bess_name'], rotation=90, ha='right', fontsize=8)
ax.legend(loc='upper right', fontsize=10)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.axhline(y=0, color='black', linewidth=0.8)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'${int(y)}'))

plt.tight_layout()
plt.savefig('bess_revenue_stacked_chart_v2.png', dpi=300, bbox_inches='tight')
print("\n✅ Saved: bess_revenue_stacked_chart_v2.png")

# Save summary
summary = revenue_components[[
    'bess_name', 'effective_capacity_mw', 'capacity_mw', 'years_operating',
    'da_per_kw', 'rt_per_kw', 'regup_per_kw', 'regdown_per_kw',
    'reserves_per_kw', 'ecrs_per_kw', 'nonspin_per_kw', 'total_per_kw'
]].copy()

summary.to_csv('bess_revenue_summary_effective_capacity.csv', index=False)
print("✅ Saved: bess_revenue_summary_effective_capacity.csv")

# Stats
print(f"\n📊 Fleet Statistics (Effective Capacity):")
print(f"Batteries: {len(revenue_components)}")
print(f"Total effective capacity: {revenue_components['effective_capacity_mw'].sum():.1f} MW")
print(f"Total stated capacity: {revenue_components['capacity_mw'].sum():.1f} MW")
print(f"Avg revenue/kW: ${revenue_components['total_per_kw'].mean():.2f}")
print(f"Median revenue/kW: ${revenue_components['total_per_kw'].median():.2f}")
print(f"Max revenue/kW: ${revenue_components['total_per_kw'].max():.2f} ({revenue_components.iloc[0]['bess_name']})")

# Revenue mix
total_da = revenue_components['da_per_kw'].sum()
total_rt = revenue_components['rt_per_kw'].sum()
total_regup = revenue_components['regup_per_kw'].sum()
total_regdown = revenue_components['regdown_per_kw'].sum()
total_reserves = revenue_components['reserves_per_kw'].sum()
total_ecrs = revenue_components['ecrs_per_kw'].sum()
total_nonspin = revenue_components['nonspin_per_kw'].sum()
total_all = total_da + total_rt + total_regup + total_regdown + total_reserves + total_ecrs + total_nonspin

print(f"\n💰 Revenue Mix:")
print(f"  DA Energy: {100*total_da/total_all:.1f}%")
print(f"  RT Energy: {100*total_rt/total_all:.1f}%")
print(f"  Reg Up: {100*total_regup/total_all:.1f}%")
print(f"  Reg Down: {100*total_regdown/total_all:.1f}%")
print(f"  Reserves: {100*total_reserves/total_all:.1f}%")
print(f"  ECRS: {100*total_ecrs/total_all:.1f}%")
print(f"  Non-Spin: {100*total_nonspin/total_all:.1f}%")

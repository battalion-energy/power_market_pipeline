#!/usr/bin/env python3
"""
Create stacked bar chart visualization of BESS revenue by source (2019-2024).
Matches format from Gridmatic market report.

I/O locations are driven by .env:
- Input CSVs:   $ERCOT_DATA_DIR/bess_revenue/bess_revenue_{year}.csv
- Output chart: $CHARTS_OUTPUT_DIR/bess_revenue_stacked_chart.png
- Output CSV:   $CHARTS_OUTPUT_DIR/bess_revenue_summary_by_source.csv
"""

import os
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

load_dotenv()

ERCOT_DATA_DIR = Path(os.getenv('ERCOT_DATA_DIR', '/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data'))
CHARTS_OUTPUT_DIR = Path(os.getenv('CHARTS_OUTPUT_DIR', 'charts_output'))
CHARTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load all years
dfs = []
for year in range(2019, 2025):
    # Prefer $ERCOT_DATA_DIR/bess_revenue, fallback to CWD
    file_env = ERCOT_DATA_DIR / 'bess_revenue' / f"bess_revenue_{year}.csv"
    file_cwd = Path(f"bess_revenue_{year}.csv")
    file = file_env if file_env.exists() else file_cwd
    if file.exists():
        df = pd.read_csv(file)
        dfs.append(df)
    else:
        print(f"Warning: {file} not found")

# Combine all years
df_all = pd.concat(dfs, ignore_index=True)

print(f"Loaded {len(df_all)} battery-year records")
print(f"Years: {sorted(df_all['year'].unique())}")
print(f"Unique batteries: {df_all['bess_name'].nunique()}")

# Aggregate by battery across all years
revenue_components = df_all.groupby('bess_name').agg({
    'capacity_mw': 'first',
    'dam_discharge_revenue': 'sum',  # DA energy
    'rt_net_revenue': 'sum',  # RT net (discharge - charge)
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

# Calculate revenue per kW (total revenue / capacity / all years)
# Start with average reported capacity across years, then override with latest mapping
capacity_by_battery = df_all.groupby('bess_name')['capacity_mw'].mean()
years_operating = df_all.groupby('bess_name')['year'].nunique()

revenue_components['years_operating'] = revenue_components['bess_name'].map(years_operating)
revenue_components['avg_capacity_mw'] = revenue_components['bess_name'].map(capacity_by_battery)

# Override capacity with mapping (authoritative) when available
try:
    mapping = pd.read_csv('bess_mapping/BESS_UNIFIED_MAPPING_V3_CLARIFIED.csv')
    if 'BESS_Gen_Resource' in mapping.columns and 'IQ_Capacity_MW' in mapping.columns:
        cap_map = dict(zip(mapping['BESS_Gen_Resource'], mapping['IQ_Capacity_MW']))
        revenue_components['avg_capacity_mw'] = revenue_components.apply(
            lambda r: cap_map.get(r['bess_name'], r['avg_capacity_mw']), axis=1
        )
except Exception:
    pass

# Convert to $/kW (revenue / (capacity_mw * 1000) / years)
revenue_components['da_per_kw'] = revenue_components['dam_discharge_revenue'] / (revenue_components['avg_capacity_mw'] * 1000 * revenue_components['years_operating'])
revenue_components['rt_per_kw'] = revenue_components['rt_net_revenue'] / (revenue_components['avg_capacity_mw'] * 1000 * revenue_components['years_operating'])

# Combine all AS components (both gen and load side)
revenue_components['regup_per_kw'] = (revenue_components['dam_as_gen_regup'] + revenue_components['dam_as_load_regup']) / (revenue_components['avg_capacity_mw'] * 1000 * revenue_components['years_operating'])
revenue_components['regdown_per_kw'] = (revenue_components['dam_as_gen_regdown'] + revenue_components['dam_as_load_regdown']) / (revenue_components['avg_capacity_mw'] * 1000 * revenue_components['years_operating'])
revenue_components['reserves_per_kw'] = (revenue_components['dam_as_gen_rrs'] + revenue_components['dam_as_load_rrs']) / (revenue_components['avg_capacity_mw'] * 1000 * revenue_components['years_operating'])
revenue_components['ecrs_per_kw'] = (revenue_components['dam_as_gen_ecrs'] + revenue_components['dam_as_load_ecrs']) / (revenue_components['avg_capacity_mw'] * 1000 * revenue_components['years_operating'])
revenue_components['nonspin_per_kw'] = (revenue_components['dam_as_gen_nonspin'] + revenue_components['dam_as_load_nonspin']) / (revenue_components['avg_capacity_mw'] * 1000 * revenue_components['years_operating'])

revenue_components['total_per_kw'] = revenue_components['total_revenue'] / (revenue_components['avg_capacity_mw'] * 1000 * revenue_components['years_operating'])

# Sort by total revenue per kW (highest to lowest)
revenue_components = revenue_components.sort_values('total_per_kw', ascending=False)

print("\nTop 10 batteries by revenue/kW:")
print(revenue_components[['bess_name', 'total_per_kw', 'avg_capacity_mw', 'years_operating']].head(10))

# Create stacked bar chart
fig, ax = plt.subplots(figsize=(20, 10))

# Stack components matching the market report legend
x = np.arange(len(revenue_components))
width = 0.8

# Stack order (bottom to top)
colors = {
    'da': '#8B7BC8',      # Purple - DA ($/kW)
    'rt': '#5FD4AF',      # Teal - RT ($/kW)
    'regup': '#F4E76E',   # Yellow - Reg Up ($/kW)
    'regdown': '#F4A6A6', # Pink - Reg Down ($/kW)
    'reserves': '#5DADE2', # Blue - Reserves ($/kW)
    'ecrs': '#D7BDE2',    # Light purple - ECRS ($/kW)
    'nonspin': '#34495E'  # Dark - Non-Spin ($/kW)
}

# Build positive/negative stacks to avoid misleading visuals
pos_bottom = np.zeros(len(revenue_components))
neg_bottom = np.zeros(len(revenue_components))

bars = {}
stack_order = list(colors.keys())
for component in stack_order:
    col = f'{component}_per_kw'
    vals = revenue_components[col].values

    label = component.upper().replace('_', ' ')
    color = colors[component]

    if component in ("da", "rt"):
        # Split energy into positive and negative portions
        pos_vals = np.where(vals > 0, vals, 0.0)
        neg_vals = np.where(vals < 0, vals, 0.0)

        if np.any(neg_vals):
            ax.bar(x, neg_vals, width, bottom=neg_bottom, color=color, label=None)
            neg_bottom += neg_vals
        if np.any(pos_vals):
            bars[component] = ax.bar(x, pos_vals, width, bottom=pos_bottom, color=color, label=label)
            pos_bottom += pos_vals
    else:
        # Ancillary services should not be negative; guard just in case
        vals = np.maximum(vals, 0.0)
        bars[component] = ax.bar(x, vals, width, bottom=pos_bottom, color=color, label=label)
        pos_bottom += vals

# Formatting
ax.set_ylabel('Revenue ($/kW)', fontsize=14, fontweight='bold')
ax.set_title('ERCOT BESS Fleet Revenue by Source (2019-2024 Average)',
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(revenue_components['bess_name'], rotation=90, ha='right', fontsize=8)
ax.legend(loc='upper right', fontsize=10)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.axhline(y=0, color='black', linewidth=0.8)

# Format y-axis
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'${int(y)}'))

plt.tight_layout()
out_png = CHARTS_OUTPUT_DIR / 'bess_revenue_stacked_chart.png'
plt.savefig(out_png, dpi=300, bbox_inches='tight')
print(f"\n✅ Saved: {out_png}")

# Also save summary data
summary = revenue_components[[
    'bess_name', 'avg_capacity_mw', 'years_operating',
    'da_per_kw', 'rt_per_kw', 'regup_per_kw', 'regdown_per_kw',
    'reserves_per_kw', 'ecrs_per_kw', 'nonspin_per_kw', 'total_per_kw'
]].copy()

out_csv = CHARTS_OUTPUT_DIR / 'bess_revenue_summary_by_source.csv'
summary.to_csv(out_csv, index=False)
print(f"✅ Saved: {out_csv}")

# Print summary stats
print(f"\n📊 Fleet Statistics:")
print(f"Total batteries: {len(revenue_components)}")
print(f"Total capacity: {revenue_components['avg_capacity_mw'].sum():.1f} MW")
print(f"Avg revenue/kW: ${revenue_components['total_per_kw'].mean():.2f}")
print(f"Median revenue/kW: ${revenue_components['total_per_kw'].median():.2f}")
print(f"Max revenue/kW: ${revenue_components['total_per_kw'].max():.2f} ({revenue_components.iloc[0]['bess_name']})")
print(f"Min revenue/kW: ${revenue_components['total_per_kw'].min():.2f} ({revenue_components.iloc[-1]['bess_name']})")

# Revenue breakdown percentages
total_da = revenue_components['da_per_kw'].sum()
total_rt = revenue_components['rt_per_kw'].sum()
total_regup = revenue_components['regup_per_kw'].sum()
total_regdown = revenue_components['regdown_per_kw'].sum()
total_reserves = revenue_components['reserves_per_kw'].sum()
total_ecrs = revenue_components['ecrs_per_kw'].sum()
total_nonspin = revenue_components['nonspin_per_kw'].sum()
total_all = total_da + total_rt + total_regup + total_regdown + total_reserves + total_ecrs + total_nonspin

print(f"\n💰 Revenue Mix (fleet-wide):")
print(f"  DA Energy: {100*total_da/total_all:.1f}%")
print(f"  RT Energy: {100*total_rt/total_all:.1f}%")
print(f"  Reg Up: {100*total_regup/total_all:.1f}%")
print(f"  Reg Down: {100*total_regdown/total_all:.1f}%")
print(f"  Reserves: {100*total_reserves/total_all:.1f}%")
print(f"  ECRS: {100*total_ecrs/total_all:.1f}%")
print(f"  Non-Spin: {100*total_nonspin/total_all:.1f}%")

#!/usr/bin/env python3
"""Test single BESS calculation"""

import pandas as pd
import numpy as np
from pathlib import Path

# Load data
data_dir = Path("/home/enrico/data/ERCOT_data")
rollup_dir = data_dir / "rollup_files"

# Load 2024 DA prices
da_prices = pd.read_parquet(rollup_dir / "flattened/DA_prices_2024.parquet")
print(f"DA prices shape: {da_prices.shape}")

# Load 2024 DAM Gen 
dam_gen = pd.read_parquet(rollup_dir / "DAM_Gen_Resources/2024.parquet")
bess = dam_gen[dam_gen['ResourceType'] == 'PWRSTR']
print(f"BESS resources in 2024: {bess['ResourceName'].nunique()}")

# Get first BESS resource
first_bess = bess['ResourceName'].iloc[0]
print(f"\nAnalyzing: {first_bess}")

# Get awards for this BESS
bess_awards = bess[bess['ResourceName'] == first_bess].copy()
print(f"Awards found: {len(bess_awards)}")

# Calculate DA energy revenue
if 'datetime' not in bess_awards.columns:
    bess_awards['datetime'] = pd.to_datetime(bess_awards['DeliveryDate'])

# Use HB_BUSAVG price
bess_awards = bess_awards.merge(
    da_prices[['datetime_ts', 'HB_BUSAVG']].rename(columns={'datetime_ts': 'datetime', 'HB_BUSAVG': 'price'}),
    on='datetime',
    how='left'
)

# Calculate revenue
bess_awards['da_revenue'] = bess_awards['AwardedQuantity'] * bess_awards['price']
total_da_revenue = bess_awards['da_revenue'].sum()
total_mwh = bess_awards['AwardedQuantity'].sum()

print(f"\nResults for {first_bess}:")
print(f"Total MWh awarded: {total_mwh:,.2f}")
print(f"Total DA revenue: ${total_da_revenue:,.2f}")
print(f"Average price: ${bess_awards['price'].mean():.2f}/MWh")

# Check AS awards
as_cols = ['RegUpAwarded', 'RegDownAwarded', 'RRSAwarded', 'NonSpinAwarded', 'ECRSAwarded']
for col in as_cols:
    if col in bess_awards.columns:
        total = bess_awards[col].sum()
        if total > 0:
            print(f"{col}: {total:,.2f} MW")

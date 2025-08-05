#!/usr/bin/env python3
"""Check available dates in DA data."""

import pandas as pd

# Load DA data
da_file = "rt_rust_processor/annual_output/Settlement_Point_Prices_at_Resource_Nodes__Hubs_and_Load_Zones/Settlement_Point_Prices_at_Resource_Nodes__Hubs_and_Load_Zones_2023.parquet"
df = pd.read_parquet(da_file)

# Check date range
print(f"Date range in DA data: {df['DeliveryDate'].min()} to {df['DeliveryDate'].max()}")
print(f"\nUnique dates: {df['DeliveryDate'].nunique()}")
print(f"\nFirst 10 dates: {sorted(df['DeliveryDate'].unique())[:10]}")
print(f"\nLast 10 dates: {sorted(df['DeliveryDate'].unique())[-10:]}")

# Check Houston specifically
houston_df = df[df['SettlementPointName'] == 'LZ_HOUSTON']
print(f"\nHouston dates: {houston_df['DeliveryDate'].nunique()}")
print(f"Houston date range: {houston_df['DeliveryDate'].min()} to {houston_df['DeliveryDate'].max()}")
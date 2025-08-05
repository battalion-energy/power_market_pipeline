#!/usr/bin/env python3
"""Check available settlement points in RT data that match load zones."""

import pandas as pd
import numpy as np

# Load RT data
rt_file = "rt_rust_processor/annual_output/LMPs_by_Resource_Nodes__Load_Zones_and_Trading_Hubs/LMPs_by_Resource_Nodes__Load_Zones_and_Trading_Hubs_2023.parquet"
print("Loading RT data...")
df = pd.read_parquet(rt_file)

# Get unique settlement points
unique_points = df['SettlementPoint'].unique()
print(f"\nTotal unique settlement points: {len(unique_points)}")

# Look for load zones
print("\nSettlement points containing 'LZ':")
lz_points = [p for p in unique_points if 'LZ' in p]
for p in sorted(lz_points):
    print(f"  {p}")

# Look for common zone names
print("\nSettlement points containing zone names:")
zone_keywords = ['HOUSTON', 'NORTH', 'WEST', 'SOUTH', 'AUSTIN', 'CPS', 'LCRA', 'AEN']
for keyword in zone_keywords:
    matching = [p for p in unique_points if keyword in p.upper()]
    if matching:
        print(f"\n{keyword}:")
        for p in sorted(matching)[:10]:  # Show first 10
            print(f"  {p}")
        if len(matching) > 10:
            print(f"  ... and {len(matching)-10} more")

# Check if we have hub data
print("\nSettlement points containing 'HB' (hubs):")
hub_points = [p for p in unique_points if 'HB' in p]
for p in sorted(hub_points)[:20]:
    print(f"  {p}")
if len(hub_points) > 20:
    print(f"  ... and {len(hub_points)-20} more")
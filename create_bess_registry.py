#!/usr/bin/env python3
"""
Create BESS registry parquet file from mapping CSV
"""

import pandas as pd
from pathlib import Path

# Read the mapping CSV
mapping = pd.read_csv("bess_resource_mapping.csv")

# Create registry with required columns
registry = pd.DataFrame({
    'resource_name': mapping['battery_name'],
    'settlement_point': mapping['settlement_points'].fillna('UNKNOWN'),
    'capacity_mw': mapping['max_power_mw'].fillna(100.0),  # Default 100MW if missing
    'duration_hours': mapping['duration_hours'].fillna(2.0),  # Default 2 hours
    'gen_resources': mapping['gen_resources'].fillna(''),
    'load_resources': mapping['load_resources'].fillna(''),
    'is_complete': mapping['is_complete']
})

# Filter to only resources with some data
registry = registry[registry['resource_name'].notna()]

# Save as parquet
output_dir = Path("/home/enrico/data/ERCOT_data/bess_analysis")
output_dir.mkdir(exist_ok=True)
output_file = output_dir / "bess_registry.parquet"

registry.to_parquet(output_file, index=False)
print(f"✅ Created BESS registry with {len(registry)} resources")
print(f"   Saved to: {output_file}")
print(f"   Complete mappings: {registry['is_complete'].sum()}")
print(f"   Average capacity: {registry['capacity_mw'].mean():.1f} MW")
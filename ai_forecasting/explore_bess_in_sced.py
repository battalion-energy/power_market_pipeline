#!/usr/bin/env python3
"""
Explore BESS Resources in SCED Generation Data
================================================

Find how battery storage resources are labeled in SCED data.
Look for fuel type indicators like "STORAGE", "ESR", "BESS", etc.
"""

import polars as pl
from pathlib import Path

print("="*80)
print("EXPLORING BESS IN SCED GENERATION DATA")
print("="*80)

SCED_DIR = Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/rollup_files/SCED_Gen_Resources")

# Load a recent year to see structure
print("\n1. LOADING 2024 DATA TO EXPLORE STRUCTURE")
print("="*80)

df_2024 = pl.read_parquet(SCED_DIR / "2024.parquet")
print(f"Loaded 2024: {len(df_2024):,} records")
print(f"Columns ({len(df_2024.columns)}): {df_2024.columns[:10]}")

# Check all columns
print("\nAll columns:")
for i, col in enumerate(df_2024.columns, 1):
    print(f"  {i:2}. {col}")

# Look for fuel type or resource type columns
fuel_cols = [col for col in df_2024.columns if 'FUEL' in col.upper() or 'TYPE' in col.upper() or 'RESOURCE' in col.upper()]
print(f"\n2. POTENTIAL FUEL/TYPE COLUMNS:")
print("="*80)
for col in fuel_cols:
    print(f"  - {col}")

# Sample data
print("\n3. SAMPLE DATA")
print("="*80)
print(df_2024.head(10))

# Check unique values in fuel-related columns
if fuel_cols:
    print("\n4. UNIQUE VALUES IN FUEL/TYPE COLUMNS")
    print("="*80)
    for col in fuel_cols[:3]:  # First 3 columns
        unique_vals = df_2024.select(pl.col(col).unique()).to_series().to_list()
        print(f"\n{col}:")
        if len(unique_vals) < 50:
            for val in sorted([str(v) for v in unique_vals if v is not None]):
                print(f"  - {val}")
        else:
            print(f"  ({len(unique_vals)} unique values - sampling first 20)")
            for val in sorted([str(v) for v in unique_vals if v is not None])[:20]:
                print(f"  - {val}")

# Search for storage/battery keywords
print("\n5. SEARCHING FOR STORAGE/BATTERY KEYWORDS")
print("="*80)

keywords = ['STORAGE', 'BESS', 'BATTERY', 'ESS', 'ESR', 'STOR']

for col in df_2024.columns:
    col_data = df_2024.select(col).to_series()

    # Check if column contains string data
    if col_data.dtype == pl.Utf8 or col_data.dtype == pl.String:
        for keyword in keywords:
            matches = col_data.str.contains(keyword, literal=False)
            if matches.any():
                count = matches.sum()
                print(f"\n  Found '{keyword}' in column '{col}': {count:,} matches")

                # Show sample
                sample = df_2024.filter(matches).select([col]).head(10)
                print(f"  Sample values:")
                for val in sample.to_series().to_list():
                    print(f"    - {val}")

print("\n" + "="*80)
print("EXPLORATION COMPLETE")
print("="*80)

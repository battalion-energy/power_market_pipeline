#!/usr/bin/env python3
import pandas as pd

# Check DAM Load Resources
print("="*80)
print("DAM LOAD RESOURCES - 2024")
print("="*80)
df = pd.read_parquet('/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/rollup_files/DAM_Load_Resources/2024.parquet')
print(f"\nTotal rows: {len(df):,}")
print(f"\nColumns ({len(df.columns)}):")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:2d}. {col}")

# Check for BESS load resources
print("\n" + "="*80)
print("BESS LOAD RESOURCES (sample)")
print("="*80)
bess_mask = df['Load Resource Name'].str.contains('_LD[0-9]', na=False, regex=True)
bess_df = df[bess_mask]
print(f"\nTotal BESS Load Resources found: {bess_df['Load Resource Name'].nunique()}")
print(f"\nSample BESS Load Resources:")
sample = bess_df[['Load Resource Name', 'Delivery Date', 'Hour Ending']].drop_duplicates('Load Resource Name').head(10)
print(sample.to_string(index=False))

# Check for energy-related columns
print("\n" + "="*80)
print("LOOKING FOR ENERGY AWARD COLUMNS")
print("="*80)
energy_cols = [col for col in df.columns if 'award' in col.lower() or 'quantity' in col.lower() or 'energy' in col.lower()]
print(f"\nEnergy-related columns: {energy_cols}")

# Sample one BESS to see all data
print("\n" + "="*80)
print("SAMPLE BESS DATA (first row)")
print("="*80)
if len(bess_df) > 0:
    sample_row = bess_df.iloc[0]
    for col in df.columns:
        val = sample_row[col]
        if pd.notna(val) and val != 0:
            print(f"  {col:50s}: {val}")

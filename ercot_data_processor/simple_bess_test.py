#!/usr/bin/env python3
"""Simplified BESS test for fast comparison"""

import pandas as pd
import numpy as np
from pathlib import Path
import time

data_dir = Path("/home/enrico/data/ERCOT_data/rollup_files")

# Process just first 10 BESS for 2024
print("=" * 60)
print("SIMPLIFIED BESS REVENUE TEST - 2024")
print("=" * 60)

start_time = time.time()

# Load data
da_prices = pd.read_parquet(data_dir / "flattened/DA_prices_2024.parquet")
as_prices = pd.read_parquet(data_dir / "flattened/AS_prices_2024.parquet")
dam_gen = pd.read_parquet(data_dir / "DAM_Gen_Resources/2024.parquet")

# Filter for BESS
bess = dam_gen[dam_gen['ResourceType'] == 'PWRSTR']
unique_bess = bess['ResourceName'].unique()[:10]  # Just first 10

print(f"\nProcessing {len(unique_bess)} BESS resources")

results = []
for bess_name in unique_bess:
    bess_data = bess[bess['ResourceName'] == bess_name].copy()
    if bess_data.empty:
        continue
    
    # Merge with DA prices (use HB_BUSAVG)
    if 'datetime' not in bess_data.columns:
        bess_data['datetime'] = pd.to_datetime(bess_data['DeliveryDate'])
    
    bess_data = bess_data.merge(
        da_prices[['datetime_ts', 'HB_BUSAVG']].rename(columns={'datetime_ts': 'datetime'}),
        on='datetime',
        how='left'
    )
    
    # Calculate DA energy revenue
    da_revenue = (bess_data['AwardedQuantity'] * bess_data['HB_BUSAVG']).sum()
    
    # Calculate AS revenues
    as_revenue = 0
    if 'RegUpAwarded' in bess_data.columns:
        # Merge with AS prices
        bess_with_as = bess_data.merge(
            as_prices[['datetime_ts', 'REGUP', 'REGDN']].rename(columns={'datetime_ts': 'datetime'}),
            on='datetime',
            how='left'
        )
        as_revenue = (
            (bess_with_as['RegUpAwarded'] * bess_with_as['REGUP']).sum() +
            (bess_with_as.get('RegDownAwarded', 0) * bess_with_as.get('REGDN', 0)).sum()
        )
    
    total_revenue = da_revenue + as_revenue
    
    results.append({
        'resource': bess_name,
        'da_revenue': da_revenue,
        'as_revenue': as_revenue,
        'total_revenue': total_revenue
    })
    
    print(f"  {bess_name}: DA=${da_revenue:,.0f}, AS=${as_revenue:,.0f}, Total=${total_revenue:,.0f}")

elapsed = time.time() - start_time

print(f"\n⏱️  Processing time: {elapsed:.2f} seconds")
print(f"📊 Total resources processed: {len(results)}")
print(f"💵 Total revenue: ${sum(r['total_revenue'] for r in results):,.0f}")

# Save results
results_df = pd.DataFrame(results)
results_df.to_parquet('/tmp/python_bess_results.parquet')
print("\n✅ Results saved to /tmp/python_bess_results.parquet")

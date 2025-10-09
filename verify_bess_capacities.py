#!/usr/bin/env python3
"""
Verify BESS capacities by comparing mapping file against actual DAM AS awards.
Batteries can't receive more AS awards than their capacity.
"""

import polars as pl
import pandas as pd
from pathlib import Path

# Load mapping
df_mapping = pd.read_csv('bess_mapping/BESS_UNIFIED_MAPPING_V3_CLARIFIED.csv')
df_mapping = df_mapping[df_mapping['True_Operational_Status'] == 'Operational (has Load Resource)']

print(f"Checking {len(df_mapping)} operational BESS units\n")

# Check capacities against 2022 DAM awards (good year with lots of data)
dam_file = "/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/rollup_files/DAM_Gen_Resources/2022.parquet"
df_dam = pl.read_parquet(dam_file)

issues = []

for _, row in df_mapping.iterrows():
    gen_resource = row['BESS_Gen_Resource']
    stated_capacity = row['IQ_Capacity_MW']

    if pd.isna(stated_capacity) or stated_capacity == 0:
        continue

    # Get max AS awards for this battery
    df_resource = df_dam.filter(pl.col("ResourceName") == gen_resource)

    if len(df_resource) == 0:
        continue

    try:
        regup_max = df_resource.select(pl.col("RegUpAwarded").max()).item() or 0
        regdown_max = df_resource.select(pl.col("RegDownAwarded").max()).item() or 0
        rrs_max = df_resource.select(pl.col("RRSAwarded").max()).item() or 0
        ecrs_max = df_resource.select(pl.col("ECRSAwarded").max()).item() or 0
        nonspin_max = df_resource.select(pl.col("NonSpinAwarded").max()).item() or 0

        inferred_capacity = max(regup_max, regdown_max, rrs_max, ecrs_max, nonspin_max)

        # Check for significant mismatch (>10% difference or >5 MW)
        if inferred_capacity > 0:
            diff_pct = abs(inferred_capacity - stated_capacity) / stated_capacity * 100
            diff_mw = abs(inferred_capacity - stated_capacity)

            if diff_pct > 10 and diff_mw > 5:
                issues.append({
                    'gen_resource': gen_resource,
                    'stated_capacity': stated_capacity,
                    'inferred_capacity': inferred_capacity,
                    'diff_mw': diff_mw,
                    'diff_pct': diff_pct,
                    'regup_max': regup_max,
                    'regdown_max': regdown_max,
                    'rrs_max': rrs_max
                })
    except Exception as e:
        print(f"Error checking {gen_resource}: {e}")
        continue

# Report issues
if issues:
    df_issues = pd.DataFrame(issues).sort_values('diff_pct', ascending=False)
    print("⚠️  Capacity Mismatches Found:\n")
    print(df_issues[['gen_resource', 'stated_capacity', 'inferred_capacity', 'diff_mw', 'diff_pct']].to_string(index=False))

    print(f"\n\nTotal issues: {len(issues)}")
    print(f"\nProposed fixes:")
    for _, issue in df_issues.iterrows():
        print(f"  {issue['gen_resource']}: {issue['stated_capacity']} MW → {issue['inferred_capacity']} MW")
else:
    print("✅ All capacities verified - no significant mismatches found!")

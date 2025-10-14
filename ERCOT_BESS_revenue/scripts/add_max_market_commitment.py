#!/usr/bin/env python3
"""
Add Max_Market_Commitment_MW column to BESS mapping file.

This represents the maximum simultaneous commitment observed across all years:
  Max hour of: DAM Energy + RegUp + RRS + ECRS + NonSpin
"""

import pandas as pd
from pathlib import Path

# Load mapping file
mapping_file = Path("bess_mapping/BESS_UNIFIED_MAPPING_V3_CLARIFIED.csv")
df_mapping = pd.read_csv(mapping_file)

print(f"Loaded mapping file: {len(df_mapping)} batteries")
print(f"Columns: {len(df_mapping.columns)}")

# Load capacity analysis results
capacity_file = Path("bess_capacity_summary.csv")
df_capacity = pd.read_csv(capacity_file)

print(f"Loaded capacity analysis: {len(df_capacity)} batteries with market data")

# Create lookup dictionary: gen_resource -> max_total_commitment
capacity_lookup = df_capacity.set_index('gen_resource')['max_total_commitment'].to_dict()

# Add new column to mapping
df_mapping['Max_Market_Commitment_MW'] = df_mapping['BESS_Gen_Resource'].map(capacity_lookup)

# Calculate commitment ratio (for monitoring)
df_mapping['Market_Commitment_Ratio'] = (
    df_mapping['Max_Market_Commitment_MW'] / df_mapping['IQ_Capacity_MW']
).round(3)

# Summary statistics
batteries_with_data = df_mapping['Max_Market_Commitment_MW'].notna().sum()
operational = df_mapping[df_mapping['True_Operational_Status'] == 'Operational (has Load Resource)']
operational_with_data = operational['Max_Market_Commitment_MW'].notna().sum()

print(f"\n✅ Added Max_Market_Commitment_MW column")
print(f"   Batteries with market data: {batteries_with_data}/{len(df_mapping)}")
print(f"   Operational batteries with data: {operational_with_data}/{len(operational)}")

# Show statistics for operational batteries
if len(operational[operational['Max_Market_Commitment_MW'].notna()]) > 0:
    operational_data = operational[operational['Max_Market_Commitment_MW'].notna()]

    print(f"\n📊 Operational BESS Market Commitment:")
    print(f"   Total nameplate: {operational_data['IQ_Capacity_MW'].sum():.1f} MW")
    print(f"   Total committed: {operational_data['Max_Market_Commitment_MW'].sum():.1f} MW")
    print(f"   Fleet commitment ratio: {operational_data['Max_Market_Commitment_MW'].sum() / operational_data['IQ_Capacity_MW'].sum():.1%}")

    # Batteries with issues
    over_committed = operational_data[operational_data['Market_Commitment_Ratio'] > 1.1]
    under_committed = operational_data[operational_data['Market_Commitment_Ratio'] < 0.5]

    print(f"\n⚠️  Potential Issues:")
    print(f"   Over-committed (>110%): {len(over_committed)} batteries")
    if len(over_committed) > 0:
        for _, row in over_committed.nlargest(5, 'Market_Commitment_Ratio').iterrows():
            print(f"      {row['BESS_Gen_Resource']}: {row['IQ_Capacity_MW']:.1f} MW → {row['Max_Market_Commitment_MW']:.1f} MW ({row['Market_Commitment_Ratio']:.1%})")

    print(f"\n   Under-committed (<50%): {len(under_committed)} batteries")
    if len(under_committed) > 0:
        for _, row in under_committed.nsmallest(5, 'Market_Commitment_Ratio').iterrows():
            print(f"      {row['BESS_Gen_Resource']}: {row['IQ_Capacity_MW']:.1f} MW → {row['Max_Market_Commitment_MW']:.1f} MW ({row['Market_Commitment_Ratio']:.1%})")

# Save updated mapping file
output_file = mapping_file
df_mapping.to_csv(output_file, index=False)

print(f"\n✅ Updated mapping file: {output_file}")
print(f"   New columns: Max_Market_Commitment_MW, Market_Commitment_Ratio")

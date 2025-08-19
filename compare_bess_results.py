#!/usr/bin/env python3
"""Compare BESS results between Python and Rust implementations"""

import pandas as pd
import numpy as np

print("=" * 60)
print("BESS REVENUE COMPARISON - PYTHON VS RUST")
print("=" * 60)

# Load Python results
try:
    python_results = pd.read_parquet('/tmp/python_bess_results.parquet')
    print("\n✅ Python Results Loaded:")
    print(f"   Resources: {len(python_results)}")
    print(f"   Total Revenue: ${python_results['total_revenue'].sum():,.0f}")
    print("\nTop 5 BESS by revenue (Python):")
    print(python_results.nlargest(5, 'total_revenue')[['resource', 'da_revenue', 'as_revenue', 'total_revenue']])
except Exception as e:
    print(f"❌ Could not load Python results: {e}")
    python_results = None

# Load Rust results (if available)
try:
    rust_results = pd.read_parquet('/tmp/rust_bess_results.parquet')
    print("\n✅ Rust Results Loaded:")
    print(f"   Resources: {len(rust_results)}")
    print(f"   Total Revenue: ${rust_results['total_revenue'].sum():,.0f}")
    print("\nTop 5 BESS by revenue (Rust):")
    print(rust_results.nlargest(5, 'total_revenue')[['resource', 'da_revenue', 'as_revenue', 'total_revenue']])
except Exception as e:
    print(f"\n⚠️  No Rust results available yet: {e}")
    rust_results = None

# Compare if both available
if python_results is not None and rust_results is not None:
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    
    # Merge on resource name
    comparison = pd.merge(
        python_results, 
        rust_results, 
        on='resource', 
        suffixes=('_python', '_rust')
    )
    
    # Calculate differences
    comparison['da_diff'] = comparison['da_revenue_rust'] - comparison['da_revenue_python']
    comparison['as_diff'] = comparison['as_revenue_rust'] - comparison['as_revenue_python']
    comparison['total_diff'] = comparison['total_revenue_rust'] - comparison['total_revenue_python']
    
    print("\nRevenue Differences (Rust - Python):")
    print(comparison[['resource', 'total_revenue_python', 'total_revenue_rust', 'total_diff']])
    
    # Summary statistics
    print(f"\nTotal Revenue Difference: ${comparison['total_diff'].sum():,.0f}")
    print(f"Mean Absolute Difference: ${abs(comparison['total_diff']).mean():,.0f}")
    print(f"Max Difference: ${comparison['total_diff'].max():,.0f}")
    print(f"Min Difference: ${comparison['total_diff'].min():,.0f}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

if python_results is not None:
    print(f"Python Implementation:")
    print(f"  - Total Resources: {len(python_results)}")
    print(f"  - Total Revenue: ${python_results['total_revenue'].sum():,.0f}")
    print(f"  - Avg Revenue/Resource: ${python_results['total_revenue'].mean():,.0f}")
    
if rust_results is not None:
    print(f"\nRust Implementation:")
    print(f"  - Total Resources: {len(rust_results)}")
    print(f"  - Total Revenue: ${rust_results['total_revenue'].sum():,.0f}")
    print(f"  - Avg Revenue/Resource: ${rust_results['total_revenue'].mean():,.0f}")

if python_results is None and rust_results is None:
    print("❌ No results available from either implementation")
elif rust_results is None:
    print("\n⚠️  Rust implementation not yet available for comparison")
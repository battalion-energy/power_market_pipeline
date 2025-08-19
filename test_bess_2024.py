#!/usr/bin/env python3
"""Test BESS calculator for 2024 only"""

from unified_bess_revenue_calculator import UnifiedBESSCalculator
import logging

logging.basicConfig(level=logging.INFO)

calculator = UnifiedBESSCalculator()

# Process just 2024
df = calculator.process_all_years(start_year=2024, end_year=2024)

if df.empty:
    print("No revenue data calculated")
else:
    print(f"\n2024 Results:")
    print(f"Total resources: {len(df['resource_name'].unique())}")
    print(f"Total records: {len(df)}")
    print(f"Total revenue: ${df['total_revenue'].sum():,.2f}")
    
    # Top 10
    top10 = df.groupby('resource_name')['total_revenue'].sum().nlargest(10)
    print("\nTop 10 BESS by revenue:")
    for resource, revenue in top10.items():
        print(f"  {resource}: ${revenue:,.2f}")
    
    # Save results
    leaderboard = calculator.create_leaderboard(df)
    output_dir = calculator.save_to_database_format(df, leaderboard)
    print(f"\nResults saved to: {output_dir}")

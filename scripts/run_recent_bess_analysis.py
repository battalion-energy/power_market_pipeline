#!/usr/bin/env python3
"""
Run BESS revenue analysis for recent data only
Focus on 2024 data where we have more BESS resources
"""

import pandas as pd
from datetime import datetime, timedelta
from comprehensive_bess_revenue_calculator_v2 import ComprehensiveBessCalculator
import warnings
warnings.filterwarnings('ignore')

def run_recent_analysis():
    """Run analysis for recent months with better tracking"""
    
    # Focus on 2024 data (January to October)
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 10, 31)
    
    print(f"Running BESS revenue analysis for {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print("="*80)
    
    calculator = ComprehensiveBessCalculator(start_date, end_date)
    
    # First identify resources
    print("\nPhase 1: Identifying BESS resources...")
    earliest = calculator.identify_all_bess_resources()
    
    if not calculator.bess_resources:
        print("No BESS resources found!")
        return
        
    print(f"\nFound {len(calculator.bess_resources)} BESS resources")
    print("\nTop 10 BESS by capacity:")
    sorted_bess = sorted(calculator.bess_resources.items(), 
                        key=lambda x: x[1]['capacity_mw'], 
                        reverse=True)[:10]
    
    for name, info in sorted_bess:
        print(f"  {name:30s} {info['capacity_mw']:6.1f} MW  QSE: {info['qse']}")
    
    # Process data
    print("\nPhase 2: Processing revenue data...")
    print("This will take a few minutes...\n")
    
    calculator.process_historical_data()
    
    # Quick summary from results
    if not calculator.results_monthly.empty:
        print("\n\nQuick Monthly Summary:")
        monthly_summary = calculator.results_monthly.groupby(['year', 'month']).agg({
            'total_revenue': 'sum',
            'energy_revenue': 'sum',
            'total_as_revenue': 'sum',
            'resource_name': 'nunique'
        }).rename(columns={'resource_name': 'active_bess'})
        
        monthly_summary['avg_revenue_per_bess'] = monthly_summary['total_revenue'] / monthly_summary['active_bess']
        
        print(f"\n{'Year-Month':>10} {'BESS':>6} {'Total Rev':>15} {'Energy Rev':>15} {'AS Rev':>15} {'Avg/BESS':>15}")
        print("-" * 90)
        
        for (year, month), row in monthly_summary.iterrows():
            print(f"{year:4d}-{month:02d}    {row['active_bess']:>6} "
                  f"${row['total_revenue']:>14,.0f} ${row['energy_revenue']:>14,.0f} "
                  f"${row['total_as_revenue']:>14,.0f} ${row['avg_revenue_per_bess']:>14,.0f}")
    
    return calculator

if __name__ == "__main__":
    calculator = run_recent_analysis()
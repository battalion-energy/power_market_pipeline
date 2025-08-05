#!/usr/bin/env python3
"""
Run BESS historical revenue analysis
Start with a small date range to test, then expand
"""

import sys
from datetime import datetime, timedelta
from comprehensive_bess_revenue_calculator_v2 import ComprehensiveBessCalculator

def run_analysis(start_date, end_date):
    """Run analysis for specified date range"""
    print(f"\nRunning BESS revenue analysis from {start_date} to {end_date}")
    print("="*80)
    
    calculator = ComprehensiveBessCalculator(start_date, end_date)
    calculator.process_historical_data()

if __name__ == "__main__":
    # Start with October 2024 (we know we have data)
    test_start = datetime(2024, 10, 1)
    test_end = datetime(2024, 10, 7)  # One week
    
    print("Phase 1: Testing with one week of data...")
    run_analysis(test_start, test_end)
    
    # If successful, run for full historical range
    print("\n\nContinue with full historical analysis? (y/n)")
    response = input().strip().lower()
    
    if response == 'y':
        # Run from first PWRSTR appearance (April 2018) to 60 days ago
        historical_start = datetime(2018, 4, 1)
        historical_end = datetime.now() - timedelta(days=60)
        
        print("\nPhase 2: Running full historical analysis...")
        print("This may take several hours...")
        run_analysis(historical_start, historical_end)
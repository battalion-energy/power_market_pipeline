#!/usr/bin/env python3
"""
Run complete TBX analysis for all years and all settlement points
"""

from calculate_tbx_all_nodes import TBXCalculatorAllNodes
import sys

if __name__ == "__main__":
    print("🚀 Starting FULL TBX Analysis for ALL settlement points")
    print("=" * 80)
    
    # Initialize calculator
    calculator = TBXCalculatorAllNodes(efficiency=0.9)
    
    # Run for all available years
    years = [2021, 2022, 2023, 2024, 2025]
    
    print(f"📅 Processing years: {years}")
    print("⚠️  This will take some time - processing ~1000 nodes × 5 years")
    
    try:
        calculator.run(years=years)
        print("\n✅ Full TBX analysis complete!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
"""
Test script to verify the download system works with a single dataset.
This will test downloading Wind Power Production data.
"""

import os
import sys

# Test configuration
TEST_DATASET_KEY = "NP4-732-CD"  # Wind Power Production - Hourly Averaged Actual and Forecasted Values

print("="*80)
print("ERCOT Download System - Test Script")
print("="*80)
print()
print(f"This will test downloading a single dataset: {TEST_DATASET_KEY}")
print("Dataset: Wind Power Production - Hourly Averaged Actual and Forecasted Values")
print()
print("The test will:")
print("1. Load the catalog CSV")
print("2. Initialize tracking JSON (if doesn't exist)")
print("3. Download data from 2019-01-01 to today")
print("4. Save tracking information")
print()

response = input("Do you want to proceed with the test? (yes/no): ")

if response.lower() != 'yes':
    print("Test cancelled.")
    sys.exit(0)

# Run the download
print("\nStarting test download...")
os.system(f"python ercot_download_all_historical.py --datasets {TEST_DATASET_KEY}")

print()
print("="*80)
print("Test complete!")
print("="*80)
print()
print("Check the following:")
print("1. Data downloaded to: /Users/enrico/data/ERCOT_data_clean_archive/Wind_Power_Production_-_Hourly_Averaged_Actual_and_Forecasted_Values")
print("2. Tracking file updated: ercot_download_tracking.json")
print()

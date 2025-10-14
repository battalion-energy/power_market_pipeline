#!/usr/bin/env python3
"""
Wait for Downloads to Complete

Checks every 10 minutes and exits when all datasets are complete.
This script blocks until downloads finish.

FOR YOUR DAUGHTER'S FUTURE!
"""

import json
import time
import sys
from datetime import datetime
from pathlib import Path

# Expected datasets
EXPECTED_DATASETS = [
    'Wind_Power_Production',
    'Solar_Power_Production',
    'System_Wide_Load',
    'Seven_Day_Load_Forecast',
    'Fuel_Mix',
    'DAM_Hourly_LMPs',
    'RT_5Min_LMPs',
    'AS_Prices',
    'Reserves_Capacity',
    'Wind_Farms',
    'Solar_Farms',
    'Generation_Resources',
    'Load_Resources'
]

STATE_FILE = Path("forecast_download_state.json")

def check_completion():
    """Check if all datasets are complete."""
    if not STATE_FILE.exists():
        return False, 0, len(EXPECTED_DATASETS)

    try:
        with open(STATE_FILE, 'r') as f:
            state = json.load(f)
    except:
        return False, 0, len(EXPECTED_DATASETS)

    if 'datasets' not in state:
        return False, 0, len(EXPECTED_DATASETS)

    datasets = state['datasets']
    completed = sum(1 for name in EXPECTED_DATASETS
                   if name in datasets and 'last_download' in datasets[name])

    return completed == len(EXPECTED_DATASETS), completed, len(EXPECTED_DATASETS)

def main():
    """Wait for all downloads to complete."""
    print("\n" + "="*80)
    print("⏳ WAITING FOR DOWNLOADS TO COMPLETE")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%I:%M %p')}")
    print(f"Expected datasets: {len(EXPECTED_DATASETS)}")
    print(f"Check interval: 10 minutes")
    print("="*80 + "\n")
    sys.stdout.flush()

    check_num = 0

    while True:
        check_num += 1
        now = datetime.now()

        complete, num_done, total = check_completion()

        progress_pct = (num_done / total) * 100
        bar = "█" * int(progress_pct / 5) + "░" * (20 - int(progress_pct / 5))

        print(f"\n[{now.strftime('%I:%M %p')}] Check #{check_num}: [{bar}] {num_done}/{total} ({progress_pct:.0f}%)")
        sys.stdout.flush()

        if complete:
            print("\n" + "="*80)
            print("✅ ALL DOWNLOADS COMPLETE!")
            print("="*80)
            print(f"Completed at: {now.strftime('%I:%M %p')}")
            print(f"Total checks: {check_num}")
            print("\n🚀 Ready to proceed with model training!")
            print("="*80 + "\n")
            sys.stdout.flush()
            return 0

        if check_num == 1:
            print(f"Downloads in progress... checking every 10 minutes")

        sys.stdout.flush()
        time.sleep(600)  # 10 minutes

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted")
        sys.exit(1)

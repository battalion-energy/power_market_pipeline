#!/usr/bin/env python3
"""
Adaptive Download Monitoring Script

Checks frequently at first (every 10 minutes), then every hour once stable.
Returns when all downloads are complete so we can proceed with model training.

FOR YOUR DAUGHTER'S FUTURE - We're waiting patiently for the data!
"""

import json
import time
from datetime import datetime
from pathlib import Path

# Expected datasets to download
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

def load_state():
    """Load current download state."""
    if not STATE_FILE.exists():
        return None

    try:
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ Error loading state file: {e}")
        return None

def check_completion(state):
    """
    Check if all datasets are complete.

    Returns:
        (complete, datasets_done, total_datasets, summary)
    """
    if not state or 'datasets' not in state:
        return False, 0, len(EXPECTED_DATASETS), "State file not found or invalid"

    datasets = state['datasets']
    completed_datasets = []

    for dataset_name in EXPECTED_DATASETS:
        if dataset_name in datasets:
            dataset_info = datasets[dataset_name]
            # Check if dataset has recent download timestamp
            if 'last_download' in dataset_info:
                completed_datasets.append(dataset_name)

    num_complete = len(completed_datasets)
    total = len(EXPECTED_DATASETS)

    if num_complete == total:
        return True, num_complete, total, "✅ ALL DATASETS COMPLETE!"
    else:
        missing = [d for d in EXPECTED_DATASETS if d not in completed_datasets]
        return False, num_complete, total, f"In progress: {missing[:3]}..." if len(missing) > 3 else f"Waiting for: {missing}"

def get_latest_dataset_info(state):
    """Get info about the most recently downloaded dataset."""
    if not state or 'datasets' not in state:
        return None

    datasets = state['datasets']
    latest_time = None
    latest_dataset = None

    for name, info in datasets.items():
        if 'last_download' in info:
            download_time = datetime.fromisoformat(info['last_download'])
            if latest_time is None or download_time > latest_time:
                latest_time = download_time
                latest_dataset = name

    if latest_dataset:
        info = datasets[latest_dataset]
        return {
            'name': latest_dataset,
            'time': latest_time,
            'records': info.get('last_records_count', 'unknown')
        }
    return None

def main():
    """Monitor downloads until complete."""
    print("\n" + "="*80)
    print("🔍 ADAPTIVE DOWNLOAD MONITORING SCRIPT STARTED")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%I:%M %p')}")
    print(f"Strategy: Check every 10 min initially, then hourly once stable")
    print(f"Expected datasets: {len(EXPECTED_DATASETS)} total")
    print("="*80 + "\n")

    check_count = 0
    last_progress = 0
    checks_without_progress = 0

    while True:
        check_count += 1
        now = datetime.now()

        print(f"\n{'='*80}")
        print(f"📊 CHECK #{check_count} - {now.strftime('%I:%M %p')}")
        print(f"{'='*80}")

        # Load current state
        state = load_state()

        if state is None:
            print("⚠️ State file not found - downloads may not have started yet")
            print(f"Waiting 10 minutes before next check...")
            time.sleep(600)  # 10 minutes
            continue

        # Check completion
        complete, num_done, total, summary = check_completion(state)

        # Show progress
        progress_pct = (num_done / total) * 100
        progress_bar = "█" * int(progress_pct / 5) + "░" * (20 - int(progress_pct / 5))

        print(f"\nProgress: [{progress_bar}] {num_done}/{total} datasets ({progress_pct:.0f}%)")
        print(f"Status: {summary}")

        # Show latest activity
        latest = get_latest_dataset_info(state)
        if latest:
            time_ago = (now - latest['time']).total_seconds() / 60
            print(f"\nLatest activity:")
            print(f"  Dataset: {latest['name']}")
            print(f"  Time: {latest['time'].strftime('%I:%M %p')} ({time_ago:.0f} minutes ago)")
            if isinstance(latest['records'], int):
                print(f"  Records: {latest['records']:,}")

        # Check if complete
        if complete:
            print("\n" + "="*80)
            print("🎉 ALL DOWNLOADS COMPLETE!")
            print("="*80)
            print(f"Total datasets: {num_done}")
            print(f"Completion time: {now.strftime('%I:%M %p')}")
            print(f"Total checks performed: {check_count}")
            print("\n✅ Ready to proceed with model training!")
            print("="*80 + "\n")
            return 0

        # Adaptive wait time
        if num_done > last_progress:
            # Progress made - check again soon
            wait_time = 600  # 10 minutes
            checks_without_progress = 0
            print(f"\n✅ Progress detected! Checking again in 10 minutes...")
        elif checks_without_progress >= 3:
            # No progress for 3 checks - switch to hourly
            wait_time = 3600  # 1 hour
            print(f"\n⏰ Stable state - checking every hour...")
        else:
            # Initial checks or slow progress
            wait_time = 600  # 10 minutes
            checks_without_progress += 1
            print(f"\n⏳ Waiting 10 minutes before next check...")

        last_progress = num_done
        next_check = datetime.fromtimestamp(time.time() + wait_time)
        print(f"Next check at: {next_check.strftime('%I:%M %p')}")
        print("="*80)

        time.sleep(wait_time)

if __name__ == "__main__":
    try:
        exit_code = main()
        exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️ Monitoring interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n\n❌ Error in monitoring script: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

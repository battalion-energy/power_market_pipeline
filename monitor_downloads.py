#!/usr/bin/env python3
"""
Download Monitoring Script

Checks every hour if all 13 ERCOT datasets have completed downloading.
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
    print("🔍 DOWNLOAD MONITORING SCRIPT STARTED")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%I:%M %p')}")
    print(f"Checking every hour until all {len(EXPECTED_DATASETS)} datasets are complete")
    print(f"Expected datasets: {', '.join(EXPECTED_DATASETS[:5])}... ({len(EXPECTED_DATASETS)} total)")
    print("="*80 + "\n")

    check_count = 0

    while True:
        check_count += 1
        now = datetime.now()

        print(f"\n{'='*80}")
        print(f"📊 CHECK #{check_count} - {now.strftime('%I:%M %p on %B %d, %Y')}")
        print(f"{'='*80}")

        # Load current state
        state = load_state()

        if state is None:
            print("⚠️ State file not found - downloads may not have started yet")
            print(f"Waiting 1 hour before next check...")
            time.sleep(3600)  # 1 hour
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
            print(f"  Records: {latest['records']:,}" if isinstance(latest['records'], int) else f"  Records: {latest['records']}")

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

        # Wait 1 hour before next check
        print(f"\n⏰ Waiting 1 hour before next check...")
        print(f"Next check at: {(now.replace(hour=now.hour+1, minute=now.minute)).strftime('%I:%M %p')}")
        print("="*80)

        time.sleep(3600)  # 1 hour

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

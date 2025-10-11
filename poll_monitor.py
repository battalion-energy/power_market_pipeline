#!/usr/bin/env python3
"""
Poll the monitoring script and continue when complete.

This script checks the wait_for_downloads.py process every 5 minutes
and exits when downloads are complete.
"""

import subprocess
import time
import sys
from datetime import datetime

MONITOR_SCRIPT = "wait_for_downloads.py"

def is_monitor_running():
    """Check if the monitoring script is still running."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", MONITOR_SCRIPT],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except:
        return False

def main():
    """Poll until monitoring completes."""
    print("\n" + "="*80)
    print("🔄 POLLING DOWNLOAD MONITOR")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%I:%M %p')}")
    print(f"Monitoring script: {MONITOR_SCRIPT}")
    print(f"Poll interval: 5 minutes")
    print("="*80 + "\n")
    sys.stdout.flush()

    poll_count = 0

    while True:
        poll_count += 1
        now = datetime.now()

        if not is_monitor_running():
            print(f"\n[{now.strftime('%I:%M %p')}] Monitor completed!")
            print(f"Total polls: {poll_count}")
            print("\n✅ Downloads finished - ready to continue!")
            sys.stdout.flush()
            return 0

        if poll_count == 1:
            print(f"[{now.strftime('%I:%M %p')}] Monitor is running... will check every 5 minutes")
        else:
            print(f"[{now.strftime('%I:%M %p')}] Poll #{poll_count}: Monitor still running...")

        sys.stdout.flush()
        time.sleep(300)  # 5 minutes

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted")
        sys.exit(1)

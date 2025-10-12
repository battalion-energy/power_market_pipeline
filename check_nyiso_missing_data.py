#!/usr/bin/env python3
"""
Check for missing NYISO PAL data in alternative locations.
"""

import requests
from datetime import datetime, timedelta

# Dates to check
dates = [
    '2021-03-31',  # Day before gap
    '2021-04-01',
    '2021-04-02',
    '2021-04-03',
    '2021-04-04',
    '2021-04-05',
    '2021-04-06',
    '2021-04-07',  # Day after gap
]

print("Checking for NYISO PAL (load) data...\n")

# Check standard daily files
print("=" * 80)
print("CHECKING: Standard daily PAL files")
print("=" * 80)
for date_str in dates:
    date = datetime.strptime(date_str, '%Y-%m-%d')
    date_fmt = date.strftime('%Y%m%d')
    url = f"http://mis.nyiso.com/public/csv/pal/{date_fmt}pal_csv.zip"

    try:
        response = requests.head(url, timeout=5)
        status = "✓ EXISTS" if response.status_code == 200 else f"✗ {response.status_code}"
    except Exception as e:
        status = f"✗ ERROR: {e}"

    print(f"{date_str}: {status} ({url})")

# Check if there might be monthly archives
print("\n" + "=" * 80)
print("CHECKING: Monthly archive (if exists)")
print("=" * 80)
monthly_url = "http://mis.nyiso.com/public/csv/pal/202104pal_csv.zip"
try:
    response = requests.head(monthly_url, timeout=5)
    print(f"Monthly archive: {'✓ EXISTS' if response.status_code == 200 else f'✗ {response.status_code}'} ({monthly_url})")
except Exception as e:
    print(f"Monthly archive: ✗ ERROR: {e}")

# Check NYISO's OASIS system
print("\n" + "=" * 80)
print("CHECKING: Alternative data sources")
print("=" * 80)

# Try integrated real-time actual load
alt_url = "http://mis.nyiso.com/public/csv/pal/202104IntegratedRealTimeActual.csv"
try:
    response = requests.head(alt_url, timeout=5)
    print(f"Integrated RT Actual: {'✓ EXISTS' if response.status_code == 200 else f'✗ {response.status_code}'} ({alt_url})")
except Exception as e:
    print(f"Integrated RT Actual: ✗ ERROR: {e}")

# Try the archive directory
print("\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)
print("If data doesn't exist in daily files, possible alternatives:")
print("1. Check NYISO OASIS system: http://mis.nyiso.com/public/")
print("2. Contact NYISO directly for historical data recovery")
print("3. Use interpolation from surrounding dates (April 31 and April 7)")
print("4. Document as known data gap in dataset")

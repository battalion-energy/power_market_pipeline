#!/usr/bin/env python3
"""Test gridstatus library to understand data structure."""

from datetime import datetime, timedelta
import gridstatus

# Test NYISO
print("Testing NYISO...")
nyiso = gridstatus.NYISO()

# Get data for one day
start = datetime(2024, 1, 1)
end = datetime(2024, 1, 2)

print(f"\n1. Testing LMP data for {start.date()}:")
try:
    lmp_data = nyiso.get_lmp(date=start, market="DAY_AHEAD_HOURLY", locations="ALL")
    print(f"   Columns: {list(lmp_data.columns)}")
    print(f"   Rows: {len(lmp_data)}")
    print(f"   Sample:\n{lmp_data.head()}")
except Exception as e:
    print(f"   Error: {e}")

print(f"\n2. Testing Real-Time LMP:")
try:
    lmp_rt = nyiso.get_lmp(date=start, market="REAL_TIME_5_MIN", locations="ALL")
    print(f"   Columns: {list(lmp_rt.columns)}")
    print(f"   Rows: {len(lmp_rt)}")
except Exception as e:
    print(f"   Error: {e}")

print(f"\n3. Testing Load data:")
try:
    load = nyiso.get_load(start=start, end=end)
    print(f"   Columns: {list(load.columns)}")
    print(f"   Rows: {len(load)}")
except Exception as e:
    print(f"   Error: {e}")

print(f"\n4. Available methods:")
print([m for m in dir(nyiso) if not m.startswith('_') and callable(getattr(nyiso, m))])

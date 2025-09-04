#!/usr/bin/env python3
"""
Test the improved BESS mapping logic on specific problematic entries
"""

import pandas as pd
import sys
sys.path.append('/home/enrico/projects/power_market_pipeline')

# Import the improved functions
from create_bess_mapping_with_coordinates_v2 import determine_load_zone

# Test cases
test_cases = [
    {"name": "CROSSETT_BES1", "lat": 29.77682, "lon": -95.2556, "expected": "LZ_HOUSTON", "current": "LZ_WEST"},
    {"name": "CROSSETT_BES2", "lat": 29.77682, "lon": -95.2556, "expected": "LZ_HOUSTON", "current": "LZ_WEST"},
    {"name": "West Texas Example", "lat": 31.867, "lon": -102.541, "expected": "LZ_WEST", "current": "LZ_WEST"},
    {"name": "Dallas Example", "lat": 32.777, "lon": -96.797, "expected": "LZ_NORTH", "current": "LZ_NORTH"},
    {"name": "Austin Example", "lat": 30.267, "lon": -97.743, "expected": "LZ_SOUTH", "current": "LZ_SOUTH"},
]

print("Testing Load Zone Determination Logic")
print("=" * 60)

for test in test_cases:
    calculated_zone = determine_load_zone(test["lat"], test["lon"])
    status = "✅" if calculated_zone == test["expected"] else "❌"
    
    print(f"\n{test['name']}:")
    print(f"  Location: ({test['lat']:.4f}, {test['lon']:.4f})")
    print(f"  Current Zone: {test['current']}")
    print(f"  Expected Zone: {test['expected']}")
    print(f"  Calculated Zone: {calculated_zone} {status}")
    
    if calculated_zone != test["expected"]:
        print(f"  ⚠️  MISMATCH: Should be {test['expected']} but got {calculated_zone}")

# Now let's check the actual CSV data for Houston area BESS
print("\n" + "=" * 60)
print("Checking Houston Area BESS in Current Data")
print("=" * 60)

# Load current data
df = pd.read_csv('/home/enrico/projects/power_market_pipeline/BESS_COMPREHENSIVE_WITH_COORDINATES.csv')

# Find all BESS with coordinates in Houston area (lat 28.5-30.5, lon -96 to -94.5)
houston_area = df[
    (df['Latitude'].between(28.5, 30.5, inclusive='both')) &
    (df['Longitude'].between(-96.0, -94.5, inclusive='both'))
]

if not houston_area.empty:
    print(f"\nFound {len(houston_area)} BESS in Houston geographic area:")
    
    # Check their assigned zones
    zone_counts = houston_area['Load_Zone'].value_counts()
    print("\nCurrent zone assignments for Houston-area BESS:")
    for zone, count in zone_counts.items():
        print(f"  {zone}: {count}")
    
    # Show mismatched ones
    mismatched = houston_area[houston_area['Load_Zone'] != 'LZ_HOUSTON']
    if not mismatched.empty:
        print(f"\n⚠️  {len(mismatched)} BESS in Houston area with wrong zone:")
        for _, row in mismatched.head(10).iterrows():
            print(f"  - {row['BESS_Gen_Resource']}: Zone={row['Load_Zone']}, Lat={row['Latitude']:.4f}, Lon={row['Longitude']:.4f}")
else:
    print("\nNo BESS found in Houston geographic area")

# Check West Texas BESS that might be misplaced
print("\n" + "=" * 60)
print("Checking LZ_WEST BESS for Misplacements")
print("=" * 60)

west_zone = df[df['Load_Zone'] == 'LZ_WEST']
# Check if any are actually in Houston area
misplaced_west = west_zone[
    (west_zone['Latitude'].between(28.5, 30.5, inclusive='both')) &
    (west_zone['Longitude'].between(-96.0, -94.5, inclusive='both'))
]

if not misplaced_west.empty:
    print(f"\n❌ Found {len(misplaced_west)} LZ_WEST BESS that are actually in Houston area:")
    for _, row in misplaced_west.iterrows():
        print(f"  - {row['BESS_Gen_Resource']}: Lat={row['Latitude']:.4f}, Lon={row['Longitude']:.4f}")
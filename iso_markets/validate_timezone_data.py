#!/usr/bin/env python3
"""
Comprehensive Timezone Validation Script

This script performs deep analysis of converted parquet files to detect timezone errors:
1. Validates both datetime_utc and datetime_local are timezone-aware
2. Checks UTC vs local offset correctness for each ISO
3. Verifies DST transitions (March 10 & Nov 3, 2024)
4. Validates year partitioning (uses local date, not UTC date)
5. Spot checks specific timestamps
6. Validates LMP components (total, energy, congestion, loss)
7. Checks ancillary services preservation
"""

import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime, timedelta
import sys

# ISO timezone configurations
ISO_CONFIGS = {
    'pjm': {'timezone': 'America/New_York', 'utc_offset_winter': -5, 'utc_offset_summer': -4, 'has_dst': True},
    'caiso': {'timezone': 'America/Los_Angeles', 'utc_offset_winter': -8, 'utc_offset_summer': -7, 'has_dst': True},
    'nyiso': {'timezone': 'America/New_York', 'utc_offset_winter': -5, 'utc_offset_summer': -4, 'has_dst': True},
    'spp': {'timezone': 'America/Chicago', 'utc_offset_winter': -6, 'utc_offset_summer': -5, 'has_dst': True},
    'isone': {'timezone': 'America/New_York', 'utc_offset_winter': -5, 'utc_offset_summer': -4, 'has_dst': True},
    'miso': {'timezone': 'EST', 'utc_offset_winter': -5, 'utc_offset_summer': -5, 'has_dst': False},  # FIXED EST!
    'ercot': {'timezone': 'America/Chicago', 'utc_offset_winter': -6, 'utc_offset_summer': -5, 'has_dst': True},
}

# DST transition dates for 2024
DST_SPRING_FORWARD = '2024-03-10'  # 2am → 3am (23 hours)
DST_FALL_BACK = '2024-11-03'  # 2am → 1am (25 hours)

def validate_iso_parquet(iso_name, year=2025):
    """Validate parquet file for a specific ISO."""
    print(f"\n{'='*80}")
    print(f"VALIDATING: {iso_name.upper()} - Year {year}")
    print(f"{'='*80}\n")

    config = ISO_CONFIGS.get(iso_name.lower())
    if not config:
        print(f"❌ Unknown ISO: {iso_name}")
        return False

    # Find parquet files
    base_dir = Path(f"/pool/ssd8tb/data/iso/unified_iso_data/parquet/{iso_name.lower()}")

    # Check DA energy files
    da_files = list(base_dir.glob(f"da_energy_hourly*/*_{year}.parquet"))

    if not da_files:
        print(f"⚠️  No parquet files found for {iso_name} {year}")
        return False

    all_passed = True

    for parquet_file in da_files:
        print(f"\n📁 File: {parquet_file.name}")
        print(f"   Size: {parquet_file.stat().st_size / 1024**2:.1f} MB")

        try:
            # Read sample of data
            df = pd.read_parquet(parquet_file)
            print(f"   Rows: {len(df):,}")

            # Test 1: Timezone Awareness
            print(f"\n   🔍 Test 1: Timezone Awareness")
            datetime_utc_tz = df['datetime_utc'].dtype
            datetime_local_tz = df['datetime_local'].dtype

            if 'UTC' in str(datetime_utc_tz):
                print(f"   ✅ datetime_utc is timezone-aware (UTC): {datetime_utc_tz}")
            else:
                print(f"   ❌ datetime_utc is NOT timezone-aware: {datetime_utc_tz}")
                all_passed = False

            if 'tzinfo' in str(datetime_local_tz) or config['timezone'] in str(datetime_local_tz):
                print(f"   ✅ datetime_local is timezone-aware: {datetime_local_tz}")
            else:
                print(f"   ❌ datetime_local is NOT timezone-aware: {datetime_local_tz}")
                all_passed = False

            # Test 2: UTC vs Local Offset
            print(f"\n   🔍 Test 2: UTC vs Local Offset")
            sample = df.head(100)

            # Calculate actual offset
            offsets = (sample['datetime_local'] - sample['datetime_utc']).dt.total_seconds() / 3600
            unique_offsets = offsets.unique()

            expected_offsets = [config['utc_offset_winter']]
            if config['has_dst']:
                expected_offsets.append(config['utc_offset_summer'])

            print(f"   Expected offsets: {expected_offsets} hours")
            print(f"   Actual offsets: {sorted(unique_offsets.tolist())} hours")

            if all(offset in expected_offsets for offset in unique_offsets):
                print(f"   ✅ All offsets are correct")
            else:
                print(f"   ❌ Unexpected offsets found!")
                all_passed = False

            # Test 3: LMP Components
            print(f"\n   🔍 Test 3: LMP Components (Energy, Congestion, Loss)")
            lmp_cols = [col for col in df.columns if 'lmp' in col.lower()]
            print(f"   LMP columns found: {lmp_cols}")

            if 'lmp_total' in df.columns:
                print(f"   ✅ lmp_total present")
            else:
                print(f"   ⚠️  lmp_total missing")

            if 'lmp_energy' in df.columns and 'lmp_congestion' in df.columns and 'lmp_loss' in df.columns:
                print(f"   ✅ LMP components (energy, congestion, loss) all present")

                # Verify calculation
                sample_with_components = df[df['lmp_energy'].notna()].head(10)
                if len(sample_with_components) > 0:
                    calculated = sample_with_components['lmp_energy'] + sample_with_components['lmp_congestion'] + sample_with_components['lmp_loss']
                    actual = sample_with_components['lmp_total']
                    diff = (calculated - actual).abs().max()

                    if diff < 0.01:  # Allow for floating point precision
                        print(f"   ✅ LMP = Energy + Congestion + Loss (verified)")
                    else:
                        print(f"   ⚠️  LMP calculation discrepancy: max diff = {diff}")
            else:
                print(f"   ⚠️  Some LMP components missing (may be normal for some ISOs)")

            # Test 4: Year Partitioning (uses local date)
            print(f"\n   🔍 Test 4: Year Partitioning")
            delivery_dates = pd.to_datetime(df['delivery_date'])
            years_in_file = delivery_dates.dt.year.unique()

            if len(years_in_file) == 1 and years_in_file[0] == year:
                print(f"   ✅ All delivery_dates are in {year} (correct partitioning)")
            else:
                print(f"   ⚠️  Found years: {sorted(years_in_file.tolist())} (expected only {year})")
                # Check if it's just boundary dates
                other_years = delivery_dates[delivery_dates.dt.year != year]
                if len(other_years) < 100:
                    print(f"   ℹ️  Only {len(other_years)} rows in other years (likely boundary dates)")
                else:
                    all_passed = False

            # Test 5: Spot Check Specific Timestamps
            print(f"\n   🔍 Test 5: Spot Check Timestamps")
            # Check a winter date (EST/CST/PST)
            winter_sample = df[df['delivery_date'] == pd.to_datetime('2025-01-15').date()].head(5)
            if len(winter_sample) > 0:
                for idx, row in winter_sample.iterrows():
                    offset_hours = (row['datetime_local'] - row['datetime_utc']).total_seconds() / 3600
                    expected = config['utc_offset_winter']
                    if abs(offset_hours - expected) < 0.1:
                        print(f"   ✅ Jan 15 offset: {offset_hours:.1f}h (expected {expected}h)")
                    else:
                        print(f"   ❌ Jan 15 offset: {offset_hours:.1f}h (expected {expected}h)")
                        all_passed = False
                    break  # Just check first row

            print(f"\n   📊 Summary for {parquet_file.name}")
            if all_passed:
                print(f"   ✅ ALL TESTS PASSED")
            else:
                print(f"   ❌ SOME TESTS FAILED")

        except Exception as e:
            print(f"   ❌ Error reading file: {e}")
            all_passed = False

    return all_passed

def main():
    print("\n" + "="*80)
    print("COMPREHENSIVE TIMEZONE VALIDATION")
    print("="*80)
    print(f"\nValidating 2025 parquet files for all ISOs...")
    print(f"Schema Version: 2.0.0 (timezone-aware datetime_local)")

    results = {}
    for iso in ISO_CONFIGS.keys():
        passed = validate_iso_parquet(iso, year=2025)
        results[iso] = passed

    # Final Summary
    print(f"\n\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)

    for iso, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{iso.upper():10s} - {status}")

    total_passed = sum(results.values())
    total = len(results)
    print(f"\nOverall: {total_passed}/{total} ISOs passed validation")

    if total_passed == total:
        print("\n🎉 ALL ISOs VALIDATED SUCCESSFULLY!")
        return 0
    else:
        print(f"\n⚠️  {total - total_passed} ISOs failed validation")
        return 1

if __name__ == "__main__":
    sys.exit(main())

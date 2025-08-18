#!/usr/bin/env python3
"""
Test date parsing logic for all ERCOT data types.
"""

import pandas as pd
from datetime import datetime, timedelta
import numpy as np

def test_date_format(date_str, expected_format, field_name):
    """Test if a date string matches expected format."""
    try:
        parsed = datetime.strptime(date_str, expected_format)
        return True, parsed, None
    except Exception as e:
        return False, None, str(e)

def main():
    print("=" * 80)
    print("DATE PARSING TEST SUITE")
    print("=" * 80)
    
    # Test cases based on spot-checked files
    test_cases = {
        "DAM_Settlement_Point_Prices": {
            "DeliveryDate": {
                "samples": ["11/05/2013", "05/12/2022", "12/04/2020"],
                "format": "%m/%d/%Y",
                "rust_format": "%m/%d/%Y"
            },
            "HourEnding": {
                "samples": ["01:00", "24:00", "13:00"],
                "format": None,  # Not a date, just hour string
                "rust_format": None
            }
        },
        "DAM_Clearing_Prices_for_Capacity": {
            "DeliveryDate": {
                "samples": ["11/01/2021", "03/08/2015", "02/10/2018"],
                "format": "%m/%d/%Y",
                "rust_format": "%m/%d/%Y"
            },
            "HourEnding": {
                "samples": ["01:00", "24:00", "13:00"],
                "format": None,
                "rust_format": None
            }
        },
        "Settlement_Point_Prices_RT": {
            "DeliveryDate": {
                "samples": ["03/13/2022", "12/31/2020", "07/22/2023"],
                "format": "%m/%d/%Y",
                "rust_format": "%m/%d/%Y"
            },
            "DeliveryHour": {
                "samples": [10, 12, 3],  # Integer values
                "format": None,
                "rust_format": None
            },
            "DeliveryInterval": {
                "samples": [1, 2, 3, 4],  # 15-minute intervals
                "format": None,
                "rust_format": None
            }
        },
        "60-Day_SCED_Gen_Resource": {
            "SCED Time Stamp": {
                "samples": ["07/25/2020 00:00:21", "07/25/2020 23:55:00", "12/31/2020 15:30:00"],
                "format": "%m/%d/%Y %H:%M:%S",
                "rust_format": "%m/%d/%Y %H:%M:%S"
            }
        },
        "60-Day_DAM_Gen_Resource": {
            "Delivery Date": {
                "samples": ["05/13/2018", "12/25/2020", "07/04/2021"],
                "format": "%m/%d/%Y",
                "rust_format": "%m/%d/%Y"
            },
            "Hour Ending": {
                "samples": ["1", "24", "13"],  # String numbers
                "format": None,
                "rust_format": None
            }
        }
    }
    
    # Test each data type
    all_passed = True
    for data_type, fields in test_cases.items():
        print(f"\n{data_type}")
        print("-" * 40)
        
        for field_name, config in fields.items():
            if config["format"]:
                print(f"  {field_name}:")
                print(f"    Expected format: {config['format']}")
                print(f"    Rust format: {config['rust_format']}")
                
                # Test each sample
                for sample in config["samples"]:
                    if isinstance(sample, str):
                        success, parsed, error = test_date_format(sample, config["format"], field_name)
                        if success:
                            print(f"    ✓ '{sample}' -> {parsed}")
                        else:
                            print(f"    ✗ '{sample}' -> ERROR: {error}")
                            all_passed = False
                    else:
                        print(f"    - '{sample}' (not a date string)")
            else:
                print(f"  {field_name}: Not a date field (values: {config['samples']})")
    
    # Now test the Python flattening logic
    print("\n" + "=" * 80)
    print("PYTHON FLATTENING DATE LOGIC TEST")
    print("=" * 80)
    
    # Test DA flattening date logic
    print("\nDA Energy Prices date construction:")
    test_delivery_date = "01/15/2023"
    test_hour_ending = "14:00"
    
    # Simulate the logic from flatten_ercot_prices.py
    delivery_date = pd.to_datetime(test_delivery_date, format="%m/%d/%Y")
    hour = int(test_hour_ending.split(':')[0])
    datetime_result = delivery_date + pd.Timedelta(hours=hour-1)
    print(f"  DeliveryDate: {test_delivery_date}, HourEnding: {test_hour_ending}")
    print(f"  Result: {datetime_result}")
    print(f"  Expected: 2023-01-15 13:00:00")
    if datetime_result == pd.Timestamp("2023-01-15 13:00:00"):
        print("  ✓ PASS")
    else:
        print("  ✗ FAIL")
        all_passed = False
    
    # Test RT 15-minute interval logic
    print("\nRT Prices 15-minute interval construction:")
    test_delivery_date = "07/22/2023"
    test_delivery_hour = 3
    test_delivery_interval = 4  # Should be :45 minutes
    
    delivery_date = pd.to_datetime(test_delivery_date, format="%m/%d/%Y")
    datetime_result = delivery_date + pd.Timedelta(
        minutes=(test_delivery_hour - 1) * 60 + (test_delivery_interval - 1) * 15
    )
    print(f"  DeliveryDate: {test_delivery_date}, Hour: {test_delivery_hour}, Interval: {test_delivery_interval}")
    print(f"  Result: {datetime_result}")
    print(f"  Expected: 2023-07-22 02:45:00")
    if datetime_result == pd.Timestamp("2023-07-22 02:45:00"):
        print("  ✓ PASS")
    else:
        print("  ✗ FAIL")
        all_passed = False
    
    # Test SCED timestamp parsing
    print("\nSCED Timestamp parsing:")
    test_sced = "07/25/2020 00:00:21"
    sced_result = pd.to_datetime(test_sced, format="%m/%d/%Y %H:%M:%S")
    print(f"  Input: {test_sced}")
    print(f"  Result: {sced_result}")
    print(f"  Expected: 2020-07-25 00:00:21")
    if sced_result == pd.Timestamp("2020-07-25 00:00:21"):
        print("  ✓ PASS")
    else:
        print("  ✗ FAIL")
        all_passed = False
    
    # Final summary
    print("\n" + "=" * 80)
    if all_passed:
        print("✅ ALL DATE PARSING TESTS PASSED")
    else:
        print("❌ SOME DATE PARSING TESTS FAILED")
    print("=" * 80)

if __name__ == "__main__":
    main()
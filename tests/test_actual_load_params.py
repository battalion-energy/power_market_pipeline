#!/usr/bin/env python3
"""
Test script to find correct parameter names for Actual Load endpoints.

NP6-345-CD (act_sys_load_by_wzn) returns 400 with deliveryDateFrom/To.
Testing alternative parameter patterns.
"""

import os
import sys
import asyncio
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ercot_ws_downloader.client import ERCOTWebServiceClient

load_dotenv()

async def test_endpoint_params(endpoint: str, param_patterns: list, test_date: str = "2025-10-09"):
    """Test an endpoint with different parameter patterns."""

    client = ERCOTWebServiceClient(
        username=os.getenv("ERCOT_USERNAME"),
        password=os.getenv("ERCOT_PASSWORD"),
        subscription_key=os.getenv("ERCOT_SUBSCRIPTION_KEY"),
    )

    print(f"\n{'='*80}")
    print(f"Testing endpoint: {endpoint}")
    print(f"Test date: {test_date}")
    print(f"{'='*80}\n")

    for pattern_name, params in param_patterns:
        print(f"Testing parameter pattern: {pattern_name}")
        print(f"  Parameters: {params}")

        try:
            response = await client._make_request(
                endpoint=endpoint,
                params=params
            )

            # Response is already JSON (dict or list)
            if isinstance(response, list):
                print(f"  ✅ SUCCESS! Got {len(response)} records")
                print(f"  Sample record: {response[0] if response else 'No data'}")
                return pattern_name, params
            elif isinstance(response, dict):
                if "data" in response:
                    data = response["data"]
                    print(f"  ✅ SUCCESS! Got {len(data)} records")
                    print(f"  Sample record: {data[0] if data else 'No data'}")
                    return pattern_name, params
                else:
                    print(f"  ✅ SUCCESS! Got response: {response}")
                    return pattern_name, params

        except Exception as e:
            error_msg = str(e)
            print(f"  ❌ Error: {error_msg[:300]}")

    print(f"\n❌ No working parameter pattern found for {endpoint}\n")
    return None, None


async def main():
    # Test date range (1 day for quick testing)
    test_date = "2025-10-09"

    # Parameter patterns to test for NP6-345-CD (5-minute actual load data)
    param_patterns_5min = [
        ("deliveryDate", {
            "deliveryDateFrom": test_date,
            "deliveryDateTo": test_date,
        }),
        ("operatingDate", {
            "operatingDateFrom": test_date,
            "operatingDateTo": test_date,
        }),
        ("SCEDTimestamp", {
            "SCEDTimestampFrom": f"{test_date}T00:00",
            "SCEDTimestampTo": f"{test_date}T23:55",
        }),
        ("hourEnding", {
            "hourEndingFrom": test_date,
            "hourEndingTo": test_date,
        }),
        ("deliveryDatetime", {
            "deliveryDatetimeFrom": f"{test_date}T00:00:00",
            "deliveryDatetimeTo": f"{test_date}T23:59:59",
        }),
        ("operatingDatetime", {
            "operatingDatetimeFrom": f"{test_date}T00:00:00",
            "operatingDatetimeTo": f"{test_date}T23:59:59",
        }),
        ("intervalStart", {
            "intervalStartFrom": f"{test_date}T00:00:00",
            "intervalStartTo": f"{test_date}T23:59:59",
        }),
        ("postedDatetime", {
            "postedDatetimeFrom": f"{test_date}T00:00:00",
            "postedDatetimeTo": f"{test_date}T23:59:59",
        }),
    ]

    # Test NP6-345-CD (Actual System Load by Weather Zone - 5-minute)
    print("\n" + "="*80)
    print("TEST 1: NP6-345-CD (Actual System Load by Weather Zone - 5-minute)")
    print("="*80)
    pattern_name, params = await test_endpoint_params(
        "np6-345-cd/act_sys_load_by_wzn",
        param_patterns_5min,
        test_date
    )

    if pattern_name:
        print(f"\n✅ SOLUTION FOUND for NP6-345-CD:")
        print(f"   Pattern: {pattern_name}")
        print(f"   Parameters: {params}")

    # Test NP6-346-CD (Actual System Load by Forecast Zone - hourly)
    # Likely uses same pattern as NP6-345-CD
    print("\n" + "="*80)
    print("TEST 2: NP6-346-CD (Actual System Load by Forecast Zone - hourly)")
    print("="*80)

    if pattern_name:
        print(f"Trying successful pattern from NP6-345-CD: {pattern_name}")
        pattern_name_2, params_2 = await test_endpoint_params(
            "np6-346-cd/act_sys_load_by_fzn",
            [(pattern_name, params)],
            test_date
        )

        if pattern_name_2:
            print(f"\n✅ SOLUTION CONFIRMED for NP6-346-CD (same pattern works)")
        else:
            print(f"\n⚠️ Same pattern doesn't work for NP6-346-CD, testing all patterns...")
            pattern_name_2, params_2 = await test_endpoint_params(
                "np6-346-cd/act_sys_load_by_fzn",
                param_patterns_5min,
                test_date
            )
    else:
        pattern_name_2, params_2 = await test_endpoint_params(
            "np6-346-cd/act_sys_load_by_fzn",
            param_patterns_5min,
            test_date
        )

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"NP6-345-CD (Weather Zone): {'✅ ' + pattern_name if pattern_name else '❌ Not found'}")
    print(f"NP6-346-CD (Forecast Zone): {'✅ ' + pattern_name_2 if pattern_name_2 else '❌ Not found'}")
    print()


if __name__ == "__main__":
    asyncio.run(main())

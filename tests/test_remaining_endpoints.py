#!/usr/bin/env python3
"""
Test the remaining 4 endpoints: Fuel Mix, Outages, DAM Lambda, System Demand.
"""

import os
import asyncio
import httpx
from dotenv import load_dotenv

load_dotenv()

async def authenticate():
    """Authenticate with ERCOT and get access token."""
    AUTH_URL = "https://ercotb2c.b2clogin.com/ercotb2c.onmicrosoft.com/B2C_1_PUBAPI-ROPC-FLOW/oauth2/v2.0/token"
    CLIENT_ID = "fec253ea-0d06-4272-a5e6-b478baeecd70"

    params = {
        "username": os.getenv("ERCOT_USERNAME"),
        "password": os.getenv("ERCOT_PASSWORD"),
        "grant_type": "password",
        "scope": f"openid {CLIENT_ID} offline_access",
        "client_id": CLIENT_ID,
        "response_type": "token",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(AUTH_URL, data=params)
        response.raise_for_status()
        return response.json()


async def test_endpoint_with_schema(endpoint, test_params_list, token_data, subscription_key):
    """Test an endpoint with multiple parameter patterns, showing schema first."""
    BASE_URL = "https://api.ercot.com/api/public-reports"

    headers = {
        "Authorization": f"Bearer {token_data['access_token']}",
        "Ocp-Apim-Subscription-Key": subscription_key,
        "Accept": "application/json",
    }

    url = f"{BASE_URL}/{endpoint}"

    print(f"\n{'='*80}")
    print(f"Testing: {endpoint}")
    print(f"{'='*80}")

    # First, get schema with no parameters
    print(f"\nStep 1: Getting schema (no parameters)...")
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            response = await client.get(url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Endpoint exists!")

                if "fields" in data:
                    print(f"\nAvailable fields:")
                    for field in data["fields"][:10]:  # Show first 10 fields
                        has_range = field.get("hasRange", False)
                        if has_range:
                            print(f"  - {field['name']} ({field['dataType']}) - HAS RANGE (can use From/To)")

                    # Look for date/time fields
                    date_fields = [f for f in data["fields"] if "date" in f["name"].lower() or "time" in f["name"].lower()]
                    if date_fields:
                        print(f"\nDate/Time fields found:")
                        for field in date_fields:
                            print(f"  - {field['name']} (hasRange: {field.get('hasRange', False)})")
            else:
                print(f"❌ Endpoint not found: {response.status_code}")
                return None

        except Exception as e:
            print(f"❌ Error getting schema: {str(e)[:200]}")
            return None

    # Now test with parameters
    print(f"\nStep 2: Testing with parameters...")
    for pattern_name, params in test_params_list:
        print(f"\n  Pattern: {pattern_name}")
        print(f"  Params: {params}")

        async with httpx.AsyncClient(timeout=15.0) as client:
            try:
                response = await client.get(url, params=params, headers=headers)

                if response.status_code == 200:
                    data = response.json()
                    print(f"  ✅ SUCCESS!")

                    if isinstance(data, dict) and "data" in data:
                        records = data["data"]
                        print(f"  Records: {len(records)}")
                        if records:
                            print(f"  Sample: {records[0][:3] if isinstance(records[0], list) else str(records[0])[:100]}")
                        return pattern_name, params
                    elif isinstance(data, list):
                        print(f"  Records: {len(data)}")
                        return pattern_name, params
                else:
                    error = response.text[:150]
                    print(f"  ❌ HTTP {response.status_code}: {error}")

            except Exception as e:
                print(f"  ❌ Error: {str(e)[:150]}")

        await asyncio.sleep(0.3)

    return None


async def main():
    print("\n" + "="*80)
    print("TESTING REMAINING 4 ERCOT ENDPOINTS")
    print("="*80)

    # Authenticate
    print("\nAuthenticating...")
    token_data = await authenticate()
    print("✅ Authenticated\n")

    subscription_key = os.getenv("ERCOT_SUBSCRIPTION_KEY")
    test_date = "2025-10-09"

    results = {}

    # Test 1: Fuel Mix (alternative endpoint)
    print("\n" + "="*80)
    print("TEST 1: FUEL MIX (Alternative: 2-Day Generation Summary)")
    print("="*80)

    fuel_mix_params = [
        ("deliveryDate", {
            "deliveryDateFrom": test_date,
            "deliveryDateTo": test_date,
        }),
        ("publishDate", {
            "publishDateFrom": test_date,
            "publishDateTo": test_date,
        }),
        ("operatingDate", {
            "operatingDateFrom": test_date,
            "operatingDateTo": test_date,
        }),
    ]

    result = await test_endpoint_with_schema(
        "np3-910-er/2d_agg_gen_summary",
        fuel_mix_params,
        token_data,
        subscription_key
    )
    results["fuel_mix"] = result

    await asyncio.sleep(1)

    # Test 2: Unplanned Outages (alternative endpoint)
    print("\n" + "="*80)
    print("TEST 2: UNPLANNED OUTAGES (Alternative: NP1-346-ER)")
    print("="*80)

    outages_params = [
        ("publishDatetime", {
            "publishDatetimeFrom": f"{test_date}T00:00:00",
            "publishDatetimeTo": f"{test_date}T23:59:59",
        }),
        ("deliveryDate", {
            "deliveryDateFrom": test_date,
            "deliveryDateTo": test_date,
        }),
        ("postedDatetime", {
            "postedDatetimeFrom": f"{test_date}T00:00:00",
            "postedDatetimeTo": f"{test_date}T23:59:59",
        }),
    ]

    result = await test_endpoint_with_schema(
        "np1-346-er/unpl_res_outages",
        outages_params,
        token_data,
        subscription_key
    )
    results["outages"] = result

    await asyncio.sleep(1)

    # Test 3: System Wide Demand (check if np6-322-cd exists at all)
    print("\n" + "="*80)
    print("TEST 3: SYSTEM WIDE DEMAND (Original endpoint)")
    print("="*80)

    demand_params = [
        ("operatingDay", {
            "operatingDayFrom": test_date,
            "operatingDayTo": test_date,
        }),
        ("deliveryDate", {
            "deliveryDateFrom": test_date,
            "deliveryDateTo": test_date,
        }),
        ("SCEDTimestamp", {
            "SCEDTimestampFrom": f"{test_date}T00:00",
            "SCEDTimestampTo": f"{test_date}T23:55",
        }),
    ]

    result = await test_endpoint_with_schema(
        "np6-322-cd/act_sys_load_5_min",
        demand_params,
        token_data,
        subscription_key
    )
    results["system_demand"] = result

    await asyncio.sleep(1)

    # Test 4: DAM System Lambda (search for alternatives)
    print("\n" + "="*80)
    print("TEST 4: DAM SYSTEM LAMBDA (Searching alternatives)")
    print("="*80)

    lambda_params = [
        ("deliveryDate", {
            "deliveryDateFrom": test_date,
            "deliveryDateTo": test_date,
        }),
        ("operatingDay", {
            "operatingDayFrom": test_date,
            "operatingDayTo": test_date,
        }),
    ]

    # Try multiple possible report codes
    lambda_endpoints = [
        "np4-523-cd/dam_sys_lambda",
        "np4-191-cd/dam_sys_lambda",
        "np4-188-cd/dam_sys_lambda",  # Same as AS prices
        "np6-787-cd/dam_sys_lambda",
    ]

    for endpoint in lambda_endpoints:
        print(f"\n  Trying: {endpoint}")
        result = await test_endpoint_with_schema(
            endpoint,
            lambda_params,
            token_data,
            subscription_key
        )
        if result:
            results["dam_lambda"] = result
            break
        await asyncio.sleep(0.5)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    for name, result in results.items():
        if result:
            pattern, params = result
            print(f"\n✅ {name.upper()}: WORKING")
            print(f"   Pattern: {pattern}")
            print(f"   Params: {params}")
        else:
            print(f"\n❌ {name.upper()}: NOT FOUND")


if __name__ == "__main__":
    asyncio.run(main())

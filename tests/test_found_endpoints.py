#!/usr/bin/env python3
"""
Test the endpoints found in the catalog.
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


async def test_endpoint(endpoint, params, token_data, subscription_key):
    """Test an endpoint with parameters."""
    BASE_URL = "https://api.ercot.com/api/public-reports"

    headers = {
        "Authorization": f"Bearer {token_data['access_token']}",
        "Ocp-Apim-Subscription-Key": subscription_key,
        "Accept": "application/json",
    }

    url = f"{BASE_URL}/{endpoint}"

    print(f"\nTesting: {endpoint}")
    print(f"Params: {params}")

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(url, params=params, headers=headers)

            if response.status_code == 200:
                data = response.json()
                print(f"✅ SUCCESS!")

                if isinstance(data, dict) and "data" in data:
                    records = data["data"]
                    print(f"Records: {len(records)}")
                    if records:
                        sample = records[0]
                        if isinstance(sample, list):
                            print(f"Sample: {sample[:5]}...")
                        else:
                            print(f"Sample keys: {list(sample.keys())[:10]}")
                elif isinstance(data, list):
                    print(f"Records: {len(data)}")

                return True
            else:
                print(f"❌ HTTP {response.status_code}: {response.text[:200]}")
                return False

        except Exception as e:
            print(f"❌ Error: {str(e)[:200]}")
            return False


async def main():
    print("\n" + "="*80)
    print("Testing endpoints found in catalog")
    print("="*80)

    # Authenticate
    print("\nAuthenticating...")
    token_data = await authenticate()
    print("✅ Authenticated")

    subscription_key = os.getenv("ERCOT_SUBSCRIPTION_KEY")
    test_date = "2025-10-09"

    results = {}

    # Test 1: Hourly Resource Outage Capacity
    print("\n" + "="*80)
    print("TEST 1: HOURLY RESOURCE OUTAGE CAPACITY")
    print("="*80)

    outage_params = [
        ("deliveryDate", {
            "deliveryDateFrom": test_date,
            "deliveryDateTo": test_date,
        }),
        ("operatingDay", {
            "operatingDayFrom": test_date,
            "operatingDayTo": test_date,
        }),
        ("hourEnding", {
            "hourEndingFrom": test_date,
            "hourEndingTo": test_date,
        }),
    ]

    for pattern_name, params in outage_params:
        print(f"\nPattern: {pattern_name}")
        success = await test_endpoint(
            "np3-233-cd/hourly_res_outage_cap",
            params,
            token_data,
            subscription_key
        )
        if success:
            results["outages"] = (pattern_name, params)
            break
        await asyncio.sleep(0.5)

    # Test 2: DAM System Lambda (CORRECT ENDPOINT NAME!)
    print("\n" + "="*80)
    print("TEST 2: DAM SYSTEM LAMBDA (dam_system_lambda, not dam_sys_lambda!)")
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

    for pattern_name, params in lambda_params:
        print(f"\nPattern: {pattern_name}")
        success = await test_endpoint(
            "np4-523-cd/dam_system_lambda",  # FULL WORD "system"!
            params,
            token_data,
            subscription_key
        )
        if success:
            results["dam_lambda"] = (pattern_name, params)
            break
        await asyncio.sleep(0.5)

    # Test 3: SCED System Lambda
    print("\n" + "="*80)
    print("TEST 3: SCED SYSTEM LAMBDA")
    print("="*80)

    sced_params = [
        ("SCEDTimestamp", {
            "SCEDTimestampFrom": f"{test_date}T00:00",
            "SCEDTimestampTo": f"{test_date}T23:55",
        }),
        ("operatingDay", {
            "operatingDayFrom": test_date,
            "operatingDayTo": test_date,
        }),
    ]

    for pattern_name, params in sced_params:
        print(f"\nPattern: {pattern_name}")
        success = await test_endpoint(
            "np6-322-cd/sced_system_lambda",
            params,
            token_data,
            subscription_key
        )
        if success:
            results["sced_lambda"] = (pattern_name, params)
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


if __name__ == "__main__":
    asyncio.run(main())

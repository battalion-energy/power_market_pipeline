#!/usr/bin/env python3
"""
Test script to find correct parameter names for Actual Load endpoints (no retries).
"""

import os
import asyncio
import httpx
from datetime import datetime
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
    """Test a single endpoint with parameters (no retries)."""
    BASE_URL = "https://api.ercot.com/api/public-reports"

    headers = {
        "Authorization": f"Bearer {token_data['access_token']}",
        "Ocp-Apim-Subscription-Key": subscription_key,
        "Accept": "application/json",
    }

    url = f"{BASE_URL}/{endpoint}"

    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            response = await client.get(url, params=params, headers=headers)

            if response.status_code == 200:
                return True, response.json()
            else:
                return False, f"HTTP {response.status_code}: {response.text[:200]}"

        except Exception as e:
            return False, str(e)[:200]


async def main():
    print("\n" + "="*80)
    print("Testing NP6-345-CD (Actual System Load by Weather Zone)")
    print("="*80)

    # Authenticate
    print("\nAuthenticating...")
    token_data = await authenticate()
    print("✅ Authenticated")

    subscription_key = os.getenv("ERCOT_SUBSCRIPTION_KEY")
    test_date = "2025-10-09"

    # All parameter patterns to test
    param_patterns = [
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
        ("publishDate", {
            "publishDateFrom": test_date,
            "publishDateTo": test_date,
        }),
        ("publishDatetime", {
            "publishDatetimeFrom": f"{test_date}T00:00:00",
            "publishDatetimeTo": f"{test_date}T23:59:59",
        }),
        ("postingDate", {
            "postingDateFrom": test_date,
            "postingDateTo": test_date,
        }),
        ("postingDatetime", {
            "postingDatetimeFrom": f"{test_date}T00:00:00",
            "postingDatetimeTo": f"{test_date}T23:59:59",
        }),
    ]

    endpoint = "np6-345-cd/act_sys_load_by_wzn"

    print(f"\nTesting {len(param_patterns)} parameter patterns...\n")

    for pattern_name, params in param_patterns:
        print(f"Testing: {pattern_name}")
        print(f"  Params: {params}")

        success, result = await test_endpoint(endpoint, params, token_data, subscription_key)

        if success:
            print(f"  ✅ SUCCESS!")
            if isinstance(result, dict) and "data" in result:
                print(f"  Got {len(result['data'])} records")
                if result['data']:
                    print(f"  Sample: {result['data'][0]}")
            print(f"\n{'='*80}")
            print(f"SOLUTION FOUND: {pattern_name}")
            print(f"Parameters: {params}")
            print(f"{'='*80}\n")
            return
        else:
            print(f"  ❌ {result}")

        # Small delay between tests
        await asyncio.sleep(0.5)

    print(f"\n{'='*80}")
    print("❌ No working parameter pattern found")
    print("{'='*80}\n")


if __name__ == "__main__":
    asyncio.run(main())

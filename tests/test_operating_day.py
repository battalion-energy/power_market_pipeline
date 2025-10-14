#!/usr/bin/env python3
"""
Test with operatingDay parameter (singular, as shown in the schema).
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

    print(f"Testing: {url}")
    print(f"Params: {params}\n")

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(url, params=params, headers=headers)

            print(f"Status Code: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                print(f"✅ SUCCESS!")
                print(f"Response type: {type(data)}")

                if isinstance(data, dict):
                    print(f"Response keys: {list(data.keys())}")
                    if "data" in data:
                        records = data['data']
                        print(f"Number of records: {len(records)}")
                        if records:
                            print(f"\nSample record:")
                            import json
                            print(json.dumps(records[0], indent=2))
                            print(f"\nRecord keys: {list(records[0].keys())}")
                elif isinstance(data, list):
                    print(f"Number of records: {len(data)}")
                    if data:
                        print(f"\nSample record:")
                        import json
                        print(json.dumps(data[0], indent=2))

                return True
            else:
                print(f"❌ HTTP {response.status_code}: {response.text[:500]}")
                return False

        except Exception as e:
            print(f"❌ Error: {str(e)[:300]}")
            return False


async def main():
    print("\n" + "="*80)
    print("Testing NP6-345-CD with operatingDay parameter")
    print("="*80 + "\n")

    # Authenticate
    print("Authenticating...")
    token_data = await authenticate()
    print("✅ Authenticated\n")

    subscription_key = os.getenv("ERCOT_SUBSCRIPTION_KEY")
    test_date = "2025-10-09"

    # Test with operatingDay (singular, not operatingDate!)
    param_patterns = [
        ("operatingDay (FROM/TO)", {
            "operatingDayFrom": test_date,
            "operatingDayTo": test_date,
        }),
        ("operatingDay (single value)", {
            "operatingDay": test_date,
        }),
        ("operatingDay (range syntax)", {
            "operatingDay[from]": test_date,
            "operatingDay[to]": test_date,
        }),
    ]

    endpoint = "np6-345-cd/act_sys_load_by_wzn"

    for pattern_name, params in param_patterns:
        print(f"\n{'='*80}")
        print(f"Pattern: {pattern_name}")
        print(f"{'='*80}\n")

        success = await test_endpoint(endpoint, params, token_data, subscription_key)

        if success:
            print(f"\n{'='*80}")
            print(f"✅ SOLUTION FOUND: {pattern_name}")
            print(f"Parameters: {params}")
            print(f"{'='*80}\n")
            return

        await asyncio.sleep(0.5)

    print(f"\n{'='*80}")
    print("❌ No working parameter pattern found")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    asyncio.run(main())

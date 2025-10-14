#!/usr/bin/env python3
"""
Test Fuel Mix endpoint with SCEDTimestamp parameters.
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


async def main():
    print("\n" + "="*80)
    print("Testing Fuel Mix with SCEDTimestamp")
    print("="*80)

    # Authenticate
    print("\nAuthenticating...")
    token_data = await authenticate()
    print("✅ Authenticated\n")

    subscription_key = os.getenv("ERCOT_SUBSCRIPTION_KEY")
    BASE_URL = "https://api.ercot.com/api/public-reports"

    headers = {
        "Authorization": f"Bearer {token_data['access_token']}",
        "Ocp-Apim-Subscription-Key": subscription_key,
        "Accept": "application/json",
    }

    test_date = "2025-10-09"
    endpoint = "np3-910-er/2d_agg_gen_summary"
    url = f"{BASE_URL}/{endpoint}"

    # Test with SCEDTimestamp
    params = {
        "SCEDTimestampFrom": f"{test_date}T00:00",
        "SCEDTimestampTo": f"{test_date}T23:55",
    }

    print(f"Testing: {endpoint}")
    print(f"Params: {params}\n")

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(url, params=params, headers=headers)

            print(f"Status Code: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                print(f"✅ SUCCESS!")

                if isinstance(data, dict) and "data" in data:
                    records = data["data"]
                    print(f"Records: {len(records)}")
                    if records:
                        print(f"\nSample record:")
                        import json
                        sample = records[0]
                        if isinstance(sample, list):
                            print(f"  Row (array): {sample[:5]}...")
                        else:
                            print(json.dumps(sample, indent=2)[:500])

                        # Get field names from schema
                        if "fields" in data:
                            field_names = [f["name"] for f in data["fields"]]
                            print(f"\nFields ({len(field_names)} total):")
                            for i, name in enumerate(field_names[:20]):
                                print(f"  {i+1}. {name}")
                            if len(field_names) > 20:
                                print(f"  ... and {len(field_names) - 20} more")

                    print(f"\n✅ FUEL MIX WORKING!")
                    print(f"Parameters: SCEDTimestampFrom/To")
                    return True

            else:
                print(f"❌ HTTP {response.status_code}: {response.text[:300]}")
                return False

        except Exception as e:
            print(f"❌ Error: {str(e)[:300]}")
            return False


if __name__ == "__main__":
    asyncio.run(main())

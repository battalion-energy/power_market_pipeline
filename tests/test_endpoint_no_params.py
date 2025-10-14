#!/usr/bin/env python3
"""
Test calling the endpoint with NO parameters to see what it returns.
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


async def test_endpoint_no_params(endpoint, token_data, subscription_key):
    """Test an endpoint with NO parameters."""
    BASE_URL = "https://api.ercot.com/api/public-reports"

    headers = {
        "Authorization": f"Bearer {token_data['access_token']}",
        "Ocp-Apim-Subscription-Key": subscription_key,
        "Accept": "application/json",
    }

    url = f"{BASE_URL}/{endpoint}"

    print(f"Testing endpoint: {url}")
    print("With NO parameters\n")

    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            response = await client.get(url, headers=headers)

            print(f"Status Code: {response.status_code}")
            print(f"Headers: {dict(response.headers)}")
            print(f"\nResponse Body:\n{response.text[:2000]}")

            if response.status_code == 200:
                data = response.json()
                print(f"\n✅ SUCCESS with no parameters!")
                print(f"Response type: {type(data)}")
                if isinstance(data, dict):
                    print(f"Response keys: {list(data.keys())}")
                    if "data" in data:
                        print(f"Number of records: {len(data['data'])}")
                        if data['data']:
                            print(f"Sample record: {data['data'][0]}")
                            print(f"Record keys: {list(data['data'][0].keys())}")
                elif isinstance(data, list):
                    print(f"Number of records: {len(data)}")
                    if data:
                        print(f"Sample record: {data[0]}")
                        print(f"Record keys: {list(data[0].keys())}")

        except Exception as e:
            print(f"\n❌ Error: {str(e)}")


async def main():
    print("\n" + "="*80)
    print("Testing NP6-345-CD with NO parameters")
    print("="*80 + "\n")

    # Authenticate
    print("Authenticating...")
    token_data = await authenticate()
    print("✅ Authenticated\n")

    subscription_key = os.getenv("ERCOT_SUBSCRIPTION_KEY")

    # Test the actual load endpoint with no params
    await test_endpoint_no_params(
        "np6-345-cd/act_sys_load_by_wzn",
        token_data,
        subscription_key
    )


if __name__ == "__main__":
    asyncio.run(main())

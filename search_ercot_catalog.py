#!/usr/bin/env python3
"""
Search ERCOT API catalog for reports related to outages, lambda, and demand.
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


async def get_catalog(token_data, subscription_key):
    """Get the full API catalog."""
    BASE_URL = "https://api.ercot.com/api/public-reports"

    headers = {
        "Authorization": f"Bearer {token_data['access_token']}",
        "Ocp-Apim-Subscription-Key": subscription_key,
        "Accept": "application/json",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            # Try to get catalog/index
            response = await client.get(BASE_URL, headers=headers)

            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ Catalog request failed: {response.status_code}")
                return None

        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return None


async def search_reports(token_data, subscription_key, search_terms):
    """Search for reports by testing known report code patterns."""
    BASE_URL = "https://api.ercot.com/api/public-reports"

    headers = {
        "Authorization": f"Bearer {token_data['access_token']}",
        "Ocp-Apim-Subscription-Key": subscription_key,
        "Accept": "application/json",
    }

    # Test various report codes for outages, lambda, demand
    test_reports = [
        # Outages
        ("np1-346-er", "unpl_res_outages"),
        ("np1-346-cd", "unpl_res_outages"),
        ("np3-233-er", "unpl_res_outages"),
        ("np3-233-cd", "unpl_res_outages"),
        ("np6-346-er", "unpl_res_outages"),
        ("np6-233-cd", "outages"),
        ("np4-346-cd", "outages"),
        # Lambda / System constraints
        ("np4-523-er", "dam_sys_lambda"),
        ("np4-191-er", "dam_sys_lambda"),
        ("np4-190-cd", "dam_sys_lambda"),
        ("np4-188-cd", "sys_lambda"),
        ("np6-523-cd", "sys_lambda"),
        # System demand
        ("np6-322-er", "act_sys_load_5_min"),
        ("np6-322-cd", "sys_load"),
        ("np6-345-er", "sys_load"),
        ("np3-322-cd", "sys_load"),
    ]

    print(f"\nSearching {len(test_reports)} potential report endpoints...")
    print("="*80)

    found = []

    for report_code, endpoint_path in test_reports:
        endpoint = f"{report_code}/{endpoint_path}"
        url = f"{BASE_URL}/{endpoint}"

        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.get(url, headers=headers)

                if response.status_code == 200:
                    data = response.json()
                    if "report" in data:
                        report_name = data["report"].get("reportDisplayName", "Unknown")
                        print(f"✅ FOUND: {endpoint}")
                        print(f"   Name: {report_name}")

                        # Show date fields
                        if "fields" in data:
                            date_fields = [f for f in data["fields"]
                                          if "date" in f["name"].lower() or "time" in f["name"].lower()]
                            if date_fields:
                                print(f"   Date fields:")
                                for field in date_fields[:3]:
                                    print(f"     - {field['name']} (hasRange: {field.get('hasRange')})")

                        found.append((endpoint, report_name))
                        print()

            except Exception:
                pass

        await asyncio.sleep(0.2)

    return found


async def main():
    print("\n" + "="*80)
    print("SEARCHING ERCOT API CATALOG")
    print("="*80)

    # Authenticate
    print("\nAuthenticating...")
    token_data = await authenticate()
    print("✅ Authenticated\n")

    subscription_key = os.getenv("ERCOT_SUBSCRIPTION_KEY")

    # Try to get catalog
    print("Attempting to fetch API catalog...")
    catalog = await get_catalog(token_data, subscription_key)

    if catalog:
        print(f"✅ Got catalog response:")
        print(f"   Keys: {list(catalog.keys())[:10]}")
    else:
        print("⚠️ No catalog endpoint available, will search manually\n")

    # Search for specific reports
    search_terms = ["outage", "lambda", "demand", "constraint"]
    found = await search_reports(token_data, subscription_key, search_terms)

    print("\n" + "="*80)
    print("SEARCH RESULTS")
    print("="*80)

    if found:
        print(f"\nFound {len(found)} working endpoints:")
        for endpoint, name in found:
            print(f"  ✅ {endpoint}")
            print(f"     {name}")
    else:
        print("\n❌ No additional working endpoints found")
        print("\nConclusion:")
        print("  - Unplanned Outages: Not available via Public API")
        print("  - DAM System Lambda: Not available via Public API")
        print("  - System Wide Demand: Can calculate from NP6-345-CD (weather zones)")


if __name__ == "__main__":
    asyncio.run(main())

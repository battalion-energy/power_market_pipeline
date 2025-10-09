#!/usr/bin/env python3
"""
Test script to check PJM Data Miner 2 API access requirements.
This will attempt to access the API without authentication to determine
if a certificate/API key is required.
"""

import requests
import json
from datetime import datetime, timedelta
import time

# PJM Data Miner 2 API base URL
BASE_URL = "https://api.pjm.com/api/v1"

# Alternative endpoints to test
DATAMINER_URL = "https://dataminer2.pjm.com/api"

def test_api_access():
    """Test various PJM API endpoints to determine access requirements."""

    print("=" * 80)
    print("Testing PJM Data Miner 2 API Access")
    print("=" * 80)

    # Test endpoints (common patterns for energy market APIs)
    test_endpoints = [
        f"{BASE_URL}/da_hrl_lmps",  # Day-ahead hourly LMPs
        f"{BASE_URL}/rt_hrl_lmps",  # Real-time hourly LMPs
        f"{DATAMINER_URL}/da_hrl_lmps",
    ]

    headers = {
        'User-Agent': 'Mozilla/5.0 (compatible; PJM-API-Test/1.0)',
        'Accept': 'application/json'
    }

    for endpoint in test_endpoints:
        print(f"\nTesting endpoint: {endpoint}")
        print("-" * 80)

        try:
            response = requests.get(
                endpoint,
                headers=headers,
                timeout=10
            )

            print(f"Status Code: {response.status_code}")
            print(f"Headers: {dict(response.headers)}")

            if response.status_code == 200:
                print("✓ SUCCESS: Public access available!")
                print(f"Response preview: {response.text[:500]}")
            elif response.status_code == 401:
                print("✗ Authentication required (401 Unauthorized)")
            elif response.status_code == 403:
                print("✗ Access forbidden (403) - May need certificate/API key")
            elif response.status_code == 404:
                print("✗ Endpoint not found (404)")
            else:
                print(f"Response: {response.text[:500]}")

        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")

    # Test the API portal endpoint
    print("\n" + "=" * 80)
    print("Testing API Portal endpoint")
    print("=" * 80)

    try:
        portal_response = requests.get(
            "https://apiportal.pjm.com/api",
            headers=headers,
            timeout=10
        )
        print(f"Status Code: {portal_response.status_code}")
        print(f"Response preview: {portal_response.text[:500]}")
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print("""
Based on the tests above:
- If you see 200 responses: Public access is available
- If you see 401/403 responses: Authentication/certificate required
- Check response headers for rate limit information (X-RateLimit-*, etc.)
    """)

if __name__ == "__main__":
    test_api_access()

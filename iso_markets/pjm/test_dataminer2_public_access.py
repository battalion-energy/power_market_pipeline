#!/usr/bin/env python3
"""
Test if PJM DataMiner2 webservices are publicly accessible without authentication.
"""

import requests
import json
from datetime import datetime, timedelta

print("=" * 80)
print("Testing PJM DataMiner2 Public Access")
print("=" * 80)

# Common DataMiner2 webservice endpoints
base_url = "https://dataminer2.pjm.com"

test_endpoints = [
    "/dataminer2/list",
    "/list",
    "/feed/hrl_load_metered/def/hrl_load_metered.xml",
    "/feed/da_hrl_lmps/def/da_hrl_lmps.xml",
    "/feed/rt_hrl_lmps/def/rt_hrl_lmps.xml",
    "/feed/ancillary_services/def/ancillary_services.xml",
]

headers = {
    'User-Agent': 'Mozilla/5.0 (compatible; PJM-Test/1.0)',
    'Accept': '*/*'
}

for endpoint in test_endpoints:
    url = f"{base_url}{endpoint}"
    print(f"\nTesting: {url}")
    print("-" * 80)

    try:
        response = requests.get(url, headers=headers, timeout=10)
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            print("✓ SUCCESS - Publicly accessible!")
            content_type = response.headers.get('Content-Type', '')
            print(f"Content-Type: {content_type}")
            print(f"Content length: {len(response.content)} bytes")

            # Show preview
            preview = response.text[:500]
            print(f"Preview:\n{preview}")

        elif response.status_code == 401:
            print("✗ Authentication required")
        elif response.status_code == 403:
            print("✗ Forbidden")
        elif response.status_code == 404:
            print("✗ Not found")
        else:
            print(f"Response: {response.text[:200]}")

    except Exception as e:
        print(f"Error: {e}")

# Try a data request with parameters
print("\n" + "=" * 80)
print("Testing data retrieval with parameters")
print("=" * 80)

# Test actual data endpoint with date range (last 7 days)
end_date = datetime.now()
start_date = end_date - timedelta(days=7)

data_tests = [
    {
        'url': f"{base_url}/feed/da_hrl_lmps",
        'params': {
            'startRow': '1',
            'rowCount': '10',
            'format': 'csv'
        }
    },
    {
        'url': f"{base_url}/feed/da_hrl_lmps",
        'params': {
            'startRow': '1',
            'rowCount': '10',
            'format': 'json'
        }
    },
    {
        'url': f"{base_url}/feed/hrl_load_metered",
        'params': {
            'startRow': '1',
            'rowCount': '10',
            'format': 'json'
        }
    },
    {
        'url': f"{base_url}/feed/rt_hrl_lmps",
        'params': {
            'startRow': '1',
            'rowCount': '10',
            'format': 'json'
        }
    }
]

for test in data_tests:
    print(f"\nTesting: {test['url']}")
    print(f"Params: {test['params']}")
    print("-" * 80)

    try:
        response = requests.get(
            test['url'],
            params=test['params'],
            headers=headers,
            timeout=10
        )
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            print("✓ PUBLIC ACCESS CONFIRMED!")
            print(f"Content-Type: {response.headers.get('Content-Type')}")
            print(f"Preview:\n{response.text[:500]}")
        else:
            print(f"Response: {response.text[:200]}")

    except Exception as e:
        print(f"Error: {e}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("""
If you see 200 status codes above:
  → DataMiner2 webservices ARE publicly accessible
  → No API key needed for these endpoints
  → Can use direct HTTP requests instead of api.pjm.com

If you see 401/403 status codes:
  → Authentication required
  → Must use api.pjm.com with API key
""")

#!/usr/bin/env python3
"""
Test if PJM API endpoints are accessible without authentication for public data.
"""

import requests
from datetime import datetime, timedelta

print("=" * 80)
print("Testing PJM API Access WITHOUT Authentication")
print("=" * 80)

base_url = "https://api.pjm.com/api/v1"

# Test common public endpoints
endpoints = [
    "da_hrl_lmps",
    "rt_hrl_lmps",
    "rt_fivemin_lmps",
    "hrl_load_metered",
]

# Use recent date range (last 2 days)
end_date = datetime.now()
start_date = end_date - timedelta(days=2)

for endpoint in endpoints:
    print(f"\n{'='*80}")
    print(f"Testing: {endpoint}")
    print('='*80)

    url = f"{base_url}/{endpoint}"

    params = {
        'startRow': '1',
        'rowCount': '5',
        'datetime_beginning_ept': start_date.strftime('%Y-%m-%d 00:00:00'),
        'datetime_ending_ept': end_date.strftime('%Y-%m-%d 23:59:59')
    }

    # Test without any authentication
    print(f"URL: {url}")
    print(f"Params: {params}")
    print("-" * 80)

    try:
        response = requests.get(url, params=params, timeout=10)

        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            print("✓ SUCCESS - Public access works WITHOUT API key!")
            print(f"Content-Type: {response.headers.get('Content-Type')}")

            # Try to parse JSON
            try:
                data = response.json()
                print(f"Response structure: {list(data.keys()) if isinstance(data, dict) else 'List of items'}")
                if isinstance(data, dict) and 'items' in data:
                    print(f"Number of items: {len(data.get('items', []))}")
                    if data.get('items'):
                        print(f"First item keys: {list(data['items'][0].keys())}")
            except:
                print(f"Response preview: {response.text[:200]}")

        elif response.status_code == 401:
            print("✗ Authentication required (401)")
        elif response.status_code == 403:
            print("✗ Forbidden (403)")
        else:
            print(f"Status: {response.status_code}")
            print(f"Response: {response.text[:200]}")

    except Exception as e:
        print(f"Error: {e}")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print("""
If you see 200 SUCCESS responses above:
  → You can use the scripts WITHOUT an API key!
  → Just comment out the API key requirement in the code

If you see 401/403 responses:
  → API key is required
  → You need to contact PJM support to get API access
""")

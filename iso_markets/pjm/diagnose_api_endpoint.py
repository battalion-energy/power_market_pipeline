#!/usr/bin/env python3
"""
Diagnose the correct PJM API endpoint.
"""

import requests
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv('PJM_API_KEY')

# Try different possible endpoints
endpoints = [
    "https://api.pjm.com/api/v1/da_hrl_lmps",
    "https://dataminer2.pjm.com/api/v1/da_hrl_lmps",
    "https://apiportal.pjm.com/api/v1/da_hrl_lmps",
    "https://pjm-dataminer2.azure-api.net/api/v1/da_hrl_lmps",
]

headers = {
    'Ocp-Apim-Subscription-Key': API_KEY,
    'Accept': 'application/json',
}

params = {
    'rowCount': '1',
    'startRow': '1',
}

print("Testing PJM API endpoints...")
print("=" * 80)

for endpoint in endpoints:
    print(f"\nTrying: {endpoint}")
    try:
        response = requests.get(endpoint, headers=headers, params=params, timeout=10)
        print(f"  Status: {response.status_code}")

        if response.status_code == 200:
            print(f"  ✓ SUCCESS! This endpoint works!")
            print(f"  Response preview: {str(response.json())[:200]}")
            break
        else:
            print(f"  Response: {response.text[:200]}")

    except requests.exceptions.ConnectionError as e:
        print(f"  ✗ Connection error: {str(e)[:100]}")
    except Exception as e:
        print(f"  ✗ Error: {str(e)[:100]}")

print("\n" + "=" * 80)
print("If none work, check:")
print("1. Your network/firewall settings")
print("2. VPN connection if required")
print("3. Contact PJM support for current API endpoint")

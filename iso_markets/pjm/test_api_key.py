#!/usr/bin/env python3
"""
Quick test script to verify PJM API key is working.
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from pjm_api_client import PJMAPIClient

# Load environment variables
load_dotenv()

print("Testing PJM API Key...")
print("=" * 80)

try:
    # Initialize client (will check for PJM_API_KEY env var)
    client = PJMAPIClient(requests_per_minute=6)
    print("✓ API key loaded successfully")

    # Test with a small query - just 1 day for AEP hub
    print("\nTesting API connection with small query (AEP hub, 1 day)...")
    data = client.get_day_ahead_lmps(
        start_date='2024-12-01',
        end_date='2024-12-01',
        pnode_id='51291'  # AEP hub
    )

    print(f"✓ API connection successful!")
    print(f"✓ Received data: {len(data) if isinstance(data, list) else 'response received'}")

    if isinstance(data, dict) and 'items' in data:
        print(f"✓ Number of records: {len(data['items'])}")
        if data['items']:
            print(f"\nSample record:")
            print(data['items'][0])
    elif isinstance(data, list) and data:
        print(f"✓ Number of records: {len(data)}")
        print(f"\nSample record:")
        print(data[0])

    print("\n" + "=" * 80)
    print("SUCCESS! Your API key is working correctly.")
    print("You can now download historical data.")
    print("=" * 80)

except ValueError as e:
    print(f"\n✗ ERROR: {e}")
    print("\nPlease update your .env file with:")
    print("  PJM_API_KEY=your_primary_or_secondary_key")
    sys.exit(1)

except Exception as e:
    print(f"\n✗ ERROR: {e}")
    print("\nThis could indicate:")
    print("  1. Invalid API key")
    print("  2. Network connection issue")
    print("  3. PJM API service issue")
    sys.exit(1)

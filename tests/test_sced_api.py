#!/usr/bin/env python3
"""Test SCED Gen endpoint to see what data is actually available."""

import asyncio
import os
from dotenv import load_dotenv
from ercot_ws_downloader import ERCOTWebServiceClient

load_dotenv()

async def test_sced_endpoint():
    """Test SCED endpoint with various date ranges."""
    client = ERCOTWebServiceClient(
        username=os.getenv("ERCOT_USERNAME"),
        password=os.getenv("ERCOT_PASSWORD"),
        subscription_key=os.getenv("ERCOT_SUBSCRIPTION_KEY"),
    )
    
    # Test recent date (should have data - 60 days ago from Oct 8, 2025 is Aug 9, 2025)
    test_dates = [
        ("2025-07-15", "Recent data within 60-day window"),
        ("2024-12-01", "December 2024"),
        ("2024-06-01", "June 2024"),
        ("2023-12-23", "December 2023 (our parquet ends here)"),
    ]
    
    for date, desc in test_dates:
        print(f"\n{'='*70}")
        print(f"Testing: {desc} ({date})")
        print('='*70)
        
        try:
            params = {
                "SCEDTimestampFrom": f"{date}T00:00",
                "SCEDTimestampTo": f"{date}T23:55",
                "page": 1,
                "size": 100,
            }
            
            data = await client.get_paginated_data(
                "np3-965-er/60_sced_gen_res_data",
                params,
                page_size=100,
                max_pages=1
            )
            print(f"✓ Got {len(data)} records")
            if data:
                print(f"  First record timestamp: {data[0].get('SCEDTimestamp', 'N/A')}")
                print(f"  Sample fields: {list(data[0].keys())[:10]}")
        except Exception as e:
            print(f"✗ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_sced_endpoint())

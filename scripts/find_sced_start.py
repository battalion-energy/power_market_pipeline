#!/usr/bin/env python3
"""Find the earliest available SCED Gen data."""

import asyncio
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from ercot_ws_downloader import ERCOTWebServiceClient

load_dotenv()

async def binary_search_start_date():
    """Binary search to find the earliest available SCED data."""
    client = ERCOTWebServiceClient(
        username=os.getenv("ERCOT_USERNAME"),
        password=os.getenv("ERCOT_PASSWORD"),
        subscription_key=os.getenv("ERCOT_SUBSCRIPTION_KEY"),
    )
    
    # We know Dec 23, 2023 has NO data, June 2024 HAS data
    earliest_no_data = datetime(2023, 12, 23)
    latest_has_data = datetime(2024, 6, 1)
    
    print("Binary searching for earliest SCED Gen data...")
    print(f"Start: {earliest_no_data.date()} (no data)")
    print(f"End: {latest_has_data.date()} (has data)")
    print()
    
    while (latest_has_data - earliest_no_data).days > 1:
        midpoint = earliest_no_data + (latest_has_data - earliest_no_data) / 2
        midpoint = midpoint.replace(hour=12, minute=0)  # Use noon
        
        params = {
            "SCEDTimestampFrom": midpoint.strftime("%Y-%m-%dT%H:%M"),
            "SCEDTimestampTo": midpoint.strftime("%Y-%m-%dT%H:%M"),
            "page": 1,
            "size": 10,
        }
        
        try:
            data = await client.get_paginated_data(
                "np3-965-er/60_sced_gen_res_data",
                params,
                page_size=10,
                max_pages=1
            )
            
            if data:
                print(f"  {midpoint.date()}: ✓ {len(data)} records (has data)")
                latest_has_data = midpoint
            else:
                print(f"  {midpoint.date()}: ✗ 0 records (no data)")
                earliest_no_data = midpoint
                
        except Exception as e:
            print(f"  {midpoint.date()}: ERROR - {e}")
            earliest_no_data = midpoint
    
    print()
    print(f"Earliest SCED Gen data: {latest_has_data.date()}")
    return latest_has_data

if __name__ == "__main__":
    result = asyncio.run(binary_search_start_date())

#!/usr/bin/env python3
"""Direct test of ERCOT WebService client."""

import asyncio
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import sys

load_dotenv()

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from downloaders.ercot.webservice_client import ERCOTWebServiceClient


async def test_ercot_webservice():
    """Test ERCOT WebService API directly."""
    print("Testing ERCOT WebService API...")
    
    # Create client - it reads credentials from env vars automatically
    client = ERCOTWebServiceClient()
    
    # Test dates - just 1 day for speed
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1)
    
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    print("\nAuthenticating...")
    
    try:
        # Get DAM prices
        print("\nFetching DAM SPP prices...")
        df = await client.get_dam_spp_prices(
            start_date=start_date,
            end_date=end_date,
            settlement_points=None  # Get all data
        )
        
        if df is not None and not df.empty:
            print(f"\nReceived {len(df)} records")
            print("\nColumns:", list(df.columns))
            print("\nFirst 5 rows:")
            print(df.head())
            print("\nData types:")
            print(df.dtypes)
            
            # Show unique settlement points
            if 'settlement_point' in df.columns:
                print("\nUnique settlement points:")
                print(df['settlement_point'].unique()[:10])
        else:
            print("\nNo data received!")
            
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pass  # No close method needed


if __name__ == "__main__":
    asyncio.run(test_ercot_webservice())

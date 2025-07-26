#!/usr/bin/env python
"""Test script to download 3 days of data for ERCOT only."""

import asyncio
import os
import sys
from datetime import datetime, timedelta

# Add the project directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from power_market_pipeline.downloaders.base_v2 import DownloadConfig
from power_market_pipeline.downloaders.ercot.downloader_v2 import ERCOTDownloaderV2
from power_market_pipeline.database import init_db, get_db
from sqlalchemy import text


async def test_ercot_download():
    """Test downloading 3 days of ERCOT data."""
    print("Power Market Pipeline - ERCOT 3 Day Download Test")
    print("="*60)
    
    # Initialize database
    print("\n1. Initializing database...")
    init_db()
    print("✓ Database initialized")
    
    # Set up test parameters
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3)
    
    config = DownloadConfig(
        start_date=start_date,
        end_date=end_date,
        data_types=["lmp"],
        output_dir="/tmp/power_market_pipeline",
        batch_size=1000,
        retry_attempts=3,
        retry_delay=60
    )
    
    print(f"\n2. Test Parameters:")
    print(f"   Start Date: {start_date.strftime('%Y-%m-%d %H:%M')}")
    print(f"   End Date: {end_date.strftime('%Y-%m-%d %H:%M')}")
    
    # Create ERCOT downloader
    downloader = ERCOTDownloaderV2(config)
    
    print("\n3. Testing ERCOT downloads...")
    print("-"*60)
    
    # Test DAM LMP
    try:
        print("\nDownloading DAM LMP data...")
        dam_count = await downloader.download_lmp("DAM", start_date, end_date)
        print(f"✓ Downloaded {dam_count} DAM records")
    except Exception as e:
        print(f"✗ DAM Error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Test RT5M LMP
    try:
        print("\nDownloading RT5M LMP data...")
        rt_count = await downloader.download_lmp("RT5M", start_date, end_date)
        print(f"✓ Downloaded {rt_count} RT5M records")
    except Exception as e:
        print(f"✗ RT5M Error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Audit database
    print("\n4. Auditing downloaded data...")
    print("-"*60)
    
    with get_db() as db:
        # Check LMP table
        result = db.execute(text("""
            SELECT 
                iso,
                market,
                COUNT(*) as record_count,
                MIN(interval_start) as earliest,
                MAX(interval_start) as latest,
                COUNT(DISTINCT location) as unique_locations
            FROM lmp
            WHERE iso = 'ERCOT'
            GROUP BY iso, market
            ORDER BY market
        """))
        
        lmp_data = result.fetchall()
        if lmp_data:
            print("\nLMP Data Summary:")
            for row in lmp_data:
                print(f"  {row.market}: {row.record_count} records")
                print(f"    - Date range: {row.earliest} to {row.latest}")
                print(f"    - Locations: {row.unique_locations}")
        else:
            print("\n  No LMP data found in database")


if __name__ == "__main__":
    # Check for required env vars
    required_vars = ["DATABASE_URL"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"Error: Missing required environment variables: {missing_vars}")
        print("\nPlease set the following in your .env file:")
        print("DATABASE_URL=postgresql://user:password@localhost:5432/power_market")
        sys.exit(1)
    
    # Check for ERCOT credentials
    if not os.getenv("ERCOT_USERNAME") or not os.getenv("ERCOT_PASSWORD"):
        print("\nWarning: ERCOT_USERNAME and ERCOT_PASSWORD not set")
        print("Selenium downloads will fail without credentials")
    
    # Run the test
    asyncio.run(test_ercot_download())
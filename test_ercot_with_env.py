#!/usr/bin/env python
"""Simple test for ERCOT downloader with env loading."""

import asyncio
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from downloaders.base_v2 import DownloadConfig
from downloaders.ercot.downloader_v2 import ERCOTDownloaderV2
from database import init_db, get_db
from sqlalchemy import text


async def main():
    """Test ERCOT download."""
    print("Testing ERCOT Downloader")
    print("="*60)
    
    # Check env vars
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("Error: DATABASE_URL not set in .env file")
        return
    
    print(f"\nDatabase URL: {db_url.split('@')[1] if '@' in db_url else db_url}")
    
    # Initialize database
    print("\nInitializing database...")
    try:
        init_db()
        print("✓ Database initialized")
    except Exception as e:
        print(f"✗ Database error: {e}")
        return
    
    # Create config
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
    
    print(f"\nDate range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Create downloader
    try:
        downloader = ERCOTDownloaderV2(config)
        print("✓ ERCOT downloader created")
    except Exception as e:
        print(f"✗ Error creating downloader: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test download for historical data (using Selenium)
    print("\nTesting download (DAM) - using Selenium for historical data...")
    try:
        # Use dates before webservice cutoff (Dec 11, 2023)
        test_start = datetime(2023, 10, 1)
        test_end = datetime(2023, 10, 3)
        count = await downloader.download_lmp("DAM", test_start, test_end)
        print(f"✓ Downloaded {count} DAM records")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        
    # Check database
    print("\nChecking database...")
    with get_db() as db:
        result = db.execute(text("SELECT COUNT(*) as count FROM lmp WHERE iso = 'ERCOT'"))
        count = result.scalar()
        print(f"Total records in database: {count}")


if __name__ == "__main__":
    asyncio.run(main())
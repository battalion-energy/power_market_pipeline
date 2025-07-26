#!/usr/bin/env python
"""Simple test for ERCOT downloader."""

import asyncio
import os
from datetime import datetime, timedelta

from downloaders.base_v2 import DownloadConfig
from downloaders.ercot.downloader_v2 import ERCOTDownloaderV2
from database import init_db, get_db
from sqlalchemy import text


async def main():
    """Test ERCOT download."""
    print("Testing ERCOT Downloader")
    print("="*60)
    
    # Check env vars
    if not os.getenv("DATABASE_URL"):
        print("Error: DATABASE_URL not set")
        return
    
    # Initialize database
    print("\nInitializing database...")
    init_db()
    
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
    
    print(f"\nDate range: {start_date} to {end_date}")
    
    # Create downloader
    try:
        downloader = ERCOTDownloaderV2(config)
        print("✓ ERCOT downloader created")
    except Exception as e:
        print(f"✗ Error creating downloader: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test download
    print("\nTesting DAM download...")
    try:
        count = await downloader.download_lmp("DAM", start_date, end_date)
        print(f"✓ Downloaded {count} records")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
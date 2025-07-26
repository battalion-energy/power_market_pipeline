#!/usr/bin/env python3
"""Simple test of ERCOT download with database storage."""

import asyncio
import os
import sys
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import init_db, get_db
from database.models_v2 import LMP
from downloaders.ercot.downloader_v2 import ERCOTDownloaderV2
from downloaders.base_v2 import DownloadConfig


async def test_ercot_download():
    """Test downloading and storing ERCOT data."""
    print("Initializing database...")
    init_db()
    
    # Create config for 1 day test
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1)
    
    config = DownloadConfig(
        start_date=start_date,
        end_date=end_date,
        data_types=['lmp'],
        output_dir='./test_data'
    )
    
    print(f"\nTesting ERCOT download for {start_date.date()} to {end_date.date()}")
    
    try:
        # Create downloader
        downloader = ERCOTDownloaderV2(config)
        
        # Download DAM data
        print("\nDownloading DAM LMP data...")
        dam_count = await downloader.download_lmp('DAM', start_date, end_date)
        print(f"Downloaded {dam_count} DAM records")
        
        # Check database
        with get_db() as db:
            total_records = db.query(LMP).count()
            print(f"\nTotal records in database: {total_records}")
            
            # Show sample records
            sample = db.query(LMP).limit(5).all()
            if sample:
                print("\nSample records:")
                for record in sample:
                    print(f"  {record.interval_start} | {record.location} | ${record.lmp:.2f}")
            
            # Show unique locations
            from sqlalchemy import distinct
            locations = db.query(distinct(LMP.location)).limit(10).all()
            print(f"\nSample locations: {[loc[0] for loc in locations]}")
            
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_ercot_download())

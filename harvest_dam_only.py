#!/usr/bin/env python3
"""Harvest DAM data from all ISOs."""

import asyncio
import os
import sys
from datetime import datetime, timedelta
from dotenv import load_dotenv
import psycopg2

load_dotenv()

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import init_db, get_db
from database.models_v2 import LMP
from downloaders.ercot.downloader_v2 import ERCOTDownloaderV2
from downloaders.caiso.downloader_v2 import CAISODownloaderV2
from downloaders.isone.downloader_v2 import ISONEDownloaderV2
from downloaders.nyiso.downloader_v2 import NYISODownloaderV2
from downloaders.base_v2 import DownloadConfig
from sqlalchemy import text, func


async def harvest_iso_dam(iso: str, downloader, start_date: datetime, end_date: datetime):
    """Harvest DAM data for a single ISO."""
    print(f"\n{'=' * 60}")
    print(f"Harvesting {iso} DAM data from {start_date.date()} to {end_date.date()}")
    print(f"{'=' * 60}")
    
    try:
        count = await downloader.download_lmp('DAM', start_date, end_date)
        print(f"\n✅ {iso}: Downloaded {count:,} DAM records")
        return count
    except Exception as e:
        print(f"\n❌ {iso} Error: {e}")
        import traceback
        traceback.print_exc()
        return 0


async def main():
    """Main harvesting function."""
    print("\n🌾 Power Market DAM Data Harvester")
    print("Harvesting Day-Ahead Market prices...\n")
    
    # Initialize database
    print("Initializing database...")
    init_db()
    
    # Clear existing data
    print("\nClearing existing data...")
    with get_db() as db:
        db.execute(text("TRUNCATE TABLE lmp CASCADE"))
        db.execute(text("TRUNCATE TABLE locations CASCADE"))
        db.commit()
        print("✓ Database cleared")
    
    # Configuration
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3)
    
    config = DownloadConfig(
        start_date=start_date,
        end_date=end_date,
        data_types=['lmp'],
        output_dir='./data'
    )
    
    print(f"\n📅 Period: {start_date.date()} to {end_date.date()} (3 days)")
    print(f"📊 Market: Day-Ahead (DAM)")
    
    # Create downloaders
    downloaders = {
        'ERCOT': ERCOTDownloaderV2(config),
        'CAISO': CAISODownloaderV2(config),
        'ISONE': ISONEDownloaderV2(config),
        'NYISO': NYISODownloaderV2(config)
    }
    
    # Download DAM data for each ISO
    results = {}
    
    for iso, downloader in downloaders.items():
        count = await harvest_iso_dam(iso, downloader, start_date, end_date)
        results[iso] = count
    
    # Summary
    print("\n\n📊 SUMMARY:")
    print("=" * 60)
    total_records = 0
    for iso, count in results.items():
        total_records += count
        print(f"{iso}: {count:,} DAM records")
    
    print(f"\nTotal records: {total_records:,}")
    
    # Check database
    with get_db() as db:
        db_count = db.query(LMP).count()
        print(f"\nRecords in database: {db_count:,}")
        
        # Show sample by ISO
        iso_counts = db.query(
            LMP.iso,
            func.count(LMP.iso).label('count'),
            func.min(LMP.interval_start).label('min_date'),
            func.max(LMP.interval_start).label('max_date')
        ).group_by(LMP.iso).all()
        
        print("\nDatabase breakdown:")
        for iso, count, min_date, max_date in iso_counts:
            print(f"  {iso}: {count:,} records ({min_date.date()} to {max_date.date()})")
    
    if total_records > 0:
        print("\n✅ Success! DAM data harvested for all ISOs.")
        
        # If 3-day test is successful, provide next steps
        if '--month' not in sys.argv and '--historical' not in sys.argv:
            print("\nNext steps:")
            print("1. Run with --month flag for 1 month of data")
            print("2. Run with --historical flag for data since Jan 1, 2019")
    else:
        print("\n❌ No data collected. Please check credentials.")


async def main_with_period(days: int = None, since_date: datetime = None):
    """Main harvesting function with custom period."""
    print("\n🌾 Power Market DAM Data Harvester")
    print("Harvesting Day-Ahead Market prices...\n")
    
    # Initialize database
    print("Initializing database...")
    init_db()
    
    # Clear existing data
    print("\nClearing existing data...")
    with get_db() as db:
        db.execute(text("TRUNCATE TABLE lmp CASCADE"))
        db.execute(text("TRUNCATE TABLE locations CASCADE"))
        db.commit()
        print("✓ Database cleared")
    
    # Configuration
    end_date = datetime.now()
    if since_date:
        start_date = since_date
        period_desc = f"since {since_date.date()}"
    elif days:
        start_date = end_date - timedelta(days=days)
        period_desc = f"{days} days"
    else:
        start_date = end_date - timedelta(days=3)
        period_desc = "3 days"
    
    config = DownloadConfig(
        start_date=start_date,
        end_date=end_date,
        data_types=['lmp'],
        output_dir='./data'
    )
    
    print(f"\n📅 Period: {start_date.date()} to {end_date.date()} ({period_desc})")
    print(f"📊 Market: Day-Ahead (DAM)")
    
    # For now, only harvest ERCOT since others aren't implemented
    print("\n⚠️  Note: Currently only ERCOT is implemented")
    
    # Create downloader
    downloader = ERCOTDownloaderV2(config)
    
    # Download DAM data
    count = await harvest_iso_dam('ERCOT', downloader, start_date, end_date)
    
    # Summary
    print("\n\n📊 SUMMARY:")
    print("=" * 60)
    print(f"ERCOT: {count:,} DAM records")
    print(f"\nTotal records: {count:,}")
    
    # Check database
    with get_db() as db:
        db_count = db.query(LMP).count()
        print(f"\nRecords in database: {db_count:,}")
        
        # Show date range
        date_range = db.query(
            func.min(LMP.interval_start).label('min_date'),
            func.max(LMP.interval_start).label('max_date')
        ).first()
        
        if date_range and date_range.min_date:
            print(f"Date range: {date_range.min_date.date()} to {date_range.max_date.date()}")
            
            # Show unique locations count
            location_count = db.query(func.count(func.distinct(LMP.location))).scalar()
            print(f"Unique locations: {location_count:,}")
    
    if count > 0:
        print("\n✅ Success! DAM data harvested.")
    else:
        print("\n❌ No data collected. Please check credentials.")


if __name__ == "__main__":
    # Check for command line arguments
    if "--month" in sys.argv:
        print("\n📅 Switching to 1 month mode...")
        asyncio.run(main_with_period(days=30))
    elif "--historical" in sys.argv:
        print("\n📅 Switching to historical mode (since Jan 1, 2019)...")
        since_date = datetime(2019, 1, 1)
        asyncio.run(main_with_period(since_date=since_date))
    else:
        asyncio.run(main())

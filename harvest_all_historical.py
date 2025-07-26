#!/usr/bin/env python3
"""Harvest all historical data from all ISOs since Jan 1, 2019."""

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
from sqlalchemy import text, func
import traceback


async def harvest_ercot_historical():
    """Harvest all ERCOT historical data."""
    print("\n🌾 ERCOT Historical Data Harvest")
    print("=" * 60)
    
    # ERCOT data availability:
    # - WebService API: Dec 11, 2023 onwards
    # - Selenium scraper: Historical data before Dec 11, 2023
    
    config = DownloadConfig(
        start_date=datetime(2019, 1, 1),
        end_date=datetime.now(),
        data_types=['lmp'],
        output_dir='./data/ercot'
    )
    
    downloader = ERCOTDownloaderV2(config)
    
    # Download in yearly chunks to manage memory
    start_year = 2019
    end_year = datetime.now().year
    total_records = 0
    
    for year in range(start_year, end_year + 1):
        print(f"\n📅 Processing year {year}...")
        
        # Process each month
        for month in range(1, 13):
            # Skip future months
            if year == end_year and month > datetime.now().month:
                break
                
            start_date = datetime(year, month, 1)
            if month == 12:
                end_date = datetime(year + 1, 1, 1) - timedelta(seconds=1)
            else:
                end_date = datetime(year, month + 1, 1) - timedelta(seconds=1)
            
            # Skip if end date is in the future
            if end_date > datetime.now():
                end_date = datetime.now()
            
            print(f"\n  Processing {start_date.strftime('%Y-%m')}...")
            
            try:
                # Download DAM data
                dam_count = await downloader.download_lmp('DAM', start_date, end_date)
                print(f"    DAM: {dam_count:,} records")
                total_records += dam_count
                
                # Add delay between requests
                await asyncio.sleep(2)
                
            except Exception as e:
                print(f"    Error: {e}")
                traceback.print_exc()
                continue
    
    print(f"\n✅ ERCOT Total: {total_records:,} records")
    return total_records


async def harvest_caiso_historical():
    """Harvest all CAISO historical data."""
    print("\n🌾 CAISO Historical Data Harvest")
    print("=" * 60)
    print("⚠️  CAISO downloader not yet implemented")
    return 0


async def harvest_isone_historical():
    """Harvest all ISO-NE historical data."""
    print("\n🌾 ISO-NE Historical Data Harvest")
    print("=" * 60)
    print("⚠️  ISO-NE downloader not yet implemented")
    return 0


async def harvest_nyiso_historical():
    """Harvest all NYISO historical data."""
    print("\n🌾 NYISO Historical Data Harvest")
    print("=" * 60)
    print("⚠️  NYISO downloader not yet implemented")
    return 0


async def main():
    """Main harvesting function."""
    print("\n🌾 Power Market Historical Data Harvester")
    print("Downloading all available data since January 1, 2019...")
    print("=" * 60)
    
    # Initialize database
    print("\nInitializing database...")
    init_db()
    
    # Check existing data before starting
    print("\nChecking existing data...")
    with get_db() as db:
        existing_stats = db.query(
            LMP.iso,
            func.count(LMP.iso).label('count'),
            func.min(LMP.interval_start).label('min_date'),
            func.max(LMP.interval_start).label('max_date')
        ).group_by(LMP.iso).all()
        
        if existing_stats:
            print("\nExisting data found:")
            for iso, count, min_date, max_date in existing_stats:
                print(f"  {iso}: {count:,} records ({min_date.date()} to {max_date.date()})")
        else:
            print("No existing data found")
    
    # Ask for confirmation
    print("\n⚠️  WARNING: This will download several years of data and may take hours to complete.")
    response = input("Continue? (y/N): ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    # Harvest data from each ISO
    results = {}
    
    # ERCOT
    try:
        ercot_count = await harvest_ercot_historical()
        results['ERCOT'] = ercot_count
    except Exception as e:
        print(f"\n❌ ERCOT failed: {e}")
        results['ERCOT'] = 0
    
    # CAISO
    try:
        caiso_count = await harvest_caiso_historical()
        results['CAISO'] = caiso_count
    except Exception as e:
        print(f"\n❌ CAISO failed: {e}")
        results['CAISO'] = 0
    
    # ISO-NE
    try:
        isone_count = await harvest_isone_historical()
        results['ISONE'] = isone_count
    except Exception as e:
        print(f"\n❌ ISO-NE failed: {e}")
        results['ISONE'] = 0
    
    # NYISO
    try:
        nyiso_count = await harvest_nyiso_historical()
        results['NYISO'] = nyiso_count
    except Exception as e:
        print(f"\n❌ NYISO failed: {e}")
        results['NYISO'] = 0
    
    # Final summary
    print("\n\n" + "=" * 60)
    print("📊 FINAL SUMMARY")
    print("=" * 60)
    
    total_records = 0
    for iso, count in results.items():
        total_records += count
        print(f"{iso}: {count:,} records")
    
    print(f"\nTotal records downloaded: {total_records:,}")
    
    # Check final database state
    with get_db() as db:
        final_stats = db.query(
            LMP.iso,
            func.count(LMP.iso).label('count'),
            func.min(LMP.interval_start).label('min_date'),
            func.max(LMP.interval_start).label('max_date'),
            func.count(func.distinct(LMP.location)).label('locations')
        ).group_by(LMP.iso).all()
        
        print("\n📊 Final Database State:")
        for iso, count, min_date, max_date, locations in final_stats:
            print(f"\n{iso}:")
            print(f"  Records: {count:,}")
            print(f"  Date range: {min_date.date()} to {max_date.date()}")
            print(f"  Unique locations: {locations:,}")
            
            # Show price statistics
            price_stats = db.query(
                func.min(LMP.lmp).label('min_price'),
                func.max(LMP.lmp).label('max_price'),
                func.avg(LMP.lmp).label('avg_price')
            ).filter(LMP.iso == iso).first()
            
            if price_stats.min_price is not None:
                print(f"  Price range: ${price_stats.min_price:.2f} to ${price_stats.max_price:.2f}")
                print(f"  Average price: ${price_stats.avg_price:.2f}")
    
    print("\n✅ Historical data harvest complete!")


if __name__ == "__main__":
    asyncio.run(main())
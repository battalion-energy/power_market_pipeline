#!/usr/bin/env python3
"""Harvest all historical data from all ISOs since Jan 1, 2019 (automated version)."""

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
from downloaders.ercot.constants import WEBSERVICE_CUTOFF_DATE
from downloaders.base_v2 import DownloadConfig
from sqlalchemy import text, func
import traceback


async def harvest_ercot_historical():
    """Harvest all ERCOT historical data."""
    print("\n🌾 ERCOT Historical Data Harvest")
    print("=" * 60)
    
    # Check what data we already have
    with get_db() as db:
        existing = db.query(
            func.min(LMP.interval_start).label('min_date'),
            func.max(LMP.interval_start).label('max_date'),
            func.count(LMP.interval_start).label('count')
        ).filter(LMP.iso == 'ERCOT').first()
        
        if existing and existing.count > 0:
            print(f"Existing data: {existing.count:,} records from {existing.min_date.date()} to {existing.max_date.date()}")
            
            # Start from 2019 but skip months we already have complete data for
            start_date = datetime(2019, 1, 1)
            existing_min = existing.min_date.replace(tzinfo=None)
        else:
            print("No existing ERCOT data found")
            start_date = datetime(2019, 1, 1)
            existing_min = datetime.now()
    
    # We need to download data from 2019 up to the existing minimum date
    if start_date >= existing_min:
        print("✓ Already have complete historical data from 2019")
        return 0
    
    config = DownloadConfig(
        start_date=start_date,
        end_date=existing_min - timedelta(days=1),  # Up to day before existing data
        data_types=['lmp'],
        output_dir='./data/ercot'
    )
    
    downloader = ERCOTDownloaderV2(config)
    
    # For pre-2023 data, we need Selenium
    # The WebService only has data from Dec 11, 2023 onwards
    print(f"\n📅 Downloading historical data from {start_date.date()} to {existing_min.date()}")
    print(f"⚠️  Note: ERCOT WebService API only has data from {WEBSERVICE_CUTOFF_DATE.date()} onwards")
    print("    Selenium scraper would be needed for pre-2023 data")
    
    total_records = 0
    
    # Process from the cutoff date backwards if we don't have that data
    if existing_min > WEBSERVICE_CUTOFF_DATE:
        # Download from cutoff date to existing data
        process_start = WEBSERVICE_CUTOFF_DATE
        process_end = existing_min - timedelta(days=1)
        
        current_date = process_start
        while current_date <= process_end:
            # Process in monthly chunks
            month_end = min(
                datetime(current_date.year, current_date.month + 1, 1) - timedelta(days=1)
                if current_date.month < 12
                else datetime(current_date.year + 1, 1, 1) - timedelta(days=1),
                process_end
            )
            
            print(f"\n  Processing {current_date.strftime('%Y-%m')} to {month_end.strftime('%Y-%m-%d')}...")
            
            try:
                # Download DAM data
                dam_count = await downloader.download_lmp('DAM', current_date, month_end)
                print(f"    DAM: {dam_count:,} records")
                total_records += dam_count
                
                # Add delay between requests
                await asyncio.sleep(2)
                
            except Exception as e:
                print(f"    Error: {e}")
                traceback.print_exc()
            
            # Move to next month
            current_date = month_end + timedelta(days=1)
    
    print(f"\n✅ ERCOT Total: {total_records:,} new records")
    return total_records


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
    
    # Harvest data from each ISO
    results = {}
    
    # ERCOT
    try:
        ercot_count = await harvest_ercot_historical()
        results['ERCOT'] = ercot_count
    except Exception as e:
        print(f"\n❌ ERCOT failed: {e}")
        results['ERCOT'] = 0
    
    # Other ISOs would be implemented here
    print("\n⚠️  Note: CAISO, ISONE, and NYISO downloaders not yet implemented")
    
    # Final summary
    print("\n\n" + "=" * 60)
    print("📊 FINAL SUMMARY")
    print("=" * 60)
    
    total_records = sum(results.values())
    for iso, count in results.items():
        print(f"{iso}: {count:,} new records")
    
    print(f"\nTotal new records downloaded: {total_records:,}")
    
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
                print(f"  Average price: ${float(price_stats.avg_price):.2f}")
    
    print("\n✅ Historical data harvest complete!")
    
    # Set up real-time collection next
    print("\n📡 Next step: Set up real-time data collection")
    print("   Run: python realtime_collector.py")


if __name__ == "__main__":
    asyncio.run(main())
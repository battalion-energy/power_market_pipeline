#!/usr/bin/env python3
"""Harvest historical data from all ISOs."""

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
from services.data_fetcher import DataFetcher
from downloaders.base_v2 import DownloadConfig
from sqlalchemy import text


async def harvest_iso_data(iso: str, start_date: datetime, end_date: datetime, data_types: list):
    """Harvest data for a single ISO."""
    print(f"\n{'=' * 60}")
    print(f"Harvesting {iso} data from {start_date.date()} to {end_date.date()}")
    print(f"{'=' * 60}")
    
    config = DownloadConfig(
        start_date=start_date,
        end_date=end_date,
        data_types=data_types,
        output_dir=f'./data/{iso.lower()}'
    )
    
    try:
        fetcher = DataFetcher(config)
        results = await fetcher.fetch_all_data(
            isos=[iso],
            start_date=start_date,
            end_date=end_date,
            data_types=data_types
        )
        
        # Display results
        if iso in results:
            print(f"\n{iso} Results:")
            for data_type, count in results[iso].items():
                print(f"  {data_type}: {count:,} records")
        else:
            print(f"\nNo results for {iso}")
            
        return results.get(iso, {})
        
    except Exception as e:
        print(f"\nError harvesting {iso} data: {e}")
        import traceback
        traceback.print_exc()
        return {}


async def main():
    """Main harvesting function."""
    print("\n🌾 Power Market Data Harvester")
    print("Harvesting historical energy market data...\n")
    
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
    
    # Test mode: Start with 3 days of data
    end_date = datetime.now()
    start_date_3days = end_date - timedelta(days=3)
    
    # ISOs to harvest
    isos = ['ERCOT', 'CAISO', 'ISONE', 'NYISO']
    data_types = []  # We'll download DAM manually for now
    
    print(f"\n📅 Test Period: {start_date_3days.date()} to {end_date.date()} (3 days)")
    print(f"📊 Data Types: {', '.join(data_types)}")
    print(f"🌐 ISOs: {', '.join(isos)}")
    
    # Phase 1: Test with 3 days
    print("\n\n🔄 PHASE 1: Testing with 3 days of data")
    all_results = {}
    
    for iso in isos:
        results = await harvest_iso_data(iso, start_date_3days, end_date, data_types)
        all_results[iso] = results
    
    # Summary of Phase 1
    print("\n\n📊 PHASE 1 SUMMARY:")
    print("=" * 60)
    total_records = 0
    for iso, results in all_results.items():
        iso_total = sum(results.values())
        total_records += iso_total
        print(f"{iso}: {iso_total:,} total records")
        for data_type, count in results.items():
            print(f"  - {data_type}: {count:,}")
    
    print(f"\nTotal records across all ISOs: {total_records:,}")
    
    # Check database
    with get_db() as db:
        db_count = db.query(LMP).count()
        print(f"\nRecords in database: {db_count:,}")
        
        # Show sample by ISO
        from sqlalchemy import func
        iso_counts = db.query(
            LMP.iso,
            func.count(LMP.iso).label('count')
        ).group_by(LMP.iso).all()
        
        print("\nDatabase breakdown by ISO:")
        for iso, count in iso_counts:
            print(f"  {iso}: {count:,}")
    
    # If 3-day test is successful, ask to proceed
    if total_records > 0:
        print("\n✅ Phase 1 completed successfully!")
        print("\nNext steps:")
        print("1. Run with --month flag for 1 month of data")
        print("2. Run with --historical flag for data since Jan 1, 2019")
    else:
        print("\n❌ No data collected. Please check credentials and try again.")


if __name__ == "__main__":
    # Check for command line arguments
    if len(sys.argv) > 1:
        if "--month" in sys.argv:
            print("Month mode not yet implemented")
        elif "--historical" in sys.argv:
            print("Historical mode (since 2019) not yet implemented")
        else:
            asyncio.run(main())
    else:
        asyncio.run(main())

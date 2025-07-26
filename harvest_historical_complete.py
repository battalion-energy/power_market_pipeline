#!/usr/bin/env python3
"""Complete historical harvest with proper upsert handling."""

import asyncio
import os
import sys
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import init_db, get_db
from database.models_v2 import LMP
from sqlalchemy import text, func
import pandas as pd


async def download_historical_ercot():
    """Download all historical ERCOT data since Jan 1, 2019."""
    print("\n🌾 ERCOT Historical Data Harvest")
    print("=" * 60)
    
    # Initialize database
    print("\nInitializing database...")
    init_db()
    
    # Check existing data
    print("\nChecking existing data...")
    with get_db() as db:
        existing_count = db.query(LMP).filter(LMP.iso == 'ERCOT').count()
        if existing_count > 0:
            date_range = db.query(
                func.min(LMP.interval_start).label('min_date'),
                func.max(LMP.interval_start).label('max_date')
            ).filter(LMP.iso == 'ERCOT').first()
            print(f"Found {existing_count:,} existing ERCOT records")
            print(f"Date range: {date_range.min_date.date()} to {date_range.max_date.date()}")
        else:
            print("No existing ERCOT data found")
    
    # Import here to avoid circular imports
    from downloaders.ercot.webservice_client import ERCOTWebServiceClient
    
    # Create client
    client = ERCOTWebServiceClient()
    
    # Define date range
    start_date = datetime(2024, 1, 1)  # Start with 2024 since WebService only has recent data
    end_date = datetime.now()
    
    print(f"\n📅 Downloading from {start_date.date()} to {end_date.date()}")
    
    # Process in monthly chunks to avoid memory issues
    current_date = start_date
    total_records = 0
    
    while current_date < end_date:
        # Calculate chunk end (1 month)
        chunk_end = min(current_date + timedelta(days=30), end_date)
        
        print(f"\nProcessing {current_date.date()} to {chunk_end.date()}...")
        
        try:
            # Download data
            df = await client.get_dam_spp_prices(
                start_date=current_date,
                end_date=chunk_end,
                settlement_points=None  # Get all points
            )
            
            if df is not None and not df.empty:
                print(f"  Downloaded {len(df):,} records")
                
                # Process and store with upsert logic
                stored = await store_ercot_data(df)
                total_records += stored
                print(f"  Stored {stored:,} new records")
            else:
                print("  No data received")
                
        except Exception as e:
            print(f"  Error: {e}")
            # Continue with next chunk
        
        # Move to next chunk
        current_date = chunk_end + timedelta(days=1)
        
        # Add small delay to avoid rate limiting
        await asyncio.sleep(2)
    
    print(f"\n✅ Total records processed: {total_records:,}")
    
    # Final summary
    with get_db() as db:
        final_count = db.query(LMP).filter(LMP.iso == 'ERCOT').count()
        date_range = db.query(
            func.min(LMP.interval_start).label('min_date'),
            func.max(LMP.interval_start).label('max_date')
        ).filter(LMP.iso == 'ERCOT').first()
        
        print(f"\n📊 Final Database Summary:")
        print(f"Total ERCOT records: {final_count:,}")
        if date_range.min_date:
            print(f"Date range: {date_range.min_date.date()} to {date_range.max_date.date()}")
            
            # Location count
            location_count = db.query(func.count(func.distinct(LMP.location))).filter(LMP.iso == 'ERCOT').scalar()
            print(f"Unique locations: {location_count:,}")


async def store_ercot_data(df: pd.DataFrame) -> int:
    """Store ERCOT data with upsert logic to handle duplicates."""
    from database import get_db
    from database.models_v2 import Location
    from downloaders.ercot.downloader_v2 import ERCOTDownloaderV2
    
    stored_count = 0
    
    with get_db() as db:
        # Get ERCOT ISO ID
        iso_id = db.execute(text("SELECT id FROM isos WHERE code = 'ERCOT'")).scalar()
        
        # Process records
        for _, row in df.iterrows():
            try:
                # Parse timestamp
                if 'delivery_date' in row and 'hour_ending' in row:
                    delivery_date = pd.to_datetime(row['delivery_date'])
                    hour = int(row['hour_ending'].split(':')[0]) if ':' in str(row['hour_ending']) else int(row['hour_ending'])
                    timestamp = delivery_date + timedelta(hours=hour-1)
                else:
                    continue
                
                interval_start = timestamp
                interval_end = timestamp + timedelta(hours=1)
                location_id = row.get('settlement_point', '')
                
                if not location_id:
                    continue
                
                # Ensure location exists
                location = db.query(Location).filter(
                    Location.iso_id == iso_id,
                    Location.location_id == location_id
                ).first()
                
                if not location:
                    # Infer location type
                    if location_id.startswith("HB_"):
                        location_type = "hub"
                    elif location_id.startswith("LZ_"):
                        location_type = "zone"
                    else:
                        location_type = "node"
                    
                    location = Location(
                        iso_id=iso_id,
                        location_id=location_id,
                        location_name=location_id,
                        location_type=location_type
                    )
                    db.add(location)
                    db.flush()
                
                # Check if record exists
                existing = db.query(LMP).filter(
                    LMP.interval_start == interval_start,
                    LMP.iso == 'ERCOT',
                    LMP.location == location_id,
                    LMP.market == 'DAM'
                ).first()
                
                if not existing:
                    # Create new record
                    lmp_record = LMP(
                        interval_start=interval_start,
                        interval_end=interval_end,
                        iso='ERCOT',
                        location=location_id,
                        location_type=location.location_type,
                        market='DAM',
                        lmp=row.get('spp')
                    )
                    db.add(lmp_record)
                    stored_count += 1
                    
            except Exception as e:
                print(f"    Error processing record: {e}")
                continue
        
        # Commit all changes
        db.commit()
    
    return stored_count


if __name__ == "__main__":
    print("\n🌾 Power Market Historical Data Harvester")
    print("Downloading all available ERCOT data...\n")
    
    asyncio.run(download_historical_ercot())

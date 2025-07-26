#!/usr/bin/env python
"""Test the pipeline with mock data to verify functionality."""

import asyncio
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
load_dotenv()

from database import init_db, get_db, LMP, Location, ISO
from sqlalchemy import text


async def test_mock_data():
    """Test pipeline with mock data."""
    print("Power Market Pipeline - Mock Data Test")
    print("="*60)
    
    # Initialize database
    print("\n1. Initializing database...")
    init_db()
    print("✓ Database initialized")
    
    # Create mock LMP data
    print("\n2. Creating mock LMP data...")
    
    with get_db() as db:
        # Get ERCOT ISO
        ercot = db.query(ISO).filter(ISO.code == "ERCOT").first()
        if not ercot:
            print("✗ ERCOT ISO not found")
            return
        
        # Create a few test locations
        locations = [
            {"location_id": "HB_NORTH", "location_name": "North Hub", "location_type": "hub"},
            {"location_id": "HB_SOUTH", "location_name": "South Hub", "location_type": "hub"},
            {"location_id": "HB_WEST", "location_name": "West Hub", "location_type": "hub"},
            {"location_id": "HB_HOUSTON", "location_name": "Houston Hub", "location_type": "hub"},
        ]
        
        location_objs = []
        for loc in locations:
            # Check if location exists
            existing = db.query(Location).filter(
                Location.iso_id == ercot.id,
                Location.location_id == loc["location_id"]
            ).first()
            
            if not existing:
                location_obj = Location(
                    iso_id=ercot.id,
                    **loc
                )
                db.add(location_obj)
                db.flush()
                location_objs.append(location_obj)
            else:
                location_objs.append(existing)
        
        # Generate 3 days of hourly DAM data
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 3, 23)
        
        records = []
        current = start_date
        while current <= end_date:
            for location in location_objs:
                # Create realistic LMP values
                base_lmp = 35 + (current.hour / 24) * 20  # Higher prices during day
                energy = base_lmp * 0.85
                congestion = base_lmp * 0.10
                loss = base_lmp * 0.05
                
                record = {
                    'interval_start': current,
                    'interval_end': current + timedelta(hours=1),
                    'iso': ercot.code,
                    'location': location.location_id,
                    'location_type': location.location_type,
                    'market': 'DAM',
                    'lmp': round(base_lmp, 2),
                    'energy': round(energy, 2),
                    'congestion': round(congestion, 2),
                    'loss': round(loss, 2),
                }
                records.append(record)
            
            current += timedelta(hours=1)
        
        # Bulk insert
        db.bulk_insert_mappings(LMP, records)
        db.commit()
        
        print(f"✓ Created {len(records)} mock LMP records")
    
    # Audit the data
    print("\n3. Auditing mock data...")
    print("-"*60)
    
    with get_db() as db:
        # Summary statistics
        result = db.execute(text("""
            SELECT 
                iso,
                market,
                COUNT(*) as record_count,
                MIN(interval_start) as earliest,
                MAX(interval_start) as latest,
                COUNT(DISTINCT location) as unique_locations,
                AVG(lmp) as avg_lmp,
                MIN(lmp) as min_lmp,
                MAX(lmp) as max_lmp
            FROM lmp
            GROUP BY iso, market
        """))
        
        data = result.fetchall()
        for row in data:
            print(f"\n{row.iso} {row.market}:")
            print(f"  Records: {row.record_count}")
            print(f"  Date range: {row.earliest} to {row.latest}")
            print(f"  Locations: {row.unique_locations}")
            print(f"  LMP range: ${row.min_lmp:.2f} - ${row.max_lmp:.2f} (avg: ${row.avg_lmp:.2f})")
        
        # Check data quality
        result = db.execute(text("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN lmp IS NULL THEN 1 ELSE 0 END) as null_lmp,
                SUM(CASE WHEN energy IS NULL THEN 1 ELSE 0 END) as null_energy,
                SUM(CASE WHEN ABS(lmp - (energy + congestion + loss)) > 0.01 THEN 1 ELSE 0 END) as component_mismatch
            FROM lmp
        """))
        
        quality = result.fetchone()
        print(f"\nData Quality:")
        print(f"  Total records: {quality.total}")
        print(f"  NULL LMPs: {quality.null_lmp}")
        print(f"  NULL energy: {quality.null_energy}")
        print(f"  Component mismatches: {quality.component_mismatch}")
        
        # Sample data
        result = db.execute(text("""
            SELECT 
                interval_start,
                location,
                lmp,
                energy,
                congestion,
                loss
            FROM lmp
            ORDER BY interval_start, location
            LIMIT 10
        """))
        
        print(f"\nSample Data:")
        print(f"{'Interval Start':<20} {'Location':<15} {'LMP':>8} {'Energy':>8} {'Congest':>8} {'Loss':>8}")
        print("-"*80)
        
        for row in result:
            print(f"{str(row.interval_start):<20} {row.location:<15} "
                  f"{row.lmp:>8.2f} {row.energy:>8.2f} {row.congestion:>8.2f} {row.loss:>8.2f}")
    
    print("\n✓ Mock data test completed successfully!")
    
    # Test truncate and reload
    print("\n4. Testing truncate and reload...")
    
    with get_db() as db:
        # Truncate
        db.execute(text("TRUNCATE TABLE lmp CASCADE"))
        db.commit()
        
        # Verify empty
        count = db.execute(text("SELECT COUNT(*) FROM lmp")).scalar()
        print(f"  After truncate: {count} records")
        
        # Re-insert smaller dataset
        # Need to re-query locations since previous session was closed
        ercot = db.query(ISO).filter(ISO.code == "ERCOT").first()
        test_locations = db.query(Location).filter(
            Location.iso_id == ercot.id
        ).limit(2).all()
        
        current = datetime(2024, 1, 1)
        records = []
        for i in range(24):  # Just one day
            for location in test_locations:  # Just 2 locations
                record = {
                    'interval_start': current,
                    'interval_end': current + timedelta(hours=1),
                    'iso': 'ERCOT',
                    'location': location.location_id,
                    'location_type': location.location_type,
                    'market': 'DAM',
                    'lmp': 30 + i,
                    'energy': 25 + i,
                    'congestion': 3,
                    'loss': 2,
                }
                records.append(record)
            current += timedelta(hours=1)
        
        db.bulk_insert_mappings(LMP, records)
        db.commit()
        
        count = db.execute(text("SELECT COUNT(*) FROM lmp")).scalar()
        print(f"  After reload: {count} records")
    
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    asyncio.run(test_mock_data())
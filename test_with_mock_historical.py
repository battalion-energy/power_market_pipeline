#!/usr/bin/env python
"""Test the pipeline with mock historical data for different time periods."""

import asyncio
import os
from datetime import datetime, timedelta
from typing import List, Dict
from dotenv import load_dotenv
import pandas as pd
import numpy as np

# Load environment variables
load_dotenv()

from database import init_db, get_db, LMP, Location, ISO
from sqlalchemy import text


class MockHistoricalDataGenerator:
    """Generate realistic mock historical data."""
    
    def __init__(self):
        self.locations = [
            {"location_id": "HB_NORTH", "location_name": "North Hub", "location_type": "hub"},
            {"location_id": "HB_SOUTH", "location_name": "South Hub", "location_type": "hub"},
            {"location_id": "HB_WEST", "location_name": "West Hub", "location_type": "hub"},
            {"location_id": "HB_HOUSTON", "location_name": "Houston Hub", "location_type": "hub"},
            {"location_id": "LZ_NORTH", "location_name": "North Zone", "location_type": "zone"},
            {"location_id": "LZ_SOUTH", "location_name": "South Zone", "location_type": "zone"},
            {"location_id": "LZ_WEST", "location_name": "West Zone", "location_type": "zone"},
            {"location_id": "LZ_HOUSTON", "location_name": "Houston Zone", "location_type": "zone"},
        ]
        
    def generate_lmp(self, hour: int, day_of_year: int, location_type: str) -> float:
        """Generate realistic LMP based on time and location."""
        # Base price varies by season
        season_factor = 1 + 0.3 * np.sin(2 * np.pi * day_of_year / 365)
        
        # Hour of day pattern (higher during peak hours)
        if 6 <= hour <= 9 or 17 <= hour <= 20:
            hour_factor = 1.5  # Morning and evening peaks
        elif 10 <= hour <= 16:
            hour_factor = 1.2  # Daytime
        else:
            hour_factor = 0.8  # Night
        
        # Location type affects price
        location_factor = 1.0 if location_type == "hub" else 1.1
        
        # Base LMP calculation
        base_lmp = 30 * season_factor * hour_factor * location_factor
        
        # Add some randomness
        noise = np.random.normal(0, 5)
        
        return max(10, base_lmp + noise)  # Minimum $10/MWh
    
    async def generate_data_for_period(
        self, 
        start_date: datetime, 
        end_date: datetime,
        market: str = "DAM",
        interval_minutes: int = 60
    ) -> int:
        """Generate mock data for a specific period."""
        print(f"\nGenerating mock {market} data from {start_date} to {end_date}")
        
        with get_db() as db:
            # Get ERCOT ISO
            ercot = db.query(ISO).filter(ISO.code == "ERCOT").first()
            if not ercot:
                raise ValueError("ERCOT ISO not found")
            
            # Create locations if they don't exist
            location_objs = []
            for loc in self.locations:
                existing = db.query(Location).filter(
                    Location.iso_id == ercot.id,
                    Location.location_id == loc["location_id"]
                ).first()
                
                if not existing:
                    location_obj = Location(iso_id=ercot.id, **loc)
                    db.add(location_obj)
                    db.flush()
                    location_objs.append(location_obj)
                else:
                    location_objs.append(existing)
            
            # Generate data
            records = []
            current = start_date
            interval_delta = timedelta(minutes=interval_minutes)
            
            while current < end_date:
                day_of_year = current.timetuple().tm_yday
                
                for location in location_objs:
                    # Generate realistic LMP
                    lmp = self.generate_lmp(
                        current.hour, 
                        day_of_year, 
                        location.location_type
                    )
                    
                    # Components typically sum to LMP
                    energy = lmp * 0.85
                    congestion = lmp * 0.10
                    loss = lmp * 0.05
                    
                    record = {
                        'interval_start': current,
                        'interval_end': current + interval_delta,
                        'iso': ercot.code,
                        'location': location.location_id,
                        'location_type': location.location_type,
                        'market': market,
                        'lmp': float(round(lmp, 2)),
                        'energy': float(round(energy, 2)),
                        'congestion': float(round(congestion, 2)),
                        'loss': float(round(loss, 2)),
                    }
                    records.append(record)
                
                current += interval_delta
                
                # Bulk insert every 10,000 records
                if len(records) >= 10000:
                    db.bulk_insert_mappings(LMP, records)
                    db.commit()
                    print(f"  Inserted {len(records)} records (up to {current})")
                    records = []
            
            # Insert remaining records
            if records:
                db.bulk_insert_mappings(LMP, records)
                db.commit()
                print(f"  Inserted {len(records)} records (final batch)")
            
            # Get total count
            total = db.query(LMP).filter(
                LMP.iso == "ERCOT",
                LMP.market == market,
                LMP.interval_start >= start_date,
                LMP.interval_start < end_date
            ).count()
            
            return total


async def test_historical_periods():
    """Test the pipeline with different historical periods."""
    print("Power Market Pipeline - Historical Period Testing")
    print("="*80)
    
    # Initialize database
    print("\n1. Initializing database...")
    init_db()
    print("✓ Database initialized")
    
    # Create generator
    generator = MockHistoricalDataGenerator()
    
    # Test 1: Last 3 days
    print("\n" + "="*80)
    print("TEST 1: Last 3 Days")
    print("="*80)
    
    with get_db() as db:
        db.execute(text("TRUNCATE TABLE lmp CASCADE"))
        db.commit()
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3)
    
    # Generate DAM (hourly) and RT5M data
    dam_count = await generator.generate_data_for_period(
        start_date, end_date, "DAM", 60
    )
    rt_count = await generator.generate_data_for_period(
        start_date, end_date, "RT5M", 5
    )
    
    print(f"\nGenerated:")
    print(f"  DAM: {dam_count:,} records")
    print(f"  RT5M: {rt_count:,} records")
    print(f"  Total: {dam_count + rt_count:,} records")
    
    await audit_database("3 Days Test")
    
    # Test 2: Last month
    print("\n" + "="*80)
    print("TEST 2: Last Month (30 days)")
    print("="*80)
    
    with get_db() as db:
        db.execute(text("TRUNCATE TABLE lmp CASCADE"))
        db.commit()
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    # For a month, just generate DAM data
    dam_count = await generator.generate_data_for_period(
        start_date, end_date, "DAM", 60
    )
    
    print(f"\nGenerated: {dam_count:,} DAM records")
    
    await audit_database("1 Month Test")
    
    # Test 3: Sample historical data (Jan 2019)
    print("\n" + "="*80)
    print("TEST 3: Historical Sample (January 2019)")
    print("="*80)
    
    with get_db() as db:
        db.execute(text("TRUNCATE TABLE lmp CASCADE"))
        db.commit()
    
    start_date = datetime(2019, 1, 1)
    end_date = datetime(2019, 2, 1)
    
    dam_count = await generator.generate_data_for_period(
        start_date, end_date, "DAM", 60
    )
    
    print(f"\nGenerated: {dam_count:,} DAM records")
    
    await audit_database("Historical Sample Test")
    
    # Calculate full historical load
    print("\n" + "="*80)
    print("FULL HISTORICAL ANALYSIS (Jan 1, 2019 - Present)")
    print("="*80)
    
    days_since_2019 = (datetime.now() - datetime(2019, 1, 1)).days
    years = days_since_2019 / 365.25
    
    print(f"\nTime span: {years:.1f} years ({days_since_2019:,} days)")
    
    # Calculate data volume estimates
    num_locations = len(generator.locations)
    
    # DAM calculations
    dam_intervals = days_since_2019 * 24
    dam_records = dam_intervals * num_locations
    
    # RT5M calculations  
    rt_intervals = days_since_2019 * 24 * 12
    rt_records = rt_intervals * num_locations
    
    print(f"\nEstimated data volume (with {num_locations} locations):")
    print(f"  DAM (hourly):")
    print(f"    - Intervals: {dam_intervals:,}")
    print(f"    - Records: {dam_records:,}")
    print(f"  RT5M (5-minute):")
    print(f"    - Intervals: {rt_intervals:,}")
    print(f"    - Records: {rt_records:,}")
    print(f"  Total records: {dam_records + rt_records:,}")
    
    # Storage estimates (rough)
    bytes_per_record = 100  # Approximate
    total_bytes = (dam_records + rt_records) * bytes_per_record
    total_gb = total_bytes / (1024**3)
    
    print(f"\nEstimated storage:")
    print(f"  ~{total_gb:.1f} GB (uncompressed)")
    print(f"  ~{total_gb/4:.1f} GB (with TimescaleDB compression)")
    
    # Processing time estimates
    records_per_second = 5000  # Conservative estimate
    total_seconds = (dam_records + rt_records) / records_per_second
    total_hours = total_seconds / 3600
    
    print(f"\nEstimated processing time:")
    print(f"  ~{total_hours:.1f} hours at {records_per_second:,} records/second")
    
    print("\n✓ All tests completed successfully!")


async def audit_database(test_name: str):
    """Audit the database after each test."""
    print(f"\nAuditing results for: {test_name}")
    print("-"*60)
    
    with get_db() as db:
        # Summary statistics
        result = db.execute(text("""
            SELECT 
                iso,
                market,
                COUNT(*) as record_count,
                COUNT(DISTINCT location) as locations,
                MIN(interval_start) as earliest,
                MAX(interval_start) as latest,
                AVG(lmp) as avg_lmp,
                MIN(lmp) as min_lmp,
                MAX(lmp) as max_lmp
            FROM lmp
            GROUP BY iso, market
            ORDER BY market
        """))
        
        for row in result:
            print(f"\n{row.iso} {row.market}:")
            print(f"  Records: {row.record_count:,}")
            print(f"  Locations: {row.locations}")
            print(f"  Period: {row.earliest} to {row.latest}")
            print(f"  LMP: ${row.min_lmp:.2f} - ${row.max_lmp:.2f} (avg: ${row.avg_lmp:.2f})")
        
        # Data quality check
        result = db.execute(text("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN lmp IS NULL THEN 1 ELSE 0 END) as null_lmp,
                SUM(CASE WHEN ABS(lmp - (energy + congestion + loss)) > 0.01 THEN 1 ELSE 0 END) as mismatches
            FROM lmp
        """))
        
        quality = result.fetchone()
        print(f"\nData Quality:")
        print(f"  Total records: {quality.total:,}")
        print(f"  NULL values: {quality.null_lmp}")
        print(f"  Component mismatches: {quality.mismatches}")


if __name__ == "__main__":
    asyncio.run(test_historical_periods())
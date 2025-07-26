#!/usr/bin/env python
"""Test the real-time updater with mock data."""

import asyncio
import os
from datetime import datetime, timedelta
from typing import Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from database import init_db, get_db, LMP
from services.realtime_updater import RealtimeUpdater, ERCOTRealtimeUpdater
from test_with_mock_historical import MockHistoricalDataGenerator
from sqlalchemy import text


async def test_realtime_updater():
    """Test real-time updater functionality."""
    print("Power Market Pipeline - Real-Time Updater Test")
    print("="*80)
    
    # Initialize database
    print("\n1. Initializing database...")
    init_db()
    print("✓ Database initialized")
    
    # Clear data
    with get_db() as db:
        db.execute(text("TRUNCATE TABLE lmp CASCADE"))
        db.commit()
    
    # Generate some initial data (simulate existing data)
    print("\n2. Generating initial mock data...")
    generator = MockHistoricalDataGenerator()
    
    # Generate last 2 hours of data
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=2)
    
    await generator.generate_data_for_period(
        start_time, end_time, "RT5M", 5
    )
    
    print("✓ Initial data generated")
    
    # Create mock updater
    print("\n3. Creating real-time updater...")
    
    class MockRealtimeUpdater(RealtimeUpdater):
        """Mock real-time updater that generates data instead of downloading."""
        
        def __init__(self):
            super().__init__(isos=["ERCOT"], data_types=["lmp"])
            self.generator = MockHistoricalDataGenerator()
            self.update_count = 0
            
        async def _update_iso_with_polling(self, iso: str) -> Dict[str, int]:
            """Generate new mock data to simulate real updates."""
            self.update_count += 1
            
            # Get current time rounded to 5 minutes
            now = datetime.now()
            minutes = (now.minute // 5) * 5
            current_interval = now.replace(minute=minutes, second=0, microsecond=0)
            
            # Check if we already have this interval
            with get_db() as db:
                existing = db.query(LMP).filter(
                    LMP.iso == iso,
                    LMP.interval_start == current_interval,
                    LMP.market == "RT5M"
                ).count()
            
            if existing > 0:
                self.logger.info(
                    f"Data already exists for interval {current_interval}",
                    count=existing
                )
                return {"lmp_rt5m": 0}
            
            # Generate new data for current interval
            count = await self.generator.generate_data_for_period(
                current_interval,
                current_interval + timedelta(minutes=5),
                "RT5M",
                5
            )
            
            self.logger.info(
                f"Generated new data for interval {current_interval}",
                count=count,
                update_num=self.update_count
            )
            
            return {"lmp_rt5m": count}
    
    updater = MockRealtimeUpdater()
    
    # Test a few update cycles
    print("\n4. Running update cycles...")
    print("-"*60)
    
    for i in range(3):
        print(f"\nUpdate cycle {i+1}:")
        await updater._run_update_cycle()
        
        # Show current data status
        with get_db() as db:
            result = db.execute(text("""
                SELECT 
                    COUNT(*) as records,
                    MIN(interval_start) as earliest,
                    MAX(interval_start) as latest,
                    COUNT(DISTINCT interval_start) as intervals
                FROM lmp
                WHERE market = 'RT5M'
            """)).fetchone()
            
            print(f"  Total records: {result.records}")
            print(f"  Time range: {result.earliest} to {result.latest}")
            print(f"  Unique intervals: {result.intervals}")
        
        # Wait a bit before next cycle
        if i < 2:
            print("  Waiting 10 seconds...")
            await asyncio.sleep(10)
    
    # Test the scheduler
    print("\n5. Testing scheduler...")
    print("-"*60)
    
    # Start the scheduler
    await updater.start()
    
    print("Scheduler started. Waiting for next 5-minute mark...")
    print("(This would normally run continuously)")
    
    # Wait for up to 1 minute to see if it triggers
    wait_time = 60
    start_wait = datetime.now()
    
    while (datetime.now() - start_wait).total_seconds() < wait_time:
        # Check if we're at a 5-minute mark
        if datetime.now().minute % 5 == 0:
            print(f"✓ Reached 5-minute mark at {datetime.now()}")
            await asyncio.sleep(10)  # Let the scheduler run
            break
        await asyncio.sleep(1)
    
    # Stop the scheduler
    await updater.stop()
    print("✓ Scheduler stopped")
    
    # Final audit
    print("\n6. Final audit...")
    print("-"*60)
    
    with get_db() as db:
        result = db.execute(text("""
            SELECT 
                market,
                COUNT(*) as total_records,
                COUNT(DISTINCT interval_start) as unique_intervals,
                MIN(interval_start) as earliest,
                MAX(interval_start) as latest
            FROM lmp
            GROUP BY market
            ORDER BY market
        """)).fetchall()
        
        for row in result:
            duration = (row.latest - row.earliest).total_seconds() / 60
            print(f"\n{row.market}:")
            print(f"  Records: {row.total_records}")
            print(f"  Intervals: {row.unique_intervals}")
            print(f"  Duration: {duration:.0f} minutes")
            print(f"  Range: {row.earliest} to {row.latest}")
    
    print("\n✓ Real-time updater test completed!")


async def test_ercot_specific_updater():
    """Test ERCOT-specific real-time updater."""
    print("\n" + "="*80)
    print("Testing ERCOT-Specific Real-Time Updater")
    print("="*80)
    
    # Clear data
    with get_db() as db:
        db.execute(text("TRUNCATE TABLE lmp CASCADE"))
        db.commit()
    
    # Create ERCOT updater
    updater = ERCOTRealtimeUpdater()
    
    print("\nERCOT updater configuration:")
    print(f"  Polling interval: {updater.polling_interval} seconds")
    print(f"  Max attempts: {updater.max_polling_attempts}")
    print(f"  Data types: {updater.data_types}")
    
    print("\n✓ ERCOT real-time updater configured")
    print("\nNote: Actual ERCOT updates require valid API credentials")


if __name__ == "__main__":
    asyncio.run(test_realtime_updater())
    asyncio.run(test_ercot_specific_updater())